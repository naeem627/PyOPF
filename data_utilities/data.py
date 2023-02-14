"""Data structures and read/write methods for input and output data file formats

Author: Jesse Holzer, jesse.holzer@pnnl.gov
Author: Arun Veeramany, arun.veeramany@pnnl.gov
Author: Randy K Tran, randy.tran@pnnl.gov

Date: 2020-07-10

"""
# data.py
# module for input and output data
# including data structures
# and read and write functions

import csv
import math
import time
import traceback

import networkx as nx
import numpy as np

#from io import StringIO
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

try:
    from data_utilities.data_json import Sup
    from data_utilities.swsh_utils import solve_py as swsh_solve
    from data_utilities.xfmr_utils import compute_xfmr_position
except:
    from data_json import Sup
    from swsh_utils import solve_py as swsh_solve
    from xfmr_utils import compute_xfmr_position


# init_defaults_in_unused_field = True # do this anyway - it is not too big
read_unused_fields = True
write_defaults_in_unused_fields = False
write_values_in_unused_fields = True
hard_constr_tol = 1e-4 # tolerance on hard constraints, in the units of the model convention, i.e. mostly pu
gen_cost_dx_margin = 1.0e-6 # ensure that consecutive x points differ by at least this amount
gen_cost_dydx_min = 1.0e-6 # ensure that the marginal cost (i.e. cost function slope) never goes below this value ???
gen_cost_y_min = 1.0e-6 # ensure that the cost never goes below this value ???
gen_cost_ddydx_margin = 1.0e-6 # ensure that consecutive slopes differ by at least this amount
gen_cost_x_bounds_margin = 1.0e-2 # ensure that the pgen lower and upper bounds are covered by at least this amount
gen_cost_default_marginal_cost = 1.0e2 # default marginal cost (usd/mw-h) used if a cost function has an error
raise_extra_field = False # set to true to raise an exception if extra fields are encountered. This can be a problem if a comma appears in an end-of-line comment.
raise_con_quote = False # set to true to raise an exception if the con file has quotes. might as well accept this since we are rewriting the files
#gen_cost_revise = False # set to true to revise generator cost functions in the event of a problem, e.g. nonconvexity, not covering pmin, pmax, etc.
normalize_participation_factors = True # set to true to normalize the participation factors so they sum to 1
#extend_cost_functions_to_p_min_max = True # set to true to extend the first cost function segment through pmin - 1 and the last one through pmax + 1
#remove_inner_cost_function_points_nondistinct = True # set to true to remove the inner points in a cost function if they are too close
#remove_inner_cost_function_points_nonconvex = True # set to true to remove the inner points in a cost function if they violate convexity
id_str_ok_chars = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
default_branch_limit = 9999.0
remove_loads_with_pq_eq_0 = True
remove_switched_shunts_with_no_nonzero_blocks = True
do_check_line_i_lt_j = False
do_check_xfmr_i_lt_j = False
do_check_pb_nonnegative = True # cannot fix this - need to check it though
do_check_id_str_ok = True # difficult fix - need to check it though
do_check_rate_pos = False #True # fixed in scrubber
do_check_swrem_zero = False #True # fixed by scrubber
do_check_binit_in_integer_set = False # this will be difficult and generally requires MIP
do_check_bmin_le_binit_le_bmax = True # this is doable
#do_combine_switched_shunt_blocks_steps = True # generally want this to be false
do_fix_swsh_binit = True # now this sets binit to the closest feasible value of binit
do_fix_xfmr_tau_theta_init = True # sets windv1/windv2 or ang1 to closest feasible value if cod1 == 1 or == 3
max_num_ctgs = 1000000 # maximum number of contingencies
do_scrub_ctg_labels = True # set to True to replace ctg labels with anonymous strings
do_scrub_unused_long_strings = True
pg_qg_stat_mode = 1 # 0: do not scrub, 1: set pg=0 and qg=0, 2: set stat=1
swsh_binit_feas_tol = 1e-4
swsh_bmin_bmax_tol = 1e-8
#num_swsh_to_test = 194 # 193 195 # problem with 11152/01
max_swsh_n = 9 # maximum number of steps in each switched shunt block
xfmr_tau_theta_init_tol = 1e-4
EMERGENCY_CAPACITY_FACTOR = 0.1
EMERGENCY_MARGINAL_COST_FACTOR = 5.0
debug_check_tau_theta_init_feas = False
ratec_ratea_2 = False   #report error only once
ratc1_rata1 = False
default_load_marginal_cost = 8000.0
default_generator_marginal_cost = 1000.0
prior_point_pow_imbalance_tol = 1.0 # MVA

def timeit(function):
    def timed(*args, **kw):
        start_time = time.time()
        result = function(*args, **kw)
        end_time = time.time()
        print('function: {}, time: {}'.format(function.__name__, end_time - start_time))
        return result
    return timed

def alert(alert_dict):
    #for line in traceback.format_stack():
    #    print( line.strip() )
    print(alert_dict)

def parse_token(token, val_type, default=None):
    if token is None:
        return token
    val = None
    if len(token) > 0:
        if val_type == int and float(token) == int(float(token)):
            val = val_type(int(float(token)))
        else:
            val = val_type(token)
    elif default is not None:
        val = val_type(default)
    else:
        try:
            print('required field missing data, token: %s, val_type: %s' % (token, val_type))
            raise Exception('empty field not allowed')
        except Exception as e:
            traceback.print_exc()
            raise e
        #raise Exception('empty field not allowed')
    return val

def pad_row(row, new_row_len, allow_extra=False):
        
    try:
        if len(row) != new_row_len:
            if len(row) < new_row_len:
                pass
                #print('missing field, row:')
                #print(row)
                #raise Exception('missing field not allowed')
            elif len(row) > new_row_len:
                if allow_extra:
                    return row
                row = remove_end_of_line_comment_from_row(row, '/')
                if len(row) > new_row_len:
                    alert(
                        {'data_type': 'Data',
                         'error_message': 'extra field, please ensure that all rows have the correct number of fields',
                         'diagnostics': str(row)})
                    if raise_extra_field:
                        raise Exception('extra field not allowed')
        else:
            row = remove_end_of_line_comment_from_row(row, '/')
    except Exception as e:
        traceback.print_exc()
        raise e
    return row
    '''
    row_len = len(row)
    row_len_diff = new_row_len - row_len
    row_new = row
    if row_len_diff > 0:
        row_new = row + row_len_diff * ['']
    return row_new
    '''

def check_row_missing_fields(row, row_len_expected):

    try:
        if len(row) < row_len_expected:
            print('missing field, row:')
            print(row)
            raise Exception('missing field not allowed')
    except Exception as e:
        traceback.print_exc()
        raise e

def check_two_char_id_str(x):

    char_ok_alert_dict = {
        'data_type':
        'IdStr',
        'error_message':
        'id string has nonallowable characters - each character must be in ["%s"]' % ('","'.join(id_str_ok_chars)),
        'diagnostics':
        {'id': x}}
    if len(x) > 2:
        alert(
            {'data_type':
             'IdStr2Char',
             'error_message':
             'id string too long - must be 1 or 2 characters',
             'diagnostics':
             {'id': x}})
    if len(x) <= 0:
        alert(
            {'data_type':
             'IdStr2Char',
             'error_message':
             'id string too short - must be 1 or 2 characters',
             'diagnostics':
             {'id': x}})
    if len(x) == 2:
        x0 = x[0]
        x1 = x[1]
        isok = check_id_str_single_char_ok(x0)
        if not isok:
            alert(char_ok_alert_dict)
        isok = check_id_str_single_char_ok(x1)
        if not isok:
            alert(char_ok_alert_dict)
    if len(x) == 1:
        x0 = x[0]
        isok = check_id_str_single_char_ok(x0)
        if not isok:
            alert(char_ok_alert_dict)

def check_id_str_single_char_ok(x):

    if do_check_id_str_ok:
        isok = False
        if x in id_str_ok_chars:
            isok = True
    else:
        isok = True
    return isok

def remove_end_of_line_comment_from_row_first_occurence(row, end_of_line_str):

    index = [r.find(end_of_line_str) for r in row]
    len_row = len(row)
    entries_with_end_of_line_strs = [i for i in range(len_row) if index[i] > -1]
    num_entries_with_end_of_line_strs = len(entries_with_end_of_line_strs)
    if num_entries_with_end_of_line_strs > 0:
        first_entry_with_end_of_line_str = min(entries_with_end_of_line_strs)
        len_row_new = first_entry_with_end_of_line_str + 1
        row_new = [row[i] for i in range(len_row_new)]
        row_new[len_row_new - 1] = remove_end_of_line_comment(row_new[len_row_new - 1], end_of_line_str)
    else:
        row_new = [r for r in row]
    return row_new

def remove_end_of_line_comment_from_row(row, end_of_line_str):

    index = [r.find(end_of_line_str) for r in row]
    len_row = len(row)
    entries_with_end_of_line_strs = [i for i in range(len_row) if index[i] > -1]
    num_entries_with_end_of_line_strs = len(entries_with_end_of_line_strs)
    if num_entries_with_end_of_line_strs > 0:
        #last_entry_with_end_of_line_str = min(entries_with_end_of_line_strs)
        #len_row_new = last_entry_with_end_of_line_str + 1
        row_new = [r for r in row]
        #row_new = [row[i] for i in range(len_row_new)]
        for i in entries_with_end_of_line_strs:
            row_new[i] = remove_end_of_line_comment(row_new[i], end_of_line_str)
        #row_new[len_row_new - 1] = remove_end_of_line_comment(row_new[len_row_new - 1], end_of_line_str)
    else:
        #row_new = [r for r in row]
        row_new = row
    return row_new

def remove_end_of_line_comment(token, end_of_line_str):
    
    token_new = token
    index = token_new.find(end_of_line_str)
    if index > -1:
        token_new = token_new[0:index]
    return token_new

def extract_number(token):

    out = token.strip()
    out = out.split()[0]
    out = out.split(',')[0]
    out = out.split('/')[0]
    return out

def check_ctg_label_err(label, max_num):
    
    err = 0 # no error
    if not label.upper().startswith('CTG_'):
        err = 1 # does not start with "CTG_"
        return err
    len_label = len(label)
    if not label[4:].isdigit():
        err = 2 # characters after "CTG_" are not all digits
        return err
    try:
        num = get_ctg_num(label)
    except:
        err = 3 # failed to convert to integer
        return err
    if num > max_num:
        err = 4 # contingency number too large
        return err
    return err # err = 0, no error

def get_ctg_num(label):
    
    return int(label[4:])
    
def get_ctg_label_err_from_code(code):
    
    err = ''
    if code == 0:
        err = 'no error'
    elif code == 1:
        err = 'needs to start with "CTG_"'
    elif code == 2:
        err = 'characters after "CTG_" need to be digits'
    elif code == 3:
        err = 'failed to convert to integer, report to developers'
    elif code == 4:
        err = 'contingency number too large'
    else:
        err = 'unknown error code: {}'.format(code)
    return err

def scrub_unused_long_string(in_str):

    out_str = in_str
    if do_scrub_unused_long_strings:
        out_str = in_str.replace(',', '.')
    return out_str
        
class Data:
    '''In physical units, i.e. data convention, i.e. input and output data files'''

    def __init__(self):

        self.raw = Raw()
        self.con = Con()
        self.sup = Sup()

    def read(self, raw_name, sup_name, con_name):

        self.raw.read(raw_name)
        self.sup.read(sup_name)
        self.con.read(con_name)

    def write(self, raw_name, sup_name, con_name):

        self.raw.write(raw_name)
        self.sup.write(sup_name)
        self.con.write(con_name)

    def check(self):
        '''Checks Grid Optimization Competition assumptions'''
        
        self.raw.check()
        self.sup.check()
        self.con.check()
        self.check_gen_implies_cost_gen()
        self.check_cost_gen_implies_gen()
        #self.check_gen_cost_x_margin()
        #self.check_no_offline_generators_in_contingencies() # not needed for c2
        #self.check_no_offline_lines_in_contingencies() # not needed for c2
        #self.check_no_offline_transformers_in_contingencies() # not needed for c2
        self.check_no_generators_in_con_not_in_raw()
        self.check_no_branches_in_con_not_in_raw()
        self.check_no_loads_in_sup_not_in_raw()
        self.check_no_loads_in_raw_not_in_sup()
        self.check_no_lines_in_sup_not_in_raw()
        self.check_no_lines_in_raw_not_in_sup()
        self.check_no_transformers_in_sup_not_in_raw()
        self.check_no_transformers_in_raw_not_in_sup()
        self.check_generator_base_case_ramp_constraints_feasible()
        self.check_load_base_case_ramp_constraints_feasible()
        self.check_connectedness(scrub_mode=False)
        self.check_gen_cost_domain(scrub_mode=False)
        self.check_load_cost_domain(scrub_mode=False)

    def scrub(self):
        '''modifies certain data elements to meet Grid Optimization Competition assumptions'''

        #if do_combine_switched_shunt_blocks_steps:
        #    self.raw.switched_shunts_combine_blocks_steps()
        self.raw.scrub()
        #self.sup.scrub()
        self.con.scrub()
        #if gen_cost_revise:
        #    self.check_gen_cost_revise()
        #self.scrub_gen_costs()
        #self.remove_contingencies_with_offline_generators()
        #self.remove_contingencies_with_offline_lines()
        #self.remove_contingencies_with_offline_transformers()
        self.remove_contingencies_with_generators_not_in_raw()
        self.remove_contingencies_with_branches_not_in_raw()
        self.remove_loads_in_sup_not_in_raw()
        self.remove_lines_in_sup_not_in_raw()
        self.remove_transformers_in_sup_not_in_raw()
        self.remove_generators_in_sup_not_in_raw()
        self.sup.check(scrub_mode=True)
        self.check_connectedness(scrub_mode=True)
        self.check_gen_cost_domain(scrub_mode=True)
        self.check_load_cost_domain(scrub_mode=True)

    def modify(self, load_mode=None, case_sol=None):

        self.modify_load_t_min_max(mode=load_mode, case_sol=case_sol)
        #self.modify_load_t_min_max(mode='max', values=None)
        #self.modify_load_t_min_max(mode='min', values=None)
        #self.modify_load_t_min_max(mode='1', values=None)
        #self.modify_load_t_min_max(mode='given', values=None)

    def print_summary(self):

        print("buses: %u" % len(self.raw.buses))
        print("loads: %u" % len(self.raw.loads))
        print("fixed_shunts: %u" % len(self.raw.fixed_shunts))
        print("generators: %u" % len(self.raw.generators))
        print("nontransformer_branches: %u" % len(self.raw.nontransformer_branches))
        print("transformers: %u" % len(self.raw.transformers))
        print("transformer impedance correction tables: %u" % len(self.raw.transformer_impedance_correction_tables))
        print("switched_shunts: %u" % len(self.raw.switched_shunts))
        print("contingencies: %u" % len(self.con.contingencies))
        print("generator contingencies: %u" % len([e for e in self.con.contingencies.values() if len(e.generator_out_events) > 0]))
        branch_contingencies = [e for e in self.con.contingencies.values() if len(e.branch_out_events) > 0]
        print("branch contingencies: %u" % len(branch_contingencies))
        branch_contingency_events = [e.branch_out_events[0] for e in branch_contingencies]
        branch_contingency_branches = [(e.i,e.j,e.ckt) for e in branch_contingency_events]
        branch_contingency_branches = [((e[0], e[1], e[2]) if (e[0] < e[1]) else (e[1], e[0], e[2])) for e in branch_contingency_branches]
        nontransformer_branches = [(e.i, e.j, e.ckt) for e in self.raw.nontransformer_branches.values()]
        nontransformer_branches = [((e[0], e[1], e[2]) if (e[0] < e[1]) else (e[1], e[0], e[2])) for e in nontransformer_branches]
        transformers = [(e.i, e.j, e.ckt) for e in self.raw.transformers.values()]
        transformers = [((e[0], e[1], e[2]) if (e[0] < e[1]) else (e[1], e[0], e[2])) for e in transformers]
        contingency_nontransformer_branches = list(set(branch_contingency_branches).intersection(set(nontransformer_branches)))
        contingency_transformers = list(set(branch_contingency_branches).intersection(set(transformers)))
        print("nontransformer branch contingencies: %u" % len(contingency_nontransformer_branches))
        print("transformer contingencies: %u" % len(contingency_transformers))

    def check_gen_cost_domain(self, scrub_mode=False):
        
        cost_domain_tol = self.raw.case_identification.sbase * hard_constr_tol # + hard_constr_tol # todo: put in this extra?
        for r in self.raw.get_generators():
            key = (r.i, r.id)
            cblocks = self.sup.generators[r.i, r.id]['cblocks']
            cblocks_total_pmax = sum([0.0] + [b['pmax'] for b in cblocks])
            diagnostics = {
                'I': r.i,
                'ID': r.id,
                'PT': r.pt,
                'cost_pmax': cblocks_total_pmax,
                'pmax_tol': cost_domain_tol,
                'cblocks': cblocks}
            self.check_cost_domain(cblocks, key, cblocks_total_pmax, r.pt, cost_domain_tol, 'Generator', diagnostics, scrub_mode=scrub_mode)

    def check_load_cost_domain(self, scrub_mode=False):
        
        for r in self.raw.get_loads():
            cost_domain_tol = r.pl * hard_constr_tol # + hard_constr_tol # todo: put in this extra?
            key = (r.i, r.id)
            pmax = r.pl * self.sup.loads[r.i, r.id]['tmax']
            cblocks = self.sup.loads[r.i, r.id]['cblocks']
            cblocks_total_pmax = sum([0.0] + [b['pmax'] for b in cblocks])
            diagnostics = {
                'I': r.i,
                'ID': r.id,
                'PL': r.pl,
                'tmax': self.sup.loads[r.i, r.id]['tmax'],
                'pmax': pmax,
                'cost_pmax': cblocks_total_pmax,
                'tmax_tol': hard_constr_tol,
                'pmax_tol': cost_domain_tol,
                'cblocks': cblocks}
            self.check_cost_domain(cblocks, key, cblocks_total_pmax, pmax, cost_domain_tol, 'Load', diagnostics, scrub_mode=scrub_mode)

    def modify_load_t_min_max(self, mode=None, case_sol=None):
        
        deltar = self.sup.sup_jsonobj['systemparameters']['deltar']
        
        # careful in case some loads have st=0 and therefore are not in case_sol
        if mode == 'given':
            t_given = {(r.i, r.id): 0.0 for r in self.raw.get_loads()}
            #print('t_given 1: {}'.format(t_given))
            t_given.update(
                {k: case_sol.load_t[v] for k, v in case_sol.load_map.items()})
            #print('t_given 2: {}'.format(t_given))

        # load_i = list(self.load_df.i.values)
        # #load_id = map(clean_string, list(self.load_df.id.values))
        # load_id = list(self.load_df.id.values)
        # load_key = zip(load_i, load_id)
        # load_index = [self.load_map[k] for k in load_key]
        # self.load_t[:] = 0.0
        # self.load_t[load_index] = self.load_df.t.values

        for r in self.raw.get_loads():
            tmax = self.sup.loads[r.i, r.id]['tmax']
            tmin = self.sup.loads[r.i, r.id]['tmin']
            prumax = self.sup.loads[r.i, r.id]['prumax']
            prdmax = self.sup.loads[r.i, r.id]['prdmax']
            feas_t_max = tmax
            feas_t_min = tmin
            if r.pl > 0.0:
                feas_t_max = min(feas_t_max, 1.0 + deltar * prumax / r.pl)
                feas_t_min = max(feas_t_min, 1.0 - deltar * prdmax / r.pl)
                assert(feas_t_max >= feas_t_min)
            if mode == 'max':
                tfix = tmax
            elif mode == 'min':
                tfix = tmin
            elif mode == '1':
                tfix = 1.0
            elif mode is None:
                tfix = 1.0
            elif mode == 'given':
                tfix = t_given[r.i, r.id]
                #pass
                #tfix = ??
                #print('mode: {} not implemented yet'.format(mode))
                #assert(False)
            else:
                print('mode: {} not implemented yet'.format(mode))
                assert(False)
            if tfix > feas_t_max:
                tfix = feas_t_max
            if tfix < feas_t_min:
                tfix = feas_t_min
            print('i: {}, id: {}, pl: {}, tmax: {}, tmin: {}'.format(r.i, r.id, r.pl, self.sup.loads[r.i, r.id]['tmax'], self.sup.loads[r.i, r.id]['tmin']))
            self.sup.loads[r.i, r.id]['tmax'] = tfix
            self.sup.loads[r.i, r.id]['tmin'] = tfix
            print('i: {}, id: {}, pl: {}, tmax: {}, tmin: {}'.format(r.i, r.id, r.pl, self.sup.loads[r.i, r.id]['tmax'], self.sup.loads[r.i, r.id]['tmin']))

    def check_cost_domain(self, cblocks, key, cblocks_total_pmax, pmax, tol, data_type, diagnostics, scrub_mode=False):

        default_marginal_cost = 0.0
        if data_type == 'Load':
            default_marginal_cost = default_load_marginal_cost
        elif data_type == 'Generator':
            default_marginal_cost = default_generator_marginal_cost

        shortfall = pmax + tol - cblocks_total_pmax
        num_cblocks = len(cblocks)

        # check/scrub messages
        if len(cblocks) == 0:
            alert(
                {'data_type':
                     data_type,
                 'error_message': (
                        'Cost function has 0 blocks. We prefer to have at least 1 block.' + (
                            ' Apply scrubber to set a wide enough cost block with default marginal cost {}.'.format(default_marginal_cost)
                            if scrub_mode else '')),
                 'diagnostics': diagnostics})
        elif shortfall > 0.0:
            alert(
                {'data_type':
                     data_type,
                 'error_message': (
                        'Cost function domain does not cover operating range with sufficient tolerance. please ensure the upper bound of the cost function domain exceeds the device operating range by more than the required tolerance.' + (
                            ' Apply scrubber to extend the most expensive cost block.' if scrub_mode else '')),
                 'diagnostics': diagnostics})
        elif max([b['pmax'] for b in cblocks]) > pmax + tol + 2.0:
            alert(
                {'data_type':
                     data_type,
                 'error_message': (
                        'Cost function domain covers the operating range far beyond needed tolerance. We suggest to set the upper bound of the cost function domain to be a small tolerance beyond the device operating range.' + (
                            ' Apply the scrubber to truncate the cost blocks so that no single one of them covers the operating range excessively.' if scrub_mode else '')),
                 'diagnostics': diagnostics})
        else:
            # return as no scrubbing needed
            return

        if not scrub_mode:
            return

        # scrubbing needed
        
        if num_cblocks == 0:
            new_cblocks = [{'pmax': (shortfall + 1.0), 'c': default_marginal_cost}]
        elif shortfall > 0.0:
            new_cblocks = sorted(cblocks, key=(lambda x: x['c']))
            new_cblocks[num_cblocks - 1]['pmax'] += (shortfall + 1.0)
        elif max([b['pmax'] for b in cblocks]) > pmax + tol + 1:
            #new_cblocks = [{'pmax': pmax + tol + 1.0, 'c': b['c']} for b in cblocks] # bug: this resets pmax of all blocks, potentially expanding some of them
            new_cblocks = [{'pmax': min(pmax + tol + 0.5, b['pmax']), 'c': b['c']} for b in cblocks] # bug fix: this resets pmax only for blocks that have too large pmax
        
        if data_type == 'Load':
            self.sup.loads[key]['cblocks'] = new_cblocks
        elif data_type == 'Generator':
            self.sup.generators[key]['cblocks'] = new_cblocks

    def convert_to_offline(self):
        '''converts the operating point to the offline starting point'''

        self.raw.set_operating_point_to_offline_solution()

    def check_connectedness(self, scrub_mode=False):

        buses_id = [r.i for r in self.raw.get_buses()]
        buses_id = sorted(buses_id)
        num_buses = len(buses_id)
        lines_id = [(r.i, r.j, r.ckt) for r in self.raw.get_nontransformer_branches() if r.st == 1] # todo check status
        num_lines = len(lines_id)
        xfmrs_id = [(r.i, r.j, r.ckt) for r in self.raw.get_transformers() if r.stat == 1] # todo check status
        num_xfmrs = len(xfmrs_id)
        branches_id = lines_id + xfmrs_id
        num_branches = len(branches_id)
        branches_id = [(r if r[0] < r[1] else (r[1], r[0], r[2])) for r in branches_id]
        branches_id = sorted(list(set(branches_id)))
        if len(branches_id) != num_branches:
            alert(
                {'data_type':
                     'Data',
                 'error_message':
                     'Repeated branch id on a given pair of buses. this is duplicated by another check and should be caught and reported in more detail there',
                 'diagnostics': None})
        ctg_branches_id = [(e.i, e.j, e.ckt) for r in self.con.get_contingencies() for e in r.branch_out_events]
        ctg_branches_id = [(r if r[0] < r[1] else (r[1], r[0], r[2])) for r in ctg_branches_id]
        ctg_branches_id = sorted(list(set(ctg_branches_id)))
        ctg_branches_id_ctg_label_map = {
            k:[]
            for k in ctg_branches_id}
        for r in self.con.get_contingencies():
            for e in r.branch_out_events:
                if e.i < e.j:
                    k = (e.i, e.j, e.ckt)
                else:
                    k = (e.j, e.i, e.ckt)
                ctg_branches_id_ctg_label_map[k].append(r.label)
        branch_bus_pairs = sorted(list(set([(r[0], r[1]) for r in branches_id])))
        bus_pair_branches_map = {
            r:[]
            for r in branch_bus_pairs}
        for r in branches_id:
            bus_pair_branches_map[(r[0], r[1])].append(r)
        bus_pair_num_branches_map = {
            k:len(v)
            for k, v in bus_pair_branches_map.items()}
        bus_nodes_id = [
            'node_bus_{}'.format(r) for r in buses_id]
        extra_nodes_id = [
            'node_extra_{}_{}_{}'.format(r[0], r[1], r[2])
            for k in branch_bus_pairs if bus_pair_num_branches_map[k] > 1
            for r in bus_pair_branches_map[k]]
        branch_edges = [
            ('node_bus_{}'.format(r[0]), 'node_bus_{}'.format(r[1]))
            for k in branch_bus_pairs if bus_pair_num_branches_map[k] == 1
            for r in bus_pair_branches_map[k]]
        branch_edge_branch_map = {
            ('node_bus_{}'.format(r[0]), 'node_bus_{}'.format(r[1])):r
            for k in branch_bus_pairs if bus_pair_num_branches_map[k] == 1
            for r in bus_pair_branches_map[k]}            
        extra_edges_1 = [
            ('node_bus_{}'.format(r[0]), 'node_extra_{}_{}_{}'.format(r[0], r[1], r[2]))
            for k in branch_bus_pairs if bus_pair_num_branches_map[k] > 1
            for r in bus_pair_branches_map[k]]
        extra_edges_2 = [
            ('node_bus_{}'.format(r[1]), 'node_extra_{}_{}_{}'.format(r[0], r[1], r[2]))
            for k in branch_bus_pairs if bus_pair_num_branches_map[k] > 1
            for r in bus_pair_branches_map[k]]
        nodes = bus_nodes_id + extra_nodes_id
        edges = branch_edges + extra_edges_1 + extra_edges_2
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        connected_components = list(nx.connected_components(graph))
        #connected_components = [set(k) for k in connected_components] # todo get only the bus nodes and take only their id number
        num_connected_components = len(connected_components)
        if num_connected_components > 1:
            if scrub_mode:
                alert(
                    {'data_type':
                         'Data',
                     'error_message':
                         'more than one connected component in the base case unswitched system graph. This is diagnosed by the data checker but cannot be fixed by the scrubber. It must be fixed by the designer of the data set.',
                     'diagnostics': connected_components})
            else:
                alert(
                    {'data_type':
                         'Data',
                     'error_message':
                         'more than one connected component in the base case unswitched system graph.',
                     'diagnostics': connected_components})
        bridges = list(nx.bridges(graph))
        num_bridges = len(bridges)
        bridges = sorted(list(set(branch_edges).intersection(set(bridges))))
        # assert len(bridges) == num_bridges i.e. all bridges are branch edges, i.e. not extra edges. extra edges should be elements of cycles
        bridges = [branch_edge_branch_map[r] for r in bridges]
        ctg_bridges = sorted(list(set(bridges).intersection(set(ctg_branches_id))))
        num_ctg_bridges = len(ctg_bridges)
        if num_ctg_bridges > 0:
            if scrub_mode:
                alert(
                    {'data_type':
                         'Data',
                     'error_message':
                         'at least one branch outage contingency causes multiple connected components in the post contingency unswitched system graph. Scrubbing by removing contingencies causing multiple connected components.',
                     'diagnostics': ctg_bridges})
                ctgs_label_to_remove = [
                    k
                    for r in ctg_bridges
                    for k in ctg_branches_id_ctg_label_map[r]]
                ctgs_label_to_remove = sorted(list(set(ctgs_label_to_remove)))
                for k in ctgs_label_to_remove:
                    alert(
                        {'data_type':
                             'Data',
                         'error_message':
                             'removing branch contingency where the loss of the branch causes islanding in the unswitched network',
                         'diagnostics':
                             {'ctg label': k}})
                    del self.con.contingencies[k]
            else:
                alert(
                    {'data_type':
                         'Data',
                     'error_message':
                         'at least one branch outage contingency causes multiple connected components in the post contingency unswitched system graph',
                     'diagnostics': ctg_bridges})
                
    def check_gen_implies_cost_gen(self):   

        gen_set = set([(g.i, g.id) for g in self.raw.get_generators()])
        cost_gen_set = self.sup.get_generator_ids()     #you will get a unique set
        gen_not_cost_gen = gen_set.difference(cost_gen_set)
        if len(gen_not_cost_gen) > 0:
            alert(
                {'data_type':
                 'Data',
                 'error_message':
                 'Please ensure that every generator in the RAW file is also in the JSON SUP file.',
                 'diagnostics':
                 {'num gens': len(gen_not_cost_gen),
                  'gens': [
                      {'gen i': g[0], 'gen id': g[1]}
                      for g in gen_not_cost_gen]}})

    def check_cost_gen_implies_gen(self):

        gen_set = set([(g.i, g.id) for g in self.raw.get_generators()])
        cost_gen_set = self.sup.get_generator_ids()     #you will get a unique set
        cost_gen_not_gen = cost_gen_set.difference(gen_set)
        if len(cost_gen_not_gen) > 0:
            alert(
                {'data_type':
                 'Data',
                 'error_message':
                 'Please ensure that every generator in the JSON SUP file is also in the RAW file.',
                 'diagnostics':
                 {'num gens': len(cost_gen_not_gen),
                  'gens': [
                      {'gen i': g[0], 'gen id': g[1]}
                      for g in cost_gen_not_gen]}})

    def remove_generators_in_sup_not_in_raw(self):

        gen_set = set([(g.i, g.id) for g in self.raw.get_generators()])
        cost_gen_set = self.sup.get_generator_ids()     #you will get a unique set
        cost_gen_not_gen = cost_gen_set.difference(gen_set)
        if len(cost_gen_not_gen) > 0:
            alert(
                {'data_type':
                 'Data',
                 'error_message':
                 'Removing generators from the JSON SUP file that are not in the RAW file.',
                 'diagnostics':
                 {'num gens': len(cost_gen_not_gen),
                  'gens': [
                      {'gen i': g[0], 'gen id': g[1]}
                      for g in cost_gen_not_gen]}})
            self.sup.remove_generators(cost_gen_not_gen)

    def check_generator_base_case_ramp_constraints_feasible(self):

        for r in self.raw.get_generators():
            if r.stat > 0:
                feas_p_min = max(
                    r.pg -
                    self.sup.sup_jsonobj['systemparameters']['deltar'] *
                    self.sup.generators[r.i, r.id]['prdmax'],
                    r.pb)
                feas_p_max = min(
                    r.pg +
                    self.sup.sup_jsonobj['systemparameters']['deltar'] *
                    self.sup.generators[r.i, r.id]['prumax'],
                    r.pt)
                if feas_p_min > feas_p_max:
                    alert(
                        {'data_type':
                             'Data',
                         'error_message':
                             'Please ensure that every generator that is committed on in the prior operating point has a feasible operating range in the base case while remaining committed on, considering PG, PMIN, PMAX, PRUMAX, PRDMAX, DELTAR.',
                         'diagnostics':
                             {'I': r.i,
                              'ID': r.id,
                              'PG': r.pg,
                              'PMIN': r.pb,
                              'PMAX': r.pt,
                              'PRUMAX': self.sup.generators[r.i, r.id]['prumax'],
                              'PRDMAX': self.sup.generators[r.i, r.id]['prdmax'],
                              'DELTAR': self.sup.sup_jsonobj['systemparameters']['deltar']}})

    def check_load_base_case_ramp_constraints_feasible(self):

        for r in self.raw.get_loads():
            feas_p_min = max(
                r.pl -
                self.sup.sup_jsonobj['systemparameters']['deltar'] *
                self.sup.loads[r.i, r.id]['prdmax'],
                self.sup.loads[r.i, r.id]['tmin'] * r.pl)
            feas_p_max = min(
                r.pl +
                self.sup.sup_jsonobj['systemparameters']['deltar'] *
                self.sup.loads[r.i, r.id]['prumax'],
                self.sup.loads[r.i, r.id]['tmax'] * r.pl)
            if feas_p_min > feas_p_max:
                alert(
                    {'data_type':
                         'Data',
                     'error_message':
                         'Please ensure that every load has a feasible operating range in the base case, considering PL, TMIN, TMAX, PRUMAX, PRDMAX, DELTAR.',
                     'diagnostics':
                         {'I': r.i,
                          'ID': r.id,
                          'PL': r.pl,
                          'TMIN': self.sup.loads[r.i, r.id]['tmin'],
                          'TMAX': self.sup.loads[r.i, r.id]['tmax'],
                          'PRUMAX': self.sup.loads[r.i, r.id]['prumax'],
                          'PRDMAX': self.sup.loads[r.i, r.id]['prdmax'],
                          'DELTAR': self.sup.sup_jsonobj['systemparameters']['deltar']}})

    def check_no_loads_in_sup_not_in_raw(self):

        raw_load_set = set([(r.i, r.id) for r in self.raw.get_loads()])
        sup_load_set = self.sup.get_load_ids()     #you will get a unique set
        sup_not_raw_load_set = sup_load_set.difference(raw_load_set)
        if len(sup_not_raw_load_set) > 0:
            alert(
                {'data_type':
                 'Data',
                 'error_message':
                 'Please ensure that every load in the JSON SUP file is also in the RAW file.',
                 'diagnostics':
                 {'num loads sup not raw ': len(sup_not_raw_load_set),
                  'loads': [
                      {'i': r[0], 'id': r[1]}
                      for r in sup_not_raw_load_set]}})

    def remove_loads_in_sup_not_in_raw(self):

        raw_load_set = set([(r.i, r.id) for r in self.raw.get_loads()])
        sup_load_set = self.sup.get_load_ids()     #you will get a unique set
        sup_not_raw_load_set = sup_load_set.difference(raw_load_set)
        if len(sup_not_raw_load_set) > 0:
            alert(
                {'data_type':
                 'Data',
                 'error_message':
                 'Removing loads from JSON SUP file that are not in the RAW file.',
                 'diagnostics':
                 {'num loads sup not raw ': len(sup_not_raw_load_set),
                  'loads': [
                      {'i': r[0], 'id': r[1]}
                      for r in sup_not_raw_load_set]}})
            self.sup.remove_loads(sup_not_raw_load_set)
        
    def check_no_loads_in_raw_not_in_sup(self):

        raw_load_set = set([(r.i, r.id) for r in self.raw.get_loads()])
        sup_load_set = self.sup.get_load_ids()     #you will get a unique set
        raw_not_sup_load_set = raw_load_set.difference(sup_load_set)
        if len(raw_not_sup_load_set) > 0:
            alert(
                {'data_type':
                 'Data',
                 'error_message':
                 'Please ensure that every load in the RAW file is also in the JSON SUP file.',
                 'diagnostics':
                 {'num loads raw not sup ': len(raw_not_sup_load_set),
                  'loads': [
                      {'i': r[0], 'id': r[1]}
                      for r in raw_not_sup_load_set]}})

    def check_no_lines_in_sup_not_in_raw(self):

        raw_set = set([(r.i, r.j, r.ckt) for r in self.raw.get_nontransformer_branches()])
        sup_set = self.sup.get_line_ids()     #you will get a unique set
        diff_set = sup_set.difference(raw_set)
        if len(diff_set) > 0:
            alert(
                {'data_type':
                 'Data',
                 'error_message':
                 'Please ensure that every line in the JSON SUP file is also in the RAW file. Also check that I/origin and J/destination designations are consistent across all data files.',
                 'diagnostics':
                 {'num lines sup not raw ': len(diff_set),
                  'lines': [
                      {'i': r[0], 'j': r[1], 'id': r[2]}
                      for r in diff_set]}})

    def remove_lines_in_sup_not_in_raw(self):

        raw_set = set([(r.i, r.j, r.ckt) for r in self.raw.get_nontransformer_branches()])
        sup_set = self.sup.get_line_ids()     #you will get a unique set
        diff_set = sup_set.difference(raw_set)
        if len(diff_set) > 0:
            alert(
                {'data_type':
                 'Data',
                 'error_message':
                 'Removing lines from the JSON SUP file that are not in the RAW file.',
                 'diagnostics':
                 {'num lines sup not raw ': len(diff_set),
                  'lines': [
                      {'i': r[0], 'j': r[1], 'id': r[2]}
                      for r in diff_set]}})
            self.sup.remove_lines(diff_set)

    def check_no_lines_in_raw_not_in_sup(self):

        raw_set = set([(r.i, r.j, r.ckt) for r in self.raw.get_nontransformer_branches()])
        sup_set = self.sup.get_line_ids()     #you will get a unique set
        diff_set = raw_set.difference(sup_set)
        if len(diff_set) > 0:
            alert(
                {'data_type':
                 'Data',
                 'error_message':
                 'Please ensure that every line in the RAW file is also in the JSON SUP file. Also check that I/origin and J/destination designations are consistent across all data files.',
                 'diagnostics':
                 {'num lines raw not sup ': len(diff_set),
                  'lines': [
                      {'i': r[0], 'j': r[1], 'id': r[2]}
                      for r in diff_set]}})

    def check_no_transformers_in_sup_not_in_raw(self):

        raw_set = set([(r.i, r.j, r.ckt) for r in self.raw.get_transformers()])
        sup_set = self.sup.get_transformer_ids()     #you will get a unique set
        diff_set = sup_set.difference(raw_set)
        if len(diff_set) > 0:
            alert(
                {'data_type':
                 'Data',
                 'error_message':
                 'Please ensure that every transformer in the JSON SUP file is also in the RAW file. Also check that I/origin and J/destination designations are consistent across all data files.',
                 'diagnostics':
                 {'num transformers sup not raw ': len(diff_set),
                  'transformers': [
                      {'i': r[0], 'j': r[1], 'id': r[2]}
                      for r in diff_set]}})

    def remove_transformers_in_sup_not_in_raw(self):

        raw_set = set([(r.i, r.j, r.ckt) for r in self.raw.get_transformers()])
        sup_set = self.sup.get_transformer_ids()     #you will get a unique set
        diff_set = sup_set.difference(raw_set)
        if len(diff_set) > 0:
            alert(
                {'data_type':
                 'Data',
                 'error_message':
                 'Removing transformers from the JSON SUP file that are not in the RAW file.',
                 'diagnostics':
                 {'num transformers sup not raw ': len(diff_set),
                  'transformers': [
                      {'i': r[0], 'j': r[1], 'id': r[2]}
                      for r in diff_set]}})
            self.sup.remove_transformers(diff_set)

    def check_no_transformers_in_raw_not_in_sup(self):

        raw_set = set([(r.i, r.j, r.ckt) for r in self.raw.get_transformers()])
        sup_set = self.sup.get_transformer_ids()     #you will get a unique set
        diff_set = raw_set.difference(sup_set)
        if len(diff_set) > 0:
            alert(
                {'data_type':
                 'Data',
                 'error_message':
                 'Please ensure that every transformer in the RAW file is also in the JSON SUP file. Also check that I/origin and J/destination designations are consistent across all data files.',
                 'diagnostics':
                 {'num transformers raw not sup ': len(diff_set),
                  'transformers': [
                      {'i': r[0], 'j': r[1], 'id': r[2]}
                      for r in diff_set]}})

    def check_gen_costs(self):

        #todo - find total pmax over all blocks
        # check it is >= g.pt + tol
        pass

    def check_load_costs(self):

        #todo - find total pmax over all blocks
        # check it is >= r.pl*tmax + tol
        pass
    
    def scrub_gen_costs(self):

        for g in self.raw.get_generators():
            g_i = g.i
            g_id = g.id
            g_pt = g.pt
            g_pb = g.pb
        #todo - find maximum marginal cost block
        # then set pmax on that block to g.pt + 1

    def scrub_load_costs(self):
        
        #todo - find maximum marginal cost block
        # then set pmax on that block to r.pl*tmax + 1
        pass
        
    def check_no_generators_in_con_not_in_raw(self):
        '''check that no generators in the contingencies are not in the raw file.'''

        ctgs = self.con.get_contingencies()
        raw_gens = self.raw.get_generators()
        raw_gens_id = sorted(list(set([(g.i, g.id) for g in raw_gens])))
        gen_ctgs = [c for c in ctgs if len(c.generator_out_events) > 0]
        gen_ctgs_event = [c.generator_out_events[0] for c in gen_ctgs]
        con_gens_id = sorted(list(set([(g.i, g.id) for g in gen_ctgs_event])))
        con_gens_not_raw_gens = sorted(list(set(con_gens_id) - set(raw_gens_id)))
        for g in con_gens_not_raw_gens:
            alert(
                {'data_type':
                 'Data',
                 'error_message':
                 'generators mentioned in contingencies should exist in RAW file.',
                 'diagnostics':
                 {'gen i': g[0],
                  'gen id': g[1]}})

    def check_no_branches_in_con_not_in_raw(self):
        '''check that no branches in the contingencies are not in the raw file.'''

        ctgs = self.con.get_contingencies()
        raw_branches = self.raw.get_nontransformer_branches() + self.raw.get_transformers()
        raw_branches_id = sorted(list(set([(b.i, b.j, b.ckt) for b in raw_branches])))
        branch_ctgs = [c for c in ctgs if len(c.branch_out_events) > 0]
        branch_ctgs_event = [c.branch_out_events[0] for c in branch_ctgs]
        con_branches_id = sorted(list(set([(b.i, b.j, b.ckt) for b in branch_ctgs_event])))
        con_branches_not_raw_branches = sorted(list(set(con_branches_id) - set(raw_branches_id)))
        for b in con_branches_not_raw_branches:
            alert(
                {'data_type':
                 'Data',
                 'error_message':
                 'branches mentioned in contingencies should exist in RAW file. Also check that I/origin and J/destination designations are consistent across all data files.',
                 'diagnostics':
                 {'branch i': b[0],
                  'branch j': b[1],
                  'branch ckt': b[2]}})

    def check_no_offline_generators_in_contingencies(self):
        '''check that no generators that are offline in the base case
        are going out of service in a contingency'''

        gens = self.raw.get_generators()
        offline_gen_keys = [(g.i, g.id) for g in gens if not (g.stat > 0)]
        ctgs = self.con.get_contingencies()
        gen_ctgs = [c for c in ctgs if len(c.generator_out_events) > 0]
        gen_ctg_out_event_map = {
            c:c.generator_out_events[0]
            for c in gen_ctgs}
        gen_ctg_gen_key_ctg_map = {
            (v.i, v.id):k
            for k, v in gen_ctg_out_event_map.items()}
        offline_gens_outaged_in_ctgs_keys = set(offline_gen_keys) & set(gen_ctg_gen_key_ctg_map.keys())
        for g in offline_gens_outaged_in_ctgs_keys:
            gen = self.raw.generators[g]
            ctg = gen_ctg_gen_key_ctg_map[g]
            alert(
                {'data_type':
                 'Data',
                 'error_message':
                 'Please ensure that every generator that goes out of service in a contingency is in service in the base case, i.e. has stat=1.',
                 'diagnostics':
                 {'gen i': gen.i,
                  'gen id': gen.id,
                  'gen stat': gen.stat,
                  'ctg label': ctg.label,
                  'ctg gen event i': ctg.generator_out_events[0].i,
                  'ctg gen event id': ctg.generator_out_events[0].id}})

    def remove_contingencies_with_generators_not_in_raw(self):
        '''remove any contingencies where a generator that is not
        present in the RAW file is going out of service'''

        ctgs_label_to_remove = []
        gens = self.raw.get_generators()
        gens_key = [(g.i, g.id) for g in gens]
        ctgs = self.con.get_contingencies()
        gen_ctgs = [c for c in ctgs if len(c.generator_out_events) > 0]
        gen_ctg_gen_key_map = {
            c:(c.generator_out_events[0].i, c.generator_out_events[0].id)
            for c in gen_ctgs}
        gen_ctg_gens_key = list(set(gen_ctg_gen_key_map.values()))
        gens_key_missing = list(set(gen_ctg_gens_key).difference(set(gens_key)))
        num_gens = len(gens_key)
        num_gens_missing = len(gens_key_missing)
        gens_dict = {gens_key[i]:i for i in range(num_gens)}
        gens_missing_dict = {gens_key_missing[i]:(num_gens + i) for i in range(num_gens_missing)}
        gens_dict.update(gens_missing_dict)      
        ctgs_to_remove = [c for c in gen_ctgs if gens_dict[gen_ctg_gen_key_map[c]] >= num_gens]
        ctgs_label_to_remove = [c.label for c in ctgs_to_remove]
        for k in ctgs_label_to_remove:
            alert(
                {'data_type':
                 'Data',
                 'error_message':
                 'removing generator contingency where the generator does not exist in the RAW file',
                 'diagnostics':
                 {'ctg label': k}})
            del self.con.contingencies[k]

    def remove_contingencies_with_branches_not_in_raw(self):
        '''remove any contingencies where a branch that is not
        present in the RAW file is going out of service'''

        ctgs_label_to_remove = []
        lines = self.raw.get_nontransformer_branches()
        transformers = self.raw.get_transformers()
        branches_key = [(l.i, l.j, l.ckt) for l in (lines + transformers)]
        ctgs = self.con.get_contingencies()
        branch_ctgs = [c for c in ctgs if len(c.branch_out_events) > 0]
        branch_ctg_branch_key_map = {
            c:(c.branch_out_events[0].i, c.branch_out_events[0].j, c.branch_out_events[0].ckt)
            for c in branch_ctgs}
        branch_ctg_branches_key = list(set(branch_ctg_branch_key_map.values()))
        branches_key_missing = list(set(branch_ctg_branches_key).difference(set(branches_key)))
        num_branches = len(branches_key)
        num_branches_missing = len(branches_key_missing)
        branches_dict = {branches_key[i]:i for i in range(num_branches)}
        branches_missing_dict = {branches_key_missing[i]:(num_branches + i) for i in range(num_branches_missing)}
        branches_dict.update(branches_missing_dict)      
        ctgs_to_remove = [c for c in branch_ctgs if branches_dict[branch_ctg_branch_key_map[c]] >= num_branches]
        ctgs_label_to_remove = [c.label for c in ctgs_to_remove]
        for k in ctgs_label_to_remove:
            alert(
                {'data_type':
                 'Data',
                 'error_message':
                 'removing branch contingency where the branch does not exist in the RAW file',
                 'diagnostics':
                 {'ctg label': k}})
            del self.con.contingencies[k]
            
    def remove_contingencies_with_offline_generators(self):
        '''remove any contingencies where a generator that is offline in
        the base case is going out of service'''

        ctgs_label_to_remove = []
        gens = self.raw.get_generators()
        offline_gen_keys = [(g.i, g.id) for g in gens if not (g.stat > 0)]
        ctgs = self.con.get_contingencies()
        gen_ctgs = [c for c in ctgs if len(c.generator_out_events) > 0]
        gen_ctg_out_event_map = {
            c:c.generator_out_events[0]
            for c in gen_ctgs}
        gen_ctg_gen_key_ctg_map = {
            (v.i, v.id):k
            for k, v in gen_ctg_out_event_map.items()}
        offline_gens_outaged_in_ctgs_keys = set(offline_gen_keys) & set(gen_ctg_gen_key_ctg_map.keys())
        ctgs_label_to_remove = list(set(
            [gen_ctg_gen_key_ctg_map[g].label
             for g in offline_gens_outaged_in_ctgs_keys]))


        for k in ctgs_label_to_remove:
            alert(
                {'data_type':
                 'Data',
                 'error_message':
                 'removing generator contingency where the generator is out of service in the base case',
                 'diagnostics':
                 {'ctg label': k}})
            del self.con.contingencies[k]

    def check_no_offline_lines_in_contingencies(self):
        '''check that no lines (nontranformer branches) that are offline in the base case
        are going out of service in a contingency'''

        lines = self.raw.get_nontransformer_branches()
        offline_line_keys = [(g.i, g.j, g.ckt) for g in lines if not (g.st > 0)]
        ctgs = self.con.get_contingencies()
        branch_ctgs = [c for c in ctgs if len(c.branch_out_events) > 0]
        branch_ctg_out_event_map = {
            c:c.branch_out_events[0]
            for c in branch_ctgs}
        branch_ctg_branch_key_ctg_map = {
            (v.i, v.j, v.ckt):k
            for k, v in branch_ctg_out_event_map.items()}
        offline_lines_outaged_in_ctgs_keys = set(offline_line_keys) & set(branch_ctg_branch_key_ctg_map.keys())
        for g in offline_lines_outaged_in_ctgs_keys:
            line = self.raw.nontransformer_branches[g]
            ctg = branch_ctg_branch_key_ctg_map[g]
            alert(
                {'data_type':
                 'Data',
                 'error_message':
                 'Please ensure that every line (nontransformer branch) that goes out of service in a contingency is in service in the base case, i.e. has st=1.',
                 'diagnostics':
                 {'line i': line.i,
                  'line j': line.j,
                  'line ckt': line.ckt,
                  'line st': line.st,
                  'ctg label': ctg.label,
                  'ctg branch event i': ctg.branch_out_events[0].i,
                  'ctg branch event j': ctg.branch_out_events[0].j,
                  'ctg branch event ckt': ctg.branch_out_events[0].ckt}})

    def remove_contingencies_with_offline_lines(self):
        '''remove any contingencies where a line that is offline in
        the base case is going out of service'''

        ctgs_label_to_remove = []
        lines = self.raw.get_nontransformer_branches()
        offline_line_keys = [(g.i, g.j, g.ckt) for g in lines if not (g.st > 0)]
        ctgs = self.con.get_contingencies()
        branch_ctgs = [c for c in ctgs if len(c.branch_out_events) > 0]
        branch_ctg_out_event_map = {
            c:c.branch_out_events[0]
            for c in branch_ctgs}
        branch_ctg_branch_key_ctg_map = {
            (v.i, v.j, v.ckt):k
            for k, v in branch_ctg_out_event_map.items()}
        offline_lines_outaged_in_ctgs_keys = set(offline_line_keys) & set(branch_ctg_branch_key_ctg_map.keys())
        ctgs_label_to_remove = list(set(
            [branch_ctg_branch_key_ctg_map[g].label
             for g in offline_lines_outaged_in_ctgs_keys]))
        for k in ctgs_label_to_remove:
            alert(
                {'data_type':
                 'Data',
                 'error_message':
                 'removing line contingency where the line is out of service in the base case',
                 'diagnostics':
                 {'ctg label': k}})
            del self.con.contingencies[k]

    def check_no_offline_transformers_in_contingencies(self):
        '''check that no branches that are offline in the base case
        are going out of service in a contingency'''

        transformers = self.raw.get_transformers()
        offline_transformer_keys = [(g.i, g.j, g.ckt) for g in transformers if not (g.stat > 0)]
        ctgs = self.con.get_contingencies()
        branch_ctgs = [c for c in ctgs if len(c.branch_out_events) > 0]
        branch_ctg_out_event_map = {
            c:c.branch_out_events[0]
            for c in branch_ctgs}
        branch_ctg_branch_key_ctg_map = {
            (v.i, v.j, v.ckt):k
            for k, v in branch_ctg_out_event_map.items()}
        offline_transformers_outaged_in_ctgs_keys = set(offline_transformer_keys) & set(branch_ctg_branch_key_ctg_map.keys())
        for g in offline_transformers_outaged_in_ctgs_keys:
            transformer = self.raw.transformers[g]
            ctg = branch_ctg_branch_key_ctg_map[g]
            alert(
                {'data_type':
                 'Data',
                 'error_message':
                 'Please ensure that every transformer that goes out of service in a contingency is in service in the base case, i.e. has stat=1.',
                 'diagnostics':
                 {'transformer i': transformer.i,
                  'transformer j': transformer.j,
                  'transformer ckt': transformer.ckt,
                  'transformer stat': transformer.stat,
                  'ctg label': ctg.label,
                  'ctg branch event i': ctg.branch_out_events[0].i,
                  'ctg branch event j': ctg.branch_out_events[0].j,
                  'ctg branch event ckt': ctg.branch_out_events[0].ckt}})

    def remove_contingencies_with_offline_transformers(self):
        '''remove any contingencies where a transformer that is offline in
        the base case is going out of service'''

        ctgs_label_to_remove = []
        transformers = self.raw.get_transformers()
        offline_transformer_keys = [(g.i, g.j, g.ckt) for g in transformers if not (g.stat > 0)]
        ctgs = self.con.get_contingencies()
        branch_ctgs = [c for c in ctgs if len(c.branch_out_events) > 0]
        branch_ctg_out_event_map = {
            c:c.branch_out_events[0]
            for c in branch_ctgs}
        branch_ctg_branch_key_ctg_map = {
            (v.i, v.j, v.ckt):k
            for k, v in branch_ctg_out_event_map.items()}
        offline_transformers_outaged_in_ctgs_keys = set(offline_transformer_keys) & set(branch_ctg_branch_key_ctg_map.keys())
        ctgs_label_to_remove = list(set(
            [branch_ctg_branch_key_ctg_map[g].label
             for g in offline_transformers_outaged_in_ctgs_keys]))
        for k in ctgs_label_to_remove:
            alert(
                {'data_type':
                 'Data',
                 'error_message':
                 'removing transformer contingency where the transformer is out of service in the base case',
                 'diagnostics':
                 {'ctg label': k}})
            del self.con.contingencies[k]

class Raw:
    '''In physical units, i.e. data convention, i.e. input and output data files'''

    def __init__(self):

        self.case_identification = CaseIdentification()
        self.buses = {}
        self.loads = {}
        self.fixed_shunts = {}
        self.generators = {}
        self.nontransformer_branches = {}
        self.transformers = {}
        self.areas = {}
        self.transformer_impedance_correction_tables = {}
        self.switched_shunts = {}

        self.num_loads_active = 0
        #self.num_swshs_active = 0      #not needed for reading solution file

    def scrub(self):

        self.scrub_buses()
        self.scrub_loads()
        self.scrub_fixed_shunts()
        self.scrub_nontransformer_branches()
        self.check_transformer_impedance_correction_tables(scrub_mode=True)
        self.scrub_transformers()
        self.scrub_generators()
        self.scrub_switched_shunts()

    def check(self):

        self.check_case_identification()
        self.check_buses()
        self.check_loads()
        self.check_fixed_shunts()
        self.check_generators()
        self.check_nontransformer_branches()
        self.check_transformer_impedance_correction_tables(scrub_mode=False) # need to do this before transformers
        self.check_transformers()
        #self.check_areas()
        self.check_switched_shunts()
        self.check_unique_branch_per_i_j_ckt()
        # todo: check buses > 0, bus fields of loads, fixed shunts, generators, nontransformer branches, transformers, areas, switched shunts are in the set of buses
        # todo : check impedance correction tables

    def check_transformer_impedance_correction_tables(self, scrub_mode=False):

        for r in self.get_transformer_impedance_correction_tables():
            r.check(scrub_mode)

    def add_gen_emergency_capacity(self):
        '''Add emergency capacity to each generator.
        for study 1.
        increase pmax by a fixed fraction = EMERGENCY_CAPACITY_FACTOR'''

        for r in self.get_generators():
            r.add_emergency_capacity()

    def scrub_switched_shunts(self):

        self.check_switched_shunts_bus_exists(scrub_mode=True)
        self.check_switched_shunts_binit_feas(scrub_mode=True)
        if remove_switched_shunts_with_no_nonzero_blocks:
            switched_shunts = self.get_switched_shunts()
            for r in switched_shunts:
                if r.swsh_susc_count == 0:
                    del self.switched_shunts[(r.i,)]
        for r in self.get_switched_shunts():
            r.scrub()

    def check_nontransformer_branches(self):

        self.check_nontransformer_branches_buses_exist(scrub_mode=False)
        for r in self.get_nontransformer_branches():
            r.check()

    def scrub_nontransformer_branches(self):

        self.check_nontransformer_branches_buses_exist(scrub_mode=True)
        for r in self.get_nontransformer_branches():
            r.scrub()

    def check_nontransformer_branches_buses_exist(self, scrub_mode=False):

        buses = self.get_buses()
        buses_id = set([r.i for r in buses])
        components = self.get_nontransformer_branches()
        component_buses_id = set([r.i for r in components] + [r.j for r in components])
        bus_id_component_map = {i:[] for i in component_buses_id.union(buses_id)}
        for r in components:
            bus_id_component_map[r.i].append(r)
            bus_id_component_map[r.j].append(r)
        buses_id_to_remove = component_buses_id.difference(buses_id)
        components_to_remove = [
            r for i in buses_id_to_remove
            for r in bus_id_component_map[i]]
        components_to_remove_key = sorted(
            list(set([(r.i, r.j, r.ckt) for r in components_to_remove]))) # use set to prevent deleting the same branch twice if both fbus and tbus are out.
        for k in components_to_remove_key:
            alert(
                {'data_type':
                     'Raw',
                 'error_message':
                     '{} line with nonexistent bus'.format('removing' if scrub_mode else 'found'),
                 'diagnostics':
                     {'key': k}})
            if scrub_mode:
                del self.nontransformer_branches[k]

    def check_transformers(self):

        self.check_transformers_buses_exist(scrub_mode=False)
        for r in self.get_transformers():
            r.check()
        self.check_transformers_impedance_correction_table_exists(scrub_mode=False)
        self.check_transformers_impedance_correction_table_covers_control_range(scrub_mode=False)

    def scrub_transformers(self):

        self.check_transformers_buses_exist(scrub_mode=True)
        if do_fix_xfmr_tau_theta_init:
            print('scrubbing all xfmr tau/theta init values')
        for r in self.get_transformers():
            r.scrub()
        self.check_transformers_impedance_correction_table_exists(scrub_mode=True)
        self.check_transformers_impedance_correction_table_covers_control_range(scrub_mode=True)

    def check_transformers_impedance_correction_table_exists(self, scrub_mode=False):

        xfmrs = self.get_transformers()
        xfmrs = [r for r in xfmrs if r.tab1 != 0]
        tab1 = sorted(list(set([r.tab1 for r in xfmrs])))
        tab1_xfmr_map = {t:[] for t in tab1}
        for r in xfmrs:
            tab1_xfmr_map[r.tab1].append(r)
        tict_keys = self.transformer_impedance_correction_tables.keys()
        tab1_not_tict = sorted(list(set(tab1).difference(set(tict_keys))))
        xfmrs_tab1_not_tict = [r for t in tab1_not_tict for r in tab1_xfmr_map[t]]
        if len(xfmrs_tab1_not_tict) > 0:
            if not scrub_mode:
                alert(
                    {'data_type':
                         'Raw',
                     'error_message':
                         'found transformers with nonzero tab1 referring to nonexistent transformer impedance correction table. scrubber will set tab1 = 0',
                     'diagnostics': {'[(i, j, ckt, tab1)]': [(r.i, r.j, r.ckt, r.tab1) for r in xfmrs_tab1_not_tict]}})
            else:
                alert(
                    {'data_type':
                         'Raw',
                     'error_message':
                         'found transformers with nonzero tab1 referring to nonexistent transformer impedance correction table. setting tab1 = 0',
                     'diagnostics': {'[(i, j, ckt, tab1)]': [(r.i, r.j, r.ckt, r.tab1) for r in xfmrs_tab1_not_tict]}})
                for r in xfmrs_tab1_not_tict:
                    r.tab1 = 0

    def check_transformers_impedance_correction_table_covers_control_range(self, scrub_mode=False):

        pu_tol = 1e-4
        adjust = 1e-6
        rad_tol = pu_tol
        deg_tol = 180.0 / math.pi * rad_tol
        for r in self.get_transformers():
            if r.tab1 > 0 and (r.tab1 in self.transformer_impedance_correction_tables.keys()) and r.cod1 in [-3, -1, 1, 3]:
                tict = self.transformer_impedance_correction_tables[r.tab1]
                n = tict.tict_point_count
                assert(n >= 2)
                tmax = tict.t[n - 1]
                tmin = tict.t[0]
                tol = pu_tol if r.cod1 in [-1, 1] else deg_tol
                if (not (r.rma1 + tol + adjust < tmax) or not (tmin < r.rmi1 - tol - adjust)):
                    alert(
                        {'data_type':
                             'Raw',
                         'error_message':
                             'found transformer with impedance correction table not covering control range. {}'.format(
                                'adjusting impedance correction table' if scrub_mode else 'scrubber will adjust correction table'),
                         'diagnostics':
                             {'i': r.i, 'j': r.j, 'ckt': r.ckt, 'cod1': r.cod1, 'tab1': r.tab1, 'rma1': r.rma1, 'rmi1': r.rmi1,
                              'tmax': tmax, 'tmin': tmin, 'pu_tol': pu_tol, 'tol': tol, 'tol_adjust': adjust}})
                    if scrub_mode:
                        if not (r.rma1 + tol + adjust < tmax):
                            attr_name = 't{}'.format(n)
                            setattr(tict, attr_name, r.rma1 + tol + 2.0 * adjust)
                            tict.t[n - 1] = getattr(tict, attr_name)
                        if not tmin < r.rmi1 - tol - adjust:
                            tict.t1 = r.rmi1 - tol - 2.0 * adjust
                            tict.t[0] = tict.t1

    def check_transformers_buses_exist(self, scrub_mode=False):

        buses = self.get_buses()
        buses_id = set([r.i for r in buses])
        components = self.get_transformers()
        component_buses_id = set([r.i for r in components] + [r.j for r in components])
        bus_id_component_map = {i:[] for i in component_buses_id.union(buses_id)}
        for r in components:
            bus_id_component_map[r.i].append(r)
            bus_id_component_map[r.j].append(r)
        buses_id_to_remove = component_buses_id.difference(buses_id)
        components_to_remove = [
            r for i in buses_id_to_remove
            for r in bus_id_component_map[i]]
        components_to_remove_key = sorted(
            list(set([(r.i, r.j, r.ckt) for r in components_to_remove]))) # use set to prevent deleting the same branch twice if both fbus and tbus are out.
        for k in components_to_remove_key:
            alert(
                {'data_type':
                     'Raw',
                 'error_message':
                     '{} transformer with nonexistent bus'.format('removing' if scrub_mode else 'found'),
                 'diagnostics':
                     {'key': k}})
            if scrub_mode:
                del self.transformers[k]

    def check_case_identification(self):
        
        self.case_identification.check()

    def check_buses(self):

        for r in self.get_buses():
            r.check()

    def scrub_buses(self):

        self.remove_buses_with_ide_eq_4()
        for r in self.get_buses():
            r.scrub()

    def remove_buses_with_ide_eq_4(self):

        buses = self.get_buses()
        buses_id_to_remove = [r.i for r in buses if r.ide == 4]
        for i in buses_id_to_remove:
            alert(
                {'data_type':
                     'Raw',
                 'error_message':
                     'removing bus with ide == 4',
                 'diagnostics':
                     {'bus id': i}})
            del self.buses[i]

    def check_loads(self):

        self.check_loads_bus_exists(scrub_mode=False)
        for r in self.get_loads():
            r.check()

    def scrub_loads(self):

        self.check_loads_bus_exists(scrub_mode=True)
        if remove_loads_with_pq_eq_0:
            loads = self.get_loads()
            for r in loads:
                if (r.pl == 0.0) and (r.ql == 0.0):
                    alert(
                        {'data_type':
                             'Raw',
                         'error_message':
                             'removing load with pl == 0.0 and ql == 0.0',
                         'diagnostics':
                             {'i': r.i, 'id': r.id, 'pl': r.pl, 'ql': r.ql}})
                    del self.loads[r.i, r.id]
        for r in self.get_loads():
            r.scrub()

    def check_loads_bus_exists(self, scrub_mode=False):

        buses = self.get_buses()
        buses_id = set([r.i for r in buses])
        components = self.get_loads()
        component_buses_id = set([r.i for r in components])
        bus_id_component_map = {i:[] for i in component_buses_id.union(buses_id)}
        for r in components:
            bus_id_component_map[r.i].append(r)
        buses_id_to_remove = component_buses_id.difference(buses_id)
        components_to_remove = [
            r for i in buses_id_to_remove
            for r in bus_id_component_map[i]]
        components_to_remove_key = [
            (r.i, r.id) for r in components_to_remove]
        for k in components_to_remove_key:
            alert(
                {'data_type':
                     'Raw',
                 'error_message':
                     '{} load with nonexistent bus'.format('removing' if scrub_mode else 'found'),
                 'diagnostics':
                     {'key': k}})
            if scrub_mode:
                del self.loads[k]

    def check_fixed_shunts(self):

        self.check_fixed_shunts_bus_exists(scrub_mode=False)
        for r in self.get_fixed_shunts():
            r.check()

    def check_fixed_shunts_bus_exists(self, scrub_mode=False):

        buses = self.get_buses()
        buses_id = set([r.i for r in buses])
        components = self.get_fixed_shunts()
        component_buses_id = set([r.i for r in components])
        bus_id_component_map = {i:[] for i in component_buses_id.union(buses_id)}
        for r in components:
            bus_id_component_map[r.i].append(r)
        buses_id_to_remove = component_buses_id.difference(buses_id)
        components_to_remove = [
            r for i in buses_id_to_remove
            for r in bus_id_component_map[i]]
        components_to_remove_key = [
            (r.i, r.id) for r in components_to_remove]
        for k in components_to_remove_key:
            alert(
                {'data_type':
                     'Raw',
                 'error_message':
                     '{} fixed_shunt with nonexistent bus'.format('removing' if scrub_mode else 'found'),
                 'diagnostics':
                     {'key': k}})
            if scrub_mode:
                del self.fixed_shunts[k]

    def scrub_fixed_shunts(self):

        self.check_fixed_shunts_bus_exists(scrub_mode=True)
        for r in self.get_fixed_shunts():
            r.scrub()

    def check_generators(self):

        self.check_generators_bus_exists(scrub_mode=False)
        for r in self.get_generators():
            r.check()

    def scrub_generators(self):

        self.check_generators_bus_exists(scrub_mode=True)
        for r in self.get_generators():
            r.scrub()

    def check_generators_bus_exists(self, scrub_mode=False):

        buses = self.get_buses()
        buses_id = set([r.i for r in buses])
        components = self.get_generators()
        component_buses_id = set([r.i for r in components])
        bus_id_component_map = {i:[] for i in component_buses_id.union(buses_id)}
        for r in components:
            bus_id_component_map[r.i].append(r)
        buses_id_to_remove = component_buses_id.difference(buses_id)
        components_to_remove = [
            r for i in buses_id_to_remove
            for r in bus_id_component_map[i]]
        components_to_remove_key = [
            (r.i, r.id) for r in components_to_remove]
        for k in components_to_remove_key:
            alert(
                {'data_type':
                     'Raw',
                 'error_message':
                     '{} generator with nonexistent bus'.format('removing' if scrub_mode else 'found'),
                 'diagnostics':
                     {'key': k}})
            if scrub_mode:
                del self.generators[k]

    # def check_nontransformer_branches(self):

    #     self.check_nontransformer_branches_buses_exist()
    #     for r in self.get_nontransformer_branches():
    #         r.check()

    # def check_transformers(self):

    #     for r in self.get_transformers():
    #         r.check()

    # def check_areas(self):

    #     for r in self.get_areas():
    #         r.check()

    def check_switched_shunts(self):

        self.check_switched_shunts_bus_exists(scrub_mode=False)
        for r in self.get_switched_shunts():
            r.check()
        self.check_switched_shunts_binit_feas(scrub_mode=False)

    def check_switched_shunts_bus_exists(self, scrub_mode=False):

        #print('here')
        buses = self.get_buses()
        buses_id = set([r.i for r in buses])
        components = self.get_switched_shunts()
        component_buses_id = set([r.i for r in components])
        bus_id_component_map = {i:[] for i in component_buses_id.union(buses_id)}
        for r in components:
            bus_id_component_map[r.i].append(r)
        buses_id_to_remove = component_buses_id.difference(buses_id)
        components_to_remove = [
            r for i in buses_id_to_remove
            for r in bus_id_component_map[i]]
        components_to_remove_key = [
            (r.i,) for r in components_to_remove]
        for k in components_to_remove_key:
            alert(
                {'data_type':
                     'Raw',
                 'error_message':
                     '{} switched_shunt with nonexistent bus'.format('removing' if scrub_mode else 'found'),
                 'diagnostics':
                     {'key': k}})
            if scrub_mode:
                del self.switched_shunts[k]

    @timeit
    def check_switched_shunts_binit_feas(self, scrub_mode=False):

        swsh = [r for r in self.get_switched_shunts()]
        #swsh = swsh[:num_swsh_to_test] # todo remove this line
        #r0 = swsh[num_swsh_to_test - 1]
        #print(
        #    'swsh check i: {}, binit: {}, n: [{}, {}, {}, {}, {}, {}, {}, {}], b: [{}, {}, {}, {}, {}, {}, {}, {}]'.format(
        #        r0.i, r0.binit,
        #        r0.n1, r0.n2, r0.n3, r0.n4, r0.n5, r0.n6, r0.n7, r0.n8,
        #        r0.b1, r0.b2, r0.b3, r0.b4, r0.b5, r0.b6, r0.b7, r0.b8))
        numh = len(swsh)
        if numh == 0:
            return
        i = np.array([r.i for r in swsh], dtype=int)
        btar = np.array([r.binit for r in swsh], dtype=float)
        n = np.array(
            [[r.n1, r.n2, r.n3, r.n4, r.n5, r.n6, r.n7, r.n8]
             for r in swsh],
            dtype=int)
        b = np.array(
            [[r.b1, r.b2, r.b3, r.b4, r.b5, r.b6, r.b7, r.b8]
             for r in swsh],
            dtype=float)
        x = np.zeros(shape=(numh, 8), dtype=int)
        br = np.zeros(shape=(numh), dtype=float)
        br_abs = np.zeros(shape=(numh), dtype=float)
        tol = swsh_binit_feas_tol
        swsh_nmax = np.amax(n, axis=1)
        swsh_nmax_index = np.argmax(swsh_nmax)
        nmax = np.amax(n)
        swsh_solve_tol = 1e-6
        if nmax <= max_swsh_n:
            if scrub_mode:
                if do_fix_swsh_binit:
                    print('scrubbing all swsh binit values')
                    swsh_solve(btar, n, b, x, br, br_abs, swsh_solve_tol)
                    #br = btar - bnew
                    bnew = btar - br
                    for index in range(len(swsh)):
                        swsh[index].binit = bnew[index]
            else:
                swsh_solve(btar, n, b, x, br, br_abs, swsh_solve_tol)
                br_abs_argmax = np.argmax(br_abs)
                br_abs_max = br_abs[br_abs_argmax]
                if br_abs_max > tol * abs(btar[br_abs_argmax]):
                    alert(
                        {'data_type': 'Raw',
                         'error_message': 'swsh binit not feasible, up to tolerance',
                         'diagnostics':
                             {'i': i[br_abs_argmax],
                              'binit': btar[br_abs_argmax],
                              'ni': n[br_abs_argmax, :].flatten().tolist(),
                              'bi': b[br_abs_argmax, :].flatten().tolist(),
                              'tol': tol,
                              'x': x[br_abs_argmax, :].flatten().tolist(),
                              'resid': br[br_abs_argmax],
                              'abs resid': br_abs[br_abs_argmax]}})
        else:
            alert(
                {'data_type': 'Raw',
                 'error_message': 'skipping check_switched_shunts_binit_feas due to ni exceeding maximum value of %u' % max_swsh_n,
                 'diagnostics':
                     {'i': i[swsh_nmax_index],
                      'binit': btar[swsh_nmax_index],
                      'ni': n[swsh_nmax_index, :].flatten().tolist(),
                      'bi': b[swsh_nmax_index, :].flatten().tolist()}})

    def check_unique_branch_per_i_j_ckt(self):

        # check that for any line or transformer (i,j,ckt),
        # there are no other lines or transformers with key (i,j,ckt)
        # or (j,i,ckt)

          

        branch_to_normalized_id_map = {
            r: ((r.i, r.j, r.ckt) if r.i < r.j else (r.j, r.i, r.ckt))
            for r in 
                     list(self.nontransformer_branches.values()) + list(self.transformers.values())   }
        normalized_ids = list(set(branch_to_normalized_id_map.values()))
        normalized_id_num_branches = {i:0 for i in normalized_ids}
        for k, v in branch_to_normalized_id_map.items():
            normalized_id_num_branches[v] += 1
        for i in normalized_ids:
            if normalized_id_num_branches[i] > 1:
                alert(
                    {'data_type': 'Raw',
                     'error_message': 'Please ensure that for any line or transformer (i,j,ckt) there are no other lines or transformers with key (i,j,ckt) or (j,i,ckt)',
                     'diagnostics': {'branch key': i, 'num branches': normalized_id_num_branches[i]}})
            
    def set_areas_from_buses(self):
        
        area_i_set = set([b.area for b in self.buses.values()])
        def area_set_i(area, i):
            area.i = i
            return area
        self.areas = {i:area_set_i(Area(), i) for i in area_i_set}

    def get_buses(self):

        return sorted(self.buses.values(), key=(lambda r: r.i))

    def get_loads(self):

        return sorted(self.loads.values(), key=(lambda r: (r.i, r.id)))

    def get_fixed_shunts(self):

        return sorted(self.fixed_shunts.values(), key=(lambda r: (r.i, r.id)))

    def get_generators(self):

        return sorted(self.generators.values(), key=(lambda r: (r.i, r.id)))

    def get_nontransformer_branches(self):

        return sorted(self.nontransformer_branches.values(), key=(lambda r: (r.i, r.j, r.ckt)))

    def get_transformers(self):

        return sorted(self.transformers.values(), key=(lambda r: (r.i, r.j, r.k, r.ckt)))

    def get_areas(self):

        return sorted(self.areas.values(), key=(lambda r: r.i))

    def get_transformer_impedance_correction_tables(self):

        return sorted(self.transformer_impedance_correction_tables.values(), key=(lambda r: r.i))
    
    def get_switched_shunts(self):

        return sorted(self.switched_shunts.values(), key=(lambda r: r.i))

    def construct_case_identification_section(self):

        #out_str = StringIO.StringIO()
        out_str = StringIO()
        #writer = csv.writer(out_str, lineterminator="\n", quotechar="'", quoting=csv.QUOTE_NONNUMERIC)
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        if write_values_in_unused_fields:
            rows = [
                [self.case_identification.ic, self.case_identification.sbase,
                 self.case_identification.rev, self.case_identification.xfrrat,
                 self.case_identification.nxfrat, self.case_identification.basfrq],
                ["%s" % self.case_identification.record_2], # no quotes here - typical RAW file
                ["%s" % self.case_identification.record_3]]
                #["'%s'" % self.case_identification.record_2],
                #["'%s'" % self.case_identification.record_3]]
                #["''"],
                #["''"]]
        elif write_defaults_in_unused_fields:
            rows = [
                [0, self.case_identification.sbase, 33, 0, 1, 60.0],
                ["''"],
                ["''"]]
        else:
            rows = [
                [None, self.case_identification.sbase, 33, None, None, None],
                ["''"],
                ["''"]]
        writer.writerows(rows)
        return out_str.getvalue()

    def construct_bus_section(self):
        # note use quote_none and quote the strings manually
        # values of None then are written as empty fields, which is what we want

        out_str = StringIO()
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
        
        if write_values_in_unused_fields:
            rows = [
                [r.i, "'%s'" % r.name, r.baskv, r.ide, r.area, r.zone, r.owner, r.vm, r.va, r.nvhi, r.nvlo, r.evhi, r.evlo]
                #for r in self.buses.values()] # might as well sort
                for r in self.get_buses()]
        elif write_defaults_in_unused_fields:
            rows = [
                [r.i, "'            '", 0.0, 1, r.area, 1, 1, r.vm, r.va, r.nvhi, r.nvlo, r.evhi, r.evlo]
                for r in self.get_buses()]
        else:
            rows = [
                [r.i, None, None, None, r.area, None, None, r.vm, r.va, r.nvhi, r.nvlo, r.evhi, r.evlo]
                for r in self.get_buses()]

        # debugging
        # for r in rows:
        #     try:
        #         writer.writerows([r])
        #     except:
        #         print('error/exception in construct_bus_section writerows. problem row:')
        #         print(r)
        #         print('end of problem row')
        #         #raise()
        # end debugging

        writer.writerows(rows)

        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        writer.writerows([['0 / END OF BUS DATA BEGIN LOAD DATA']]) # no comma allowed without escape character
        #out_str.write('0 / END OF BUS DATA, BEGIN LOAD DATA\n')
        return out_str.getvalue()

    def construct_load_section(self):

        out_str = StringIO()
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        if write_values_in_unused_fields:
            rows = [
                [r.i, "'%s'" % r.id, r.status, r.area, r.zone, r.pl, r.ql, r.ip, r.iq, r.yp, r.yq, r.owner, r.scale, r.intrpt]
                for r in self.get_loads()]
        elif write_defaults_in_unused_fields:
            rows = [
                [r.i, "'%s'" % r.id, r.status, self.buses[r.i].area, 1, r.pl, r.ql, 0.0, 0.0, 0.0, 0.0, 1, 1, 0]
                for r in self.get_loads()]
        else:
            rows = [
                [r.i, "'%s'" % r.id, r.status, None, None, r.pl, r.ql, None, None, None, None, None, None, None]
                for r in self.get_loads()]
        writer.writerows(rows)
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        writer.writerows([['0 / END OF LOAD DATA BEGIN FIXED SHUNT DATA']])
        return out_str.getvalue()

    def construct_fixed_shunt_section(self):

        out_str = StringIO()
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        if write_values_in_unused_fields:
            rows = [
                [r.i, "'%s'" % r.id, r.status, r.gl, r.bl]
                for r in self.get_fixed_shunts()]
        elif write_defaults_in_unused_fields:
            rows = [
                [r.i, "'%s'" % r.id, r.status, r.gl, r.bl]
                for r in self.get_fixed_shunts()]
        else:
            rows = [
                [r.i, "'%s'" % r.id, r.status, r.gl, r.bl]
                for r in self.get_fixed_shunts()]
        writer.writerows(rows)
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        writer.writerows([['0 / END OF FIXED SHUNT DATA BEGIN GENERATOR DATA']])
        return out_str.getvalue()

    def construct_generator_section(self):

        out_str = StringIO()
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        if write_values_in_unused_fields:
            rows = [
                [r.i, "'%s'" % r.id, r.pg, r.qg, r.qt, r.qb,
                 r.vs, r.ireg, r.mbase, r.zr, r.zx, r.rt, r.xt, r.gtap,
                 r.stat, r.rmpct, r.pt, r.pb, r.o1, r.f1, r.o2,
                 r.f2, r.o3, r.f3, r.o4, r.f4, r.wmod, r.wpf]
                for r in self.get_generators()]
        elif write_defaults_in_unused_fields:
            rows = [
                [r.i, "'%s'" % r.id, r.pg, r.qg, r.qt, r.qb,
                 1.0, 0, self.case_identification.sbase, 0.0, 1.0, 0.0, 0.0, 1.0,
                 r.stat, 100.0, r.pt, r.pb, 1, 1.0, 0,
                 1.0, 0, 1.0, 0, 1.0, 0, 1.0]
                for r in self.get_generators()]
        else:
            rows = [
                [r.i, "'%s'" % r.id, r.pg, r.qg, r.qt, r.qb,
                 None, None, None, None, None, None, None, None,
                 r.stat, None, r.pt, r.pb, None, None, None,
                 None, None, None, None, None, None, None]
                for r in self.get_generators()]
        writer.writerows(rows)
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        writer.writerows([['0 / END OF GENERATOR DATA BEGIN BRANCH DATA']])
        return out_str.getvalue()

    def construct_nontransformer_branch_section(self):

        out_str = StringIO()
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        if write_values_in_unused_fields:
            rows = [
                [r.i, r.j, "'%s'" % r.ckt, r.r, r.x, r.b, r.ratea,
                 r.rateb, r.ratec, r.gi, r.bi, r.gj, r.bj, r.st, r.met, r.len,
                 r.o1, r.f1, r.o2, r.f2, r.o3, r.f3, r.o4, r.f4 ]
                for r in self.get_nontransformer_branches()]
        elif write_defaults_in_unused_fields:
            rows = [
                [r.i, r.j, "'%s'" % r.ckt, r.r, r.x, r.b, r.ratea,
                 0.0, r.ratec, 0.0, 0.0, 0.0, 0.0, r.st, 1, 0.0,
                 1, 1.0, 0, 1.0, 0, 1.0, 0, 1.0 ]
                for r in self.get_nontransformer_branches()]
        else:
            rows = [
                [r.i, r.j, "'%s'" % r.ckt, r.r, r.x, r.b, r.ratea,
                 None, r.ratec, None, None, None, None, r.st, None, None,
                 None, None, None, None, None, None, None, None ]
                for r in self.get_nontransformer_branches()]
        writer.writerows(rows)
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        writer.writerows([['0 / END OF BRANCH DATA BEGIN TRANSFORMER DATA']])
        return out_str.getvalue()

    def construct_transformer_section(self):

        out_str = StringIO()
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        if write_values_in_unused_fields:
            rows = [
                rr
                for r in self.get_transformers()
                for rr in [
                        [r.i, r.j, r.k, "'%s'" % r.ckt, r.cw, r.cz, r.cm,
                         r.mag1, r.mag2, r.nmetr, "'%s'" % r.name, r.stat, r.o1, r.f1,
                         r.o2, r.f2, r.o3, r.f3, r.o4, r.f4, "'%s'" % r.vecgrp],
                        [r.r12, r.x12, r.sbase12],
                        [r.windv1, r.nomv1, r.ang1, r.rata1, r.ratb1, r.ratc1,
                         r.cod1, r.cont1, r.rma1, r.rmi1, r.vma1, r.vmi1, r.ntp1, r.tab1,
                         r.cr1, r.cx1, r.cnxa1],
                        [r.windv2, r.nomv2]]]
        elif write_defaults_in_unused_fields:
            rows = [
                rr
                for r in self.get_transformers()
                for rr in [
                        [r.i, r.j, 0, "'%s'" % r.ckt, 1, 1, 1,
                         r.mag1, r.mag2, 2, "'            '", r.stat, 1, 1.0,
                         0, 1.0, 0, 1.0, 0, 1.0, "'            '"],
                        [r.r12, r.x12, self.case_identification.sbase],
                        [r.windv1, 0.0, r.ang1, r.rata1, 0.0, r.ratc1,
                         r.cod1, 0, r.rma1, r.rmi1, 1.1, 0.9, r.ntp1, r.tab1,
                         0.0, 0.0, 0.0],
                        [r.windv2, 0.0]]]
        else:
            rows = [
                rr
                for r in self.get_transformers()
                for rr in [
                        [r.i, r.j, 0, "'%s'" % r.ckt, None, None, None,
                         r.mag1, r.mag2, None, None, r.stat, None, None,
                         None, None, None, None, None, None, None],
                        [r.r12, r.x12, None],
                        [r.windv1, None, r.ang1, r.rata1, None, r.ratc1,
                         r.cod1, None, r.rma1, r.rmi1, None, None, r.ntp1, r.tab1,
                         None, None, None],
                        [r.windv2, None]]]
        writer.writerows(rows)
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        writer.writerows([['0 / END OF TRANSFORMER DATA BEGIN AREA DATA']])
        return out_str.getvalue()

    def construct_area_section(self):

        out_str = StringIO()
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        if write_values_in_unused_fields:
            rows = [
                [r.i, r.isw, r.pdes, r.ptol, "'%s'" % r.arname]
                for r in self.get_areas()]
        elif write_defaults_in_unused_fields:
            rows = [
                [1, 0, 0.0, 10.0, "'            '"]
                for r in self.get_areas()]
        else:
            rows = [
                [None, None, None, None, None]
                for r in self.get_areas()]
        writer.writerows(rows)
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        writer.writerows([['0 / END OF AREA DATA BEGIN TWO-TERMINAL DC DATA']])
        return out_str.getvalue()

    def construct_two_terminal_dc_section(self):

        out_str = StringIO()
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        writer.writerows([['0 / END OF TWO-TERMINAL DC DATA BEGIN VSC DC LINE DATA']])
        return out_str.getvalue()

    def construct_vsc_dc_section(self):

        out_str = StringIO()
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        writer.writerows([['0 / END OF VSC DC LINE DATA BEGIN IMPEDANCE CORRECTION DATA']])
        return out_str.getvalue()

    def construct_transformer_impedance_section(self):

        out_str = StringIO()
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        # if write_values_in_unused_fields:
        #     pass # rows = []?
        # elif write_defaults_in_unused_fields:
        #     pass # rows = []?
        # else:
        #     pass # rows = []?
        rows = [
            [r.i, r.t1, r.f1, r.t2, r.f2, r.t3, r.f3, r.t4, r.f4, r.t5, r.f5,
             r.t6, r.f6, r.t7, r.f7, r.t8, r.f8, r.t9, r.f9, r.t10, r.f10,
             r.t11, r.f11][0:(2*r.tict_point_count + 1)]
            for r in self.get_transformer_impedance_correction_tables()]
        writer.writerows(rows)
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        writer.writerows([['0 / END OF IMPEDANCE CORRECTION DATA BEGIN MULTI-TERMINAL DC DATA']])
        return out_str.getvalue()
    
    def construct_multi_terminal_dc_section(self):

        out_str = StringIO()
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        writer.writerows([['0 / END OF MULTI-TERMINAL DC DATA BEGIN MULTI-SECTION LINE DATA']])
        return out_str.getvalue()

    def construct_multi_section_line_section(self):

        out_str = StringIO()
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        writer.writerows([['0 / END OF MULTI-SECTION LINE DATA BEGIN ZONE DATA']])
        return out_str.getvalue()

    def construct_zone_section(self):

        out_str = StringIO()
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        writer.writerows([['0 / END OF ZONE DATA BEGIN INTER-AREA TRANSFER DATA']])
        return out_str.getvalue()

    def construct_interarea_section(self):

        out_str = StringIO()
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        writer.writerows([['0 / END OF INTER-AREA TRANSFER DATA BEGIN OWNER DATA']])
        return out_str.getvalue()

    def construct_owner_section(self):

        out_str = StringIO()
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        writer.writerows([['0 / END OF OWNER DATA BEGIN FACTS DEVICE DATA']])
        return out_str.getvalue()

    def construct_facts_section(self):

        out_str = StringIO()
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        writer.writerows([['0 / END OF FACTS DEVICE DATA BEGIN SWITCHED SHUNT DATA']])
        return out_str.getvalue()

    def construct_switched_shunt_section(self):

        out_str = StringIO()
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        if write_values_in_unused_fields:
            rows = [
                [r.i, r.modsw, r.adjm, r.stat, r.vswhi, r.vswlo, r.swrem, r.rmpct, "'%s'" % r.rmidnt, r.binit] +
                [r.n1, r.b1, r.n2, r.b2, r.n3, r.b3, r.n4, r.b4, r.n5, r.b5, r.n6, r.b6, r.n7, r.b7, r.n8, r.b8][0:(2*r.swsh_susc_count)]
                for r in self.get_switched_shunts()]
        elif write_defaults_in_unused_fields:
            rows = [
                [r.i, 1, 0, r.stat, 1.0, 1.0, 0, 100.0, "'            '", r.binit] +
                [r.n1, r.b1, r.n2, r.b2, r.n3, r.b3, r.n4, r.b4, r.n5, r.b5, r.n6, r.b6, r.n7, r.b7, r.n8, r.b8][0:(2*r.swsh_susc_count)]
                for r in self.get_switched_shunts()]
        else:
            rows = [
                [r.i, None, None, r.stat, None, None, None, None, None, r.binit] +
                [r.n1, r.b1, r.n2, r.b2, r.n3, r.b3, r.n4, r.b4, r.n5, r.b5, r.n6, r.b6, r.n7, r.b7, r.n8, r.b8][0:(2*r.swsh_susc_count)]
                for r in self.get_switched_shunts()]
        writer.writerows(rows)
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        writer.writerows([['0 / END OF SWITCHED SHUNT DATA BEGIN GNE DATA']])
        return out_str.getvalue()

    def construct_gne_section(self):

        out_str = StringIO()
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        writer.writerows([['0 / END OF GNE DATA BEGIN INDUCTION MACHINE DATA']])
        return out_str.getvalue()

    def construct_induction_section(self):

        out_str = StringIO()
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        writer.writerows([['0 / END OF INDUCTION MACHINE DATA']])
        return out_str.getvalue()

    def construct_q_record(self):

        out_str = StringIO()
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE)
        rows = [['Q']]
        writer.writerows(rows)
        return out_str.getvalue()

    def write(self, file_name):
        '''write a RAW file'''

        print(f'writing to {file_name}')

        with open(file_name, 'w') as out_file:
            out_file.write(self.construct_case_identification_section())
            out_file.write(self.construct_bus_section())
            out_file.write(self.construct_load_section())
            out_file.write(self.construct_fixed_shunt_section())
            out_file.write(self.construct_generator_section())
            out_file.write(self.construct_nontransformer_branch_section())
            out_file.write(self.construct_transformer_section())
            out_file.write(self.construct_area_section())
            out_file.write(self.construct_two_terminal_dc_section())
            out_file.write(self.construct_vsc_dc_section())
            out_file.write(self.construct_transformer_impedance_section())
            out_file.write(self.construct_multi_terminal_dc_section())
            out_file.write(self.construct_multi_section_line_section())
            out_file.write(self.construct_zone_section())
            out_file.write(self.construct_interarea_section())
            out_file.write(self.construct_owner_section())
            out_file.write(self.construct_facts_section())
            out_file.write(self.construct_switched_shunt_section())
            out_file.write(self.construct_gne_section())
            out_file.write(self.construct_induction_section())
            out_file.write(self.construct_q_record())
        
    def set_operating_point_to_offline_solution(self):

        for r in self.buses.values():
            r.vm = 1.0
            r.va = 0.0
        for r in self.generators.values():
            r.pg = 0.0
            r.qg = 0.0
        for r in self.switched_shunts.values():
            r.binit = 0.0
        
    def read(self, file_name):

        lines = []
        for line in open(file_name, 'r'):
            if line.startswith('@!')==False and "=" not in line and "RATING" not in line:
                lines.append(line)
            #lines = in_file.readlines()
        delimiter_str = ","
        quote_str = "'"
        skip_initial_space = True
        rows = csv.reader(
            lines,
            delimiter=delimiter_str,
            quotechar=quote_str,
            skipinitialspace=skip_initial_space)
        rows = [[t.strip() for t in r] for r in rows]
        self.read_from_rows(rows)
        self.set_areas_from_buses()
        
    def row_is_file_end(self, row):

        is_file_end = False
        if len(row) == 0:
            is_file_end = True
        if row[0][:1] in {'','q','Q'}:
            is_file_end = True
        return is_file_end
    
    def row_is_section_end(self, row):

        is_section_end = False
        if row[0][:1] == '0':
            is_section_end = True
        return is_section_end
        
    def read_from_rows(self, rows):

        # todo: check for duplicate keys
        # need to return at least one  duplicated key if there is one
        # so it is not enough just to check that the number of unique
        # keys is the same as the number of records
        row_num = 0
        cid_rows = rows[row_num:(row_num + 3)]
        self.case_identification.read_from_rows(rows)
        row_num += 2

        # bus section
        section_num_records = 0
        keys_with_repeats = []
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
            bus = Bus()
            bus.read_from_row(row)
            self.buses[bus.i] = bus
            section_num_records += 1
            keys_with_repeats.append(bus.i)
        if section_num_records > len(self.buses):
            repeated_keys = get_repeated_keys(keys_with_repeats)
            alert(
                {'data_type': 'Raw',
                 'error_message': 'repeated key in RAW file section: %s' % 'Bus',
                 'diagnostics': {'records': section_num_records, 'distinct keys': len(self.buses), 'repeated keys': repeated_keys}})

        # load section
        section_num_records = 0
        keys_with_repeats = []
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
            load = Load()
            load.read_from_row(row)
            self.loads[(load.i, load.id)] = load
            section_num_records += 1
            keys_with_repeats.append((load.i, load.id))
        if section_num_records > len(self.loads):
            repeated_keys = get_repeated_keys(keys_with_repeats)
            alert(
                {'data_type': 'Raw',
                 'error_message': 'repeated key in RAW file section: %s' % 'Load',
                 'diagnostics': {'records': section_num_records, 'distinct keys': len(self.loads), 'repeated keys': repeated_keys}})

        # fixed shunt section
        section_num_records = 0
        keys_with_repeats = []
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
            fixed_shunt = FixedShunt()
            fixed_shunt.read_from_row(row)
            self.fixed_shunts[(fixed_shunt.i, fixed_shunt.id)] = fixed_shunt
            section_num_records += 1
            keys_with_repeats.append((fixed_shunt.i, fixed_shunt.id))
        if section_num_records > len(self.fixed_shunts):
            repeated_keys = get_repeated_keys(keys_with_repeats)
            alert(
                {'data_type': 'Raw',
                 'error_message': 'repeated key in RAW file section: %s' % 'FixedShunt',
                 'diagnostics': {'records': section_num_records, 'distinct keys': len(self.fixed_shunts), 'repeated keys': repeated_keys}})

        # generator section
        section_num_records = 0
        keys_with_repeats = []
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
            generator = Generator()
            generator.read_from_row(row)
            self.generators[(generator.i, generator.id)] = generator
            section_num_records += 1
            keys_with_repeats.append((generator.i, generator.id))
        if section_num_records > len(self.generators):
            repeated_keys = get_repeated_keys(keys_with_repeats)
            alert(
                {'data_type': 'Raw',
                 'error_message': 'repeated key in RAW file section: %s' % 'Generator',
                 'diagnostics': {'records': section_num_records, 'distinct keys': len(self.generators), 'repeated keys': repeated_keys}})

        # nontransformer branch section
        section_num_records = 0
        keys_with_repeats = []
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
            nontransformer_branch = NontransformerBranch()
            nontransformer_branch.read_from_row(row)
            self.nontransformer_branches[(
                nontransformer_branch.i,
                nontransformer_branch.j,
                nontransformer_branch.ckt)] = nontransformer_branch
            section_num_records += 1
            keys_with_repeats.append((
                nontransformer_branch.i,
                nontransformer_branch.j,
                nontransformer_branch.ckt))
        if section_num_records > len(self.nontransformer_branches):
            repeated_keys = get_repeated_keys(keys_with_repeats)
            alert(
                {'data_type': 'Raw',
                 'error_message': 'repeated key in RAW file section: %s' % 'NontransformerBranch',
                 'diagnostics': {'records': section_num_records, 'distinct keys': len(self.nontransformer_branches), 'repeated keys': repeated_keys}})

        # transformer section
        section_num_records = 0
        keys_with_repeats = []
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
            transformer = Transformer()
            num_rows = transformer.get_num_rows_from_row(row)
            rows_temp = rows[
                row_num:(row_num + num_rows)]
            transformer.read_from_rows(rows_temp)
            self.transformers[(
                transformer.i,
                transformer.j,
                #transformer.k,
                #0, # leave k out
                transformer.ckt)] = transformer
            row_num += (num_rows - 1)
            section_num_records += 1
            keys_with_repeats.append((transformer.i, transformer.j, transformer.ckt))
        if section_num_records > len(self.transformers):
            repeated_keys = get_repeated_keys(keys_with_repeats)
            alert(
                {'data_type': 'Raw',
                 'error_message': 'repeated key in RAW file section: %s' % 'Transformer',
                 'diagnostics': {'records': section_num_records, 'distinct keys': len(self.transformers), 'repeated keys': repeated_keys}})

        # skip section
        #section_num_records = 0
        while True: # areas
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
            #area = Area()
            #area.read_from_row(row)
            #self.areas[area.i] = area
            #section_num_records += 1
        # if section_num_records > len(self.loads):
        #     alert(
        #         {'data_type': 'Raw',
        #          'error_message': 'repeated key in RAW file section: %s' % 'Load',
        #          'diagnostics': {'records': section_num_records, 'distinct keys': len(self.loads)}})

        # skip section
        while True: # two-terminal DC transmission line data
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break

        # skip section
        while True: # voltage source converter (VSC) DC transmission line data
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break

        # transformer impedance correction tables section
        section_num_records = 0
        keys_with_repeats = []
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
            tict = TransformerImpedanceCorrectionTable()
            tict.read_from_row(row)
            self.transformer_impedance_correction_tables[tict.i] = tict
            section_num_records += 1
            keys_with_repeats.append(tict.i)
        if section_num_records > len(self.transformer_impedance_correction_tables):
            repeated_keys = get_repeated_keys(keys_with_repeats)
            alert(
                {'data_type': 'Raw',
                 'error_message': 'repeated key in RAW file section: %s' % 'TransformerImpedanceCorrectionTable',
                 'diagnostics': {'records': section_num_records, 'distinct keys': len(self.transformer_impedance_correction_tables), 'repeated keys': repeated_keys}})

        # skip section
        while True: # zone
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break

        # skip section
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break

        # skip section
        while True: # zone
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break

        # skip section
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break

        # skip section
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break

        # skip section
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break

        # switched shunt section
        section_num_records = 0
        keys_with_repeats = []
        while True:
            row_num += 1
            row = rows[row_num]
            if self.row_is_file_end(row):
                return
            if self.row_is_section_end(row):
                break
            switched_shunt = SwitchedShunt()
            switched_shunt.read_from_row(row)
            self.switched_shunts[(switched_shunt.i,)] = switched_shunt
            section_num_records += 1
            keys_with_repeats.append((switched_shunt.i,))
        if section_num_records > len(self.switched_shunts):
            repeated_keys = get_repeated_keys(keys_with_repeats)
            alert(
                {'data_type': 'Raw',
                 'error_message': 'repeated key in RAW file section: %s' % 'SwitchedShunt',
                 'diagnostics': {'records': section_num_records, 'distinct keys': len(self.switched_shunts), 'repeated keys': repeated_keys}})
        
        self.active_loads = dict(filter(lambda load: load[1].status >0, self.loads.items()))
        self.num_loads_active =   len(self.active_loads)
               
        self.active_swsh = dict(filter(lambda swsh: swsh[1].stat >0, self.switched_shunts.items()))
        self.num_swsh_active =   len(self.active_swsh)

    def skip_section(self):

        # todo
        pass

class Con:
    '''In physical units, i.e. data convention, i.e. input and output data files'''

    def __init__(self):

        self.contingencies = {}

    def check(self):

        self.check_too_many_contingencies()
        for r in self.get_contingencies():
            r.check()
        self.check_for_duplicate_outaged_generators(scrub_mode=False)
        self.check_for_duplicate_outaged_branches(scrub_mode=False)
        self.scrub_ctg_labels(scrub_mode=False)

    def scrub(self):

        self.check_for_duplicate_outaged_generators(scrub_mode=True)
        self.check_for_duplicate_outaged_branches(scrub_mode=True)
        self.scrub_ctg_labels(scrub_mode=True)

    def check_too_many_contingencies(self):

        ctgs = self.get_contingencies()
        num_ctgs = len(ctgs)
        if num_ctgs > max_num_ctgs:
            alert(
                {'data_type': 'Con',
                 'error_message': 'too many contingencies',
                 'diagnostics': {'num_ctgs': num_ctgs, 'max_num_ctgs': max_num_ctgs}})

    def scrub_ctg_labels(self, scrub_mode=False):

        #print('debug in scrub_ctg_labels. scrub_mode: {}'.format(scrub_mode))
        if do_scrub_ctg_labels:
            max_ctg_num = max_num_ctgs - 1
            num_ctg_digits = len(str(max_ctg_num))
            ctg = self.get_contingencies()
            num_ctg = len(ctg)
            #print('debug do_scrub_ctg_labels: {}, max_num_ctgs: {}, max_ctg_num: {}, num_ctg_digits: {}, num_ctg: {}'.format(
            #    do_scrub_ctg_labels, max_num_ctgs, max_ctg_num, num_ctg_digits, num_ctg))
            ctg_label = [c.label for c in ctg]
            #print('debug ctg_label: {}'.format(ctg_label))
            ctg_label_err = [check_ctg_label_err(l, max_ctg_num) for l in ctg_label]
            #print('debug ctg_label_err: {}'.format(ctg_label_err))
            ctg_num = [
                (get_ctg_num(ctg_label[i])
                 if ctg_label_err[i] == 0
                 else None)
                for i in range(num_ctg)]
            #print('debug ctg_num: {}'.format(ctg_num))
            ctg_num_in_use = sorted(list(set([ctg_num[i] for i in range(num_ctg) if ctg_label_err[i] == 0])))
            #print('debug ctg_num_in_use: {}'.format(ctg_num_in_use))
            ctg_num_not_in_use = sorted(list(set(range(num_ctg)).difference(set(ctg_num_in_use))))
            #print('debug ctg_num_not_in_use: {}'.format(ctg_num_not_in_use))
            indices_of_ctgs_with_err = [i for i in range(num_ctg) if ctg_label_err[i] > 0]
            #print('debug indices_of_ctgs_with_err: {}'.format(indices_of_ctgs_with_err))
            if len(indices_of_ctgs_with_err) > 0:
                i0 = indices_of_ctgs_with_err[0]
                if scrub_mode:
                    error_message = 'anonymizing contingency labels'
                else:
                    error_message = 'apply scrubber to anonymize contingency labels'
                alert(
                    {'data_type': 'Con',
                     'error_message': error_message,
                     'diagnostics': {
                         'num_ctgs_with_label_errs': len(indices_of_ctgs_with_err),
                         'example': {
                             'label': ctg_label[i0],
                             'error': get_ctg_label_err_from_code(ctg_label_err[i0])}}})
            if scrub_mode:
                if len(indices_of_ctgs_with_err) > 0:
                    counter = 0
                    label_format_str = 'CTG_%0' + str(num_ctg_digits) + 'u'
                    for i in indices_of_ctgs_with_err:
                        num = ctg_num_not_in_use[counter]
                        ctg_label[i] = (label_format_str % num)
                        counter += 1
                    #print('debug ctg_label: {}'.format(ctg_label))
                    self.contingencies = {ctg_label[i]:ctg[i] for i in range(num_ctg)}
                    for k, v in self.contingencies.items():
                        v.label = k
                    
    def check_for_duplicate_outaged_generators(self, scrub_mode=False):
        '''Each contingency outages exactly one device, either a generator or a branch.
        This function checks that no two generator contingencies outage the same generator.
        With scrub_mode=True it modifies the contingencies by removing any
        that outage a generator already outaged by a previously seen contingency.'''

        ctgs = self.get_contingencies()
        ctgs_to_remove = []
        ctgs = [c for c in ctgs if len(c.generator_out_events) > 0] # filter down to just gen ctgs
        num_ctgs = len(ctgs)
        if num_ctgs < 2:
            return
        ctgs_key_map = {c:(c.generator_out_events[0].i, c.generator_out_events[0].id) for c in ctgs}
        ctgs_sorted = sorted(ctgs, key=(lambda c: ctgs_key_map[c]))
        i = 0
        c_pre = ctgs_sorted[i]
        k_pre = ctgs_key_map[c_pre]
        i += 1 # next one to look at
        while i < num_ctgs:
            c = ctgs_sorted[i]
            k = ctgs_key_map[c]
            if k == k_pre:
                ctgs_to_remove.append((c_pre, c))
            else:
                c_pre = c
                k_pre = k
            i += 1
        if len(ctgs_to_remove) > 0:
            if scrub_mode:
                alert(
                    {'data_type':
                         'Con',
                     'error_message':
                         'Removing generator contingencies where the outaged device is the same as in a previously seen contingency',
                     'diagnostics':
                         {'[(previous ctg label, duplicate ctg label, device key) for all duplicates]':
                              [(c[0].label, c[1].label, ctgs_key_map[c[1]]) for c in ctgs_to_remove]}})
                for c in ctgs_to_remove:
                    del self.contingencies[c[1].label]
            else:
                alert(
                    {'data_type':
                         'Con',
                     'error_message':
                         'Found generator contingencies where the outaged device is the same as in a previously seen contingency',
                     'diagnostics':
                         {'[(previous ctg label, duplicate ctg label, device key) for all duplicates]':
                              [(c[0].label, c[1].label, ctgs_key_map[c[1]]) for c in ctgs_to_remove]}})

    def check_for_duplicate_outaged_branches(self, scrub_mode=False):
        '''Each contingency outages exactly one device, either a generator or a branch.
        This function checks that no two generator contingencies outage the same generator.
        With scrub_mode=True it modifies the contingencies by removing any
        that outage a generator already outaged by a previously seen contingency.'''

        ctgs = self.get_contingencies()
        ctgs_to_remove = []
        ctgs = [c for c in ctgs if len(c.branch_out_events) > 0] # filter down to just br ctgs
        num_ctgs = len(ctgs)
        if num_ctgs < 2:
            return
        ctgs_key_map = {c:(c.branch_out_events[0].i, c.branch_out_events[0].j, c.branch_out_events[0].ckt) for c in ctgs}
        ctgs_key_map = {k:((v[0], v[1], v[2]) if (v[0] < v[1]) else (v[1], v[0], v[2])) for k, v in ctgs_key_map.items()}
        ctgs_sorted = sorted(ctgs, key=(lambda c: ctgs_key_map[c]))
        i = 0
        c_pre = ctgs_sorted[i]
        k_pre = ctgs_key_map[c_pre]
        i += 1 # next one to look at
        while i < num_ctgs:
            c = ctgs_sorted[i]
            k = ctgs_key_map[c]
            if k == k_pre:
                ctgs_to_remove.append((c_pre, c))
            else:
                c_pre = c
                k_pre = k
            i += 1
        if len(ctgs_to_remove) > 0:
            if scrub_mode:
                alert(
                    {'data_type':
                         'Con',
                     'error_message':
                         'Removing branch contingencies where the outaged device is the same as in a previously seen contingency',
                     'diagnostics':
                         {'[(previous ctg label, duplicate ctg label, device key) for all duplicates]':
                              [(c[0].label, c[1].label, ctgs_key_map[c[1]]) for c in ctgs_to_remove]}})
                for c in ctgs_to_remove:
                    del self.contingencies[c[1].label]
            else:
                alert(
                    {'data_type':
                         'Con',
                     'error_message':
                         'Found branch contingencies where the outaged device is the same as in a previously seen contingency',
                     'diagnostics':
                         {'[(previous ctg label, duplicate ctg label, device key) for all duplicates]':
                              [(c[0].label, c[1].label, ctgs_key_map[c[1]]) for c in ctgs_to_remove]}})

    '''
    def read_from_phase_0(self, file_name):
        #takes the contingency.csv file as input
        with open(file_name, 'r') as in_file:
            lines = in_file.readlines()
        delimiter_str = " "
        quote_str = "'"
        skip_initial_space = True
        del lines[0]
        rows = csv.reader(
            lines,
            delimiter=delimiter_str,
            quotechar=quote_str,
            skipinitialspace=skip_initial_space)
        rows = [[t.strip() for t in r] for r in rows]
        quote_str = "'"
        contingency = Contingency()
        #there is no contingency label for continency.csv
        for r in rows:
            tmprow=r[0].split(',')
            if tmprow[1].upper()=='B' or tmprow[1].upper()=='T':
                contingency.label ="LINE-"+tmprow[2]+"-"+tmprow[3]+"-"+tmprow[4]
                branch_out_event = BranchOutEvent()
                branch_out_event.read_from_csv(tmprow)
                contingency.branch_out_events.append(branch_out_event)
                self.contingencies[contingency.label] = branch_out_event
            elif tmprow[1].upper()=='G':
                contingency.label = "GEN-"+tmprow[2]+"-"+tmprow[3]
                generator_out_event = GeneratorOutEvent()
                generator_out_event.read_from_csv(tmprow)
                contingency.generator_out_events.append(generator_out_event)
                self.contingency.generator_out_event.read_from_csv(tmprow)
    '''

    def get_contingencies(self):

        return sorted(self.contingencies.values(), key=(lambda r: r.label))

    def construct_data_records(self):

        out_str = StringIO()
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE, delimiter=' ')
        rows = [
            row
            for r in self.get_contingencies()
            for row in r.construct_record_rows()]
        writer.writerows(rows)
        return out_str.getvalue()

    def construct_end_record(self):

        out_str = StringIO()
        writer = csv.writer(out_str, lineterminator="\n", quoting=csv.QUOTE_NONE, delimiter=' ')
        rows = [['END']]
        writer.writerows(rows)
        return out_str.getvalue()        

    def write(self, file_name):
        '''write a CON file'''

        with open(file_name, 'w') as out_file:
            out_file.write(self.construct_data_records())
            out_file.write(self.construct_end_record())

    def read(self, file_name, target_contingency = None):
    
        with open(file_name, 'r') as in_file:
            lines = in_file.readlines()
        try:
            for l in lines:
                if l.find("'") > -1 or l.find('"') > -1:
                    print('no quotes allowed, line:')
                    print(l)
                    alert(
                        {'data_type': 'Con',
                         'error_message': 'no quotes allowed in CON file',
                         'diagnostics': l})
                    if raise_con_quote:
                        raise Exception('no quotes allowed in CON')
        except Exception as e:
            traceback.print_exc()
            raise e
        delimiter_str = " "
        #quote_str = "'"
        skip_initial_space = True
        rows = csv.reader(
            lines,
            delimiter=delimiter_str,
            #quotechar=quote_str,
            skipinitialspace=skip_initial_space,
            quoting=csv.QUOTE_NONE) # QUOTE_NONE
        rows = [[t.strip() for t in r] for r in rows]
        self.read_from_rows(rows, target_contingency)
        
    def row_is_file_end(self, row):

        is_file_end = False
        if len(row) == 0:
            is_file_end = True
        if row[0][:1] in {'','q','Q'}:
            is_file_end = True
        return is_file_end
    
    #def row_is_section_end(self, row):
    #
    #    is_section_end = False
    #    if row[0][:1] == '0':
    #        is_section_end = True
    #    return is_section_end

    def is_contingency_start(self, row):

        return (row[0].upper() == 'CONTINGENCY')

    def is_end(self, row):

        return (row[0].upper() == 'END')

    def is_branch_out_event(self, row):

        #return (
        #    row[0].upper() in {'DISCONNECT', 'OPEN', 'TRIP'} and
        #    row[1].upper() in {'BRANCH', 'LINE'})
        return (row[0] == 'OPEN' and row[1] == 'BRANCH')

    def is_three_winding(self, row):

        #print(row)
        if len(row) < 9:
            return False
        elif row[8].upper() == 'TO':
            return True
        else:
            return False

    def is_generator_out_event(self, row):

        #return(
        #    row[0].upper() == 'REMOVE' and
        #    row[1].upper() in {'UNIT', 'MACHINE'})

        if row[1] == 'MACHINE':
            row[1] = 'UNIT'

        return(row[0] == 'REMOVE' and row[1] == 'UNIT')
        
    #CHALLENGE2 - READ ONLY ONE CONTINGENCY
    def read_from_rows(self, rows, target_contingency = None):

        row_num = -1
        in_contingency = False
        num_records = 0
        keys_with_repeats = []
        while True:
            row_num += 1
            #if row_num >= len(rows): # in case the data provider failed to put an end file line
            #    return
            try:
                row = rows[row_num]
            except Exception as e:
                print('format error in CON file: missing file end row')
                traceback.print_exc()
                raise e
            if self.row_is_file_end(row):
                return
            #if self.row_is_section_end(row):
            #    break
            elif self.is_contingency_start(row):
                in_contingency = True
                contingency = Contingency()
                contingency.label = row[1]
            elif self.is_end(row):
                if in_contingency:
                    if  target_contingency == None or ( target_contingency != None and contingency.label == target_contingency):
                        self.contingencies[contingency.label] = contingency
                        num_records += 1
                        keys_with_repeats.append(contingency.label)
                    in_contingency = False
                else:
                    break
            elif self.is_branch_out_event(row):
                branch_out_event = BranchOutEvent()
                if self.is_three_winding(row):
                    branch_out_event.read_three_winding_from_row(row)
                else:
                    branch_out_event.read_from_row(row)
                contingency.branch_out_events.append(branch_out_event)
            elif self.is_generator_out_event(row):
                generator_out_event = GeneratorOutEvent()
                generator_out_event.read_from_row(row)
                contingency.generator_out_events.append(generator_out_event)
            else:
                try:
                    print('format error in CON file row:')
                    print(row)
                    raise Exception('format error in CON file')
                except Exception as e:
                    traceback.print_exc()
                    raise e
        if num_records > len(self.contingencies):
            repeated_keys = get_repeated_keys(keys_with_repeats)
            alert(
                {'data_type': 'Con',
                 'error_message': 'repeated key in CON file',
                 'diagnostics': {'records': num_records, 'distinct keys': len(self.contingencies), 'repeated keys': repeated_keys}})

def get_repeated_keys(keys):

    keys_with_repeats = sorted(keys)
    repeated_keys = []
    for i in range(len(keys_with_repeats) - 1):
        if keys_with_repeats[i] == keys_with_repeats[i + 1]:
            repeated_keys.append(keys_with_repeats[i])
    return sorted(list(set(repeated_keys)))

class CaseIdentification:

    def __init__(self):

        self.ic = 0
        self.sbase = 100.0
        self.rev = 33
        self.xfrrat = 0
        self.nxfrat = 1
        self.basfrq = 60.0
        self.record_2 = 'GRID OPTIMIZATION COMPETITION CHALLENGE 2'
        self.record_3 = 'INPUT DATA FILES ARE RAW JSON CON'
        #self.comment = ''

    def check(self):

        self.check_sbase_positive()

    def check_sbase_positive(self):

        if not (self.sbase > 0.0):
            alert(
                {'data_type':
                 'CaseIdentification',
                 'error_message':
                 'fails sbase positivitiy. please ensure that sbase > 0.0',
                 'diagnostics':
                 {'sbase': self.sbase}})

    def read_record_1_from_row(self, row):
        row = pad_row(row, 6)
        row[5] = extract_number(row[5])
        self.sbase = parse_token(row[1], float, default=None)
        if read_unused_fields:
            self.ic = parse_token(row[0], int, 0)
            self.rev = parse_token(row[2], int, 33)
            self.xfrrat = (1 if (parse_token(row[3], float, 0.0) > 0.0) else 0)
            self.nxfrat = (1 if (parse_token(row[4], float, 1.0) > 0.0) else 0)
            #self.xfrrat = parse_token(row[3], int, 0)
            #self.nxfrat = parse_token(row[4], int, 1)
            self.basfrq = parse_token(row[5], float, 60.0) # need to remove end of line comment
            #self.comment = row[6] if len(row) > 6 else ''

    def read_from_rows(self, rows):

        self.read_record_1_from_row(rows[0])
        #self.record_2 = '' # not preserving these at this point
        #self.record_3 = '' # do that later

class Bus:

    def __init__(self):

        self.i = None # no default allowed - we want this to throw an error
        self.name = 12*' '
        self.baskv = 0.0
        self.ide = 1
        self.area = 1
        self.zone = 1
        self.owner = 1
        self.vm = 1.0
        self.va = 0.0
        self.nvhi = 1.1
        self.nvlo = 0.9
        self.evhi = 1.1
        self.evlo = 0.9

    def scrub(self):

        self.name = scrub_unused_long_string(self.name)

    def check(self):

        self.check_ide_ne_4()
        self.check_i_pos()
        self.check_i_le_imax()
        self.check_area_pos()
        self.check_vm_pos()
        self.check_nvhi_pos()
        self.check_nvlo_pos()
        self.check_evhi_pos()
        self.check_evlo_pos()
        self.check_nvhi_nvlo_consistent()
        self.check_evhi_evlo_consistent()
        self.check_evhi_nvhi_consistent()
        self.check_nvlo_evlo_consistent()
        # check vm within bounds?
        # check area in areas?

    def clean_name(self):

        self.name = ''

    def check_ide_ne_4(self):

        if self.ide == 4:
            alert(
                {'data_type': 'Bus',
                 'error_message': 'fails ide != 4. Please ensure that the ide field of every bus is not equal to 4',
                 'diagnostics': {
                     'i': self.i,
                     'ide': self.ide}})

    def check_i_pos(self):

        if not (self.i > 0):
            alert(
                {'data_type': 'Bus',
                 'error_message': 'fails i positivity. Please ensure that the i field of every bus is a positive integer',
                 'diagnostics': {
                     'i': self.i}})

    def check_i_le_imax(self):

        imax = 999997 # from commercial power system software manual
        if self.i > imax:
            alert(
                {'data_type': 'Bus',
                 'error_message': 'fails i <= imax. Please ensure that the i field of every bus is <= %s' % imax,
                 'diagnostics': {
                     'i': self.i,
                     'imax': imax}})

    def check_area_pos(self):

        if not (self.area > 0):
            alert(
                {'data_type': 'Bus',
                 'error_message': 'fails area positivity. Please ensure that the area field of every bus is a positive integer',
                 'diagnostics': {
                     'i': self.i,
                     'area': self.area}})
    
    def check_vm_pos(self):

        if not (self.vm > 0.0):
            alert(
                {'data_type': 'Bus',
                 'error_message': 'fails vm positivity. Please ensure that the vm field of every bus is a positive real number',
                 'diagnostics': {
                     'i': self.i,
                     'vm': self.vm}})

    def check_nvhi_pos(self):

        if not (self.nvhi > 0.0):
            alert(
                {'data_type': 'Bus',
                 'error_message': 'fails nvhi positivity. Please ensure that the nvhi field of every bus is a positive real number',
                 'diagnostics': {
                     'i': self.i,
                     'nvhi': self.nvhi}})

    def check_nvlo_pos(self):

        if not (self.nvlo > 0.0):
            alert(
                {'data_type': 'Bus',
                 'error_message': 'fails nvlo positivity. Please ensure that the nvlo field of every bus is a positive real number',
                 'diagnostics': {
                     'i': self.i,
                     'nvlo': self.nvlo}})

    def check_evhi_pos(self):

        if not (self.evhi > 0.0):
            alert(
                {'data_type': 'Bus',
                 'error_message': 'fails evhi positivity. Please ensure that the evhi field of every bus is a positive real number',
                 'diagnostics': {
                     'i': self.i,
                     'evhi': self.evhi}})

    def check_evlo_pos(self):

        if not (self.evlo > 0.0):
            alert(
                {'data_type': 'Bus',
                 'error_message': 'fails evlo positivity. Please ensure that the evlo field of every bus is a positive real number',
                 'diagnostics': {
                     'i': self.i,
                     'evlo': self.evlo}})

    def check_nvhi_nvlo_consistent(self):

        if self.nvhi - self.nvlo < 0.0:
            alert(
                {'data_type': 'Bus',
                 'error_message': 'fails nvhi-nvlo consistency. Please ensure that the nvhi and nvlo fields of every bus satisfy: nvhi - nvlo >= 0.0',
                 'diagnostics': {
                     'i': self.i,
                     'nvhi - nvlo': (self.nvhi - self.nvlo),
                     'nvhi': self.nvhi,
                     'nvlo': self.nvlo}})

    def check_evhi_evlo_consistent(self):

        if self.evhi - self.evlo < 0.0:
            alert(
                {'data_type': 'Bus',
                 'error_message': 'fails evhi-evlo consistency. Please ensure that the evhi and evlo fields of every bus satisfy: evhi - evlo >= 0.0',
                 'diagnostics': {
                     'i': self.i,
                     'evhi - evlo': (self.evhi - self.evlo),
                     'evhi': self.evhi,
                     'evlo': self.evlo}})

    def check_evhi_nvhi_consistent(self):

        if self.evhi - self.nvhi < 0.0:
            alert(
                {'data_type': 'Bus',
                 'error_message': 'fails evhi-nvhi consistency. Please ensure that the evhi and nvhi fields of every bus satisfy: evhi - nvhi >= 0.0',
                 'diagnostics': {
                     'i': self.i,
                     'evhi - nvhi': (self.evhi - self.nvhi),
                     'evhi': self.evhi,
                     'nvhi': self.nvhi}})

    def check_nvlo_evlo_consistent(self):

        if self.nvlo - self.evlo < 0.0:
            alert(
                {'data_type': 'Bus',
                 'error_message': 'fails nvlo-evlo consistency. Please ensure that the nvlo and evlo fields of every bus satisfy: nvlo - evlo >= 0.0',
                 'diagnostics': {
                     'i': self.i,
                     'nvlo - evlo': (self.nvlo - self.evlo),
                     'nvlo': self.nvlo,
                     'evlo': self.evlo}})

    def read_from_row(self, row):

        row = pad_row(row, 13)
        self.i = parse_token(row[0], int, default=None)
        self.area = parse_token(row[4], int, default=None)
        self.vm = parse_token(row[7], float, default=None)
        self.va = parse_token(row[8], float, default=None)
        self.nvhi = parse_token(row[9], float, default=None)
        self.nvlo = parse_token(row[10], float, default=None)
        self.evhi = parse_token(row[11], float, default=None)
        self.evlo = parse_token(row[12], float, default=None)
        if read_unused_fields:
            self.name = parse_token(row[1], str, 12*' ')
            self.baskv = parse_token(row[2], float, 0.0)
            self.ide = parse_token(row[3], int, 1)
            self.zone = parse_token(row[5], int, 1)
            self.owner = parse_token(row[6], int, 1)
    
class Load:

    def __init__(self):

        self.i = None # no default allowed - should be an error
        self.id = '1'
        self.status = 1
        self.area = 1 # default is area of bus self.i, but this is not available yet
        self.zone = 1
        self.pl = 0.0
        self.ql = 0.0
        self.ip = 0.0
        self.iq = 0.0
        self.yp = 0.0
        self.yq = 0.0
        self.owner = 1
        self.scale = 1
        self.intrpt = 0

    def check(self):

        self.check_id_len_1_or_2()
        # need to check i in buses
        self.check_pl_nonnegative(scrub_mode=False)
        self.check_pl_ql_at_least_one_nonzero()

    def scrub(self):

        self.check_pl_nonnegative(scrub_mode=True)

    def check_pl_nonnegative(self, scrub_mode=False):

        if self.pl < 0.0:
            alert(
                {'data_type': 'Load',
                 'error_message': ('fails PL >= 0.0.' + (' Setting PL = 0.0.' if scrub_mode else '')),
                 'diagnostics': {
                     'i': self.i,
                     'id': self.id,
                     'pl': self.pl}})
            if scrub_mode:
                self.pl = 0.0

    def check_pl_ql_at_least_one_nonzero(self):

        if remove_loads_with_pq_eq_0:
            if (self.pl == 0.0) and (self.ql == 0.0):
                alert(
                    {'data_type':
                         'Raw',
                     'error_message':
                         'found a load with pl == 0.0 and ql == 0.0, will be removed when scrubber is applied',
                     'diagnostics':
                         {'i': self.i, 'id': self.id, 'pl': self.pl, 'ql': self.ql}})

    def check_id_len_1_or_2(self):

        if not(len(self.id) in [1, 2]):
            alert(
                {'data_type': 'Load',
                 'error_message': 'fails id string len 1 or 2. Please ensure that the id field of every load is a 1- or 2-character string with no blank characters',
                 'diagnostics': {
                     'i': self.i,
                     'id': self.id}})

    #def clean_id(self):
        '''remove spaces and non-allowed characters
        hope that this does not introduce duplication'''

    #    self.id = clean_short_str(self.id)

    def read_from_row(self, row):

        row = pad_row(row, 14)
        self.i = parse_token(row[0], int, default=None)
        self.id = parse_token(row[1], str, default=None).strip()
        self.status = parse_token(row[2], int, default=None)
        self.pl = parse_token(row[5], float, default=None)
        self.ql = parse_token(row[6], float, default=None)
        if read_unused_fields:
            self.area = parse_token(row[3], int, 1)
            self.zone = parse_token(row[4], int, 1)
            self.ip = parse_token(row[7], float, 0.0)
            self.iq = parse_token(row[8], float, 0.0)
            self.yp = parse_token(row[9], float, 0.0)
            self.yq = parse_token(row[10], float, 0.0)
            self.owner = parse_token(row[11], int, 1)
            self.scale = parse_token(row[12], int, 1)
            self.intrpt = parse_token(row[13], int, 0)

class FixedShunt:

    def __init__(self):

        self.i = None # no default allowed
        self.id = '1'
        self.status = 1
        self.gl = 0.0
        self.bl = 0.0

    def check(self):

        self.check_id_len_1_or_2()
        # need to check i in buses

    def scrub(self):

        pass

    def check_id_len_1_or_2(self):

        if not(len(self.id) in [1, 2]):
            alert(
                {'data_type': 'FixedShunt',
                 'error_message': 'fails id string len 1 or 2. Please ensure that the id field of every fixed shunt is a 1- or 2-character string with no blank characters',
                 'diagnostics': {
                     'i': self.i,
                     'id': self.id}})

    def read_from_row(self, row):

        row = pad_row(row, 5)
        self.i = parse_token(row[0], int, default=None)
        self.id = parse_token(row[1], str, default=None).strip()
        self.status = parse_token(row[2], int, default=None)
        self.gl = parse_token(row[3], float, default=None)
        self.bl = parse_token(row[4], float, default=None)
        if read_unused_fields:
            pass

class Generator:

    def __init__(self):
        self.i = None # no default allowed
        self.id = '1'
        self.pg = 0.0
        self.qg = 0.0
        self.qt = 9999.0
        self.qb = -9999.0
        self.vs = 1.0
        self.ireg = 0
        self.mbase = 100.0 # need to take default value for this from larger Raw class
        self.zr = 0.0
        self.zx = 1.0
        self.rt = 0.0
        self.xt = 0.0
        self.gtap = 1.0
        self.stat = 1
        self.rmpct = 100.0
        self.pt = 9999.0
        self.pb = -9999.0
        self.o1 = 1
        self.f1 = 1.0
        self.o2 = 0
        self.f2 = 1.0
        self.o3 = 0
        self.f3 = 1.0
        self.o4 = 0
        self.f4 = 1.0
        self.wmod = 0
        self.wpf = 1.0
        self.fuel = "None"

    def check(self):

        check_two_char_id_str(self.id)
        self.check_id_len_1_or_2()
        self.check_pg_nonnegative(scrub_mode=False)
        if do_check_pb_nonnegative:
            self.check_pb_nonnegative()
        self.check_qt_qb_consistent()
        self.check_pt_pb_consistent()
        self.check_pg_stat_consistent()
        self.check_qg_stat_consistent()
        # check pg, qg within bounds? - not for C2
        # need to check i in buses

    def scrub(self):

        self.scrub_pg_stat_consistent()
        self.scrub_qg_stat_consistent()
        self.check_pg_nonnegative(scrub_mode=True)

    def add_emergency_capacity(self):
        '''add emergency capacity
        for study 1
        increase pmax by a given factor = EMERGENCY_CAPACITY_FACTOR'''

        self.pt += EMERGENCY_CAPACITY_FACTOR * abs(self.pt)

    def check_pg_nonnegative(self, scrub_mode=False):

        if self.pg < 0.0:
            alert(
                {'data_type': 'Generator',
                 'error_message': ('fails PG >= 0.0.' + (' Setting PG = 0.0.' if scrub_mode else '')),
                 'diagnostics': {
                     'i': self.i,
                     'id': self.id,
                     'pg': self.pg}})
            if scrub_mode:
                self.pg = 0.0

    def check_id_len_1_or_2(self):

        if not(len(self.id) in [1, 2]):
            alert(
                {'data_type': 'Generator',
                 'error_message': 'fails id string len 1 or 2. Please ensure that the id field of every generator is a 1- or 2-character string with no blank characters',
                 'diagnostics': {
                     'i': self.i,
                     'id': self.id}})

    def check_pg_stat_consistent(self):
        
        if abs(self.pg) > 0.0 and self.stat == 0:
            alert(
                {'data_type': 'Generator',
                 'error_message': 'fails pg-stat consistency. Please ensure that pg = 0.0 if stat = 0',
                 'diagnostics': {
                     'i': self.i,
                     'id': self.id,
                     'pg': self.pg,
                     'stat': self.stat}})

    def check_qg_stat_consistent(self):
        
        if abs(self.qg) > 0.0 and self.stat == 0:
            alert(
                {'data_type': 'Generator',
                 'error_message': 'fails qg-stat consistency. Please ensure that qg = 0.0 if stat = 0',
                 'diagnostics': {
                     'i': self.i,
                     'id': self.id,
                     'qg': self.qg,
                     'stat': self.stat}})

    def check_qt_qb_consistent(self):
        
        if self.qt - self.qb < 0.0:
            alert(
                {'data_type': 'Generator',
                 'error_message': 'fails qt-qb consistency. Please ensure that the qt and qb fields of every generator satisfy: qt - qb >= 0.0',
                 'diagnostics': {
                     'i': self.i,
                     'id': self.id,
                     'qt - qb': (self.qt - self.qb),
                     'qt': self.qt,
                     'qb': self.qb}})

    def check_pt_pb_consistent(self):
        
        if self.pt - self.pb < 0.0:
            alert(
                {'data_type': 'Generator',
                 'error_message': 'fails pt-pb consistency. Please ensure that the pt and pb fields of every generator satisfy: pt - pb >= 0.0',
                 'diagnostics': {
                     'i': self.i,
                     'id': self.id,
                     'pt - pb': (self.pt - self.pb),
                     'pt': self.pt,
                     'pb': self.pb}})


    def scrub_pg_stat_consistent(self):
        
        if pg_qg_stat_mode == 1:
            message = 'Setting pg to 0.0.'
        elif pg_qg_stat_mode == 2:
            message = 'Setting stat to 1.'
        else:
            message = 'Skipping scrub.'
        if abs(self.pg) > 0.0 and self.stat == 0 and pg_qg_stat_mode > 0:
            alert(
                {'data_type': 'Generator',
                 'error_message': 'fails pg-stat consistency. {}'.format(message),
                 'diagnostics': {
                     'i': self.i,
                     'id': self.id,
                     'pg': self.pg,
                     'stat': self.stat}})
            if pg_qg_stat_mode == 1:
                self.pg = 0.0
            if pg_qg_stat_mode == 2:
                self.stat = 1

    def scrub_qg_stat_consistent(self):
        
        if pg_qg_stat_mode == 1:
            message = 'Setting qg to 0.0.'
        elif pg_qg_stat_mode == 2:
            message = 'Setting stat to 1.'
        else:
            message = 'Skipping scrub.'
        if abs(self.qg) > 0.0 and self.stat == 0 and pg_qg_stat_mode > 0:
            alert(
                {'data_type': 'Generator',
                 'error_message': 'fails qg-stat consistency. {}'.format(message),
                 'diagnostics': {
                     'i': self.i,
                     'id': self.id,
                     'qg': self.qg,
                     'stat': self.stat}})
            if pg_qg_stat_mode == 1:
                self.qg = 0.0
            if pg_qg_stat_mode == 2:
                self.stat = 1

    def check_pb_nonnegative(self):

        if self.pb < 0.0:
            alert(
                {'data_type': 'Generator',
                 'error_message': 'fails pb nonnegativity. Please ensure that the pb fields of every generator satisfies: pb >= 0.0',
                 'diagnostics': {
                     'i': self.i,
                     'id': self.id,
                     'pb': self.pb}})

    def read_from_row(self, row):

        row = pad_row(row, 28)
        self.i = parse_token(row[0], int, default=None)
        self.id = parse_token(row[1], str, default=None).strip()
        self.pg = parse_token(row[2], float, default=None)
        self.qg = parse_token(row[3], float, default=None)
        self.qt = parse_token(row[4], float, default=None)
        self.qb = parse_token(row[5], float, default=None)
        self.stat = parse_token(row[14], int, default=None)
        self.pt = parse_token(row[16], float, default=None)
        self.pb = parse_token(row[17], float, default=None)
        if read_unused_fields:
            self.vs = parse_token(row[6], float, 1.0)
            self.ireg = parse_token(row[7], int, 0)
            self.mbase = parse_token(row[8], float, 100.0)
            self.zr = parse_token(row[9], float, 0.0)
            self.zx = parse_token(row[10], float, 1.0)
            self.rt = parse_token(row[11], float, 0.0)
            self.xt = parse_token(row[12], float, 0.0)
            self.gtap = parse_token(row[13], float, 1.0)
            self.rmpct = parse_token(row[15], float, 100.0)
            self.o1 = parse_token(row[18] if 18 < len(row) else None,int, 1)
            self.f1 = parse_token(row[19] if 19 < len(row) else None, float, 1.0)
            self.o2 = parse_token(row[20] if 20 < len(row) else None, int, 0)
            self.f2 = parse_token(row[21] if 21 < len(row) else None, float, 1.0)
            self.o3 = parse_token(row[22] if 22 < len(row) else None, int, 0)
            self.f3 = parse_token(row[23] if 23 < len(row) else None, float, 1.0)
            self.o4 = parse_token(row[24] if 24 < len(row) else None, int, 0)
            self.f4 = parse_token(row[25] if 25 < len(row) else None, float, 1.0)
            self.wmod = parse_token(row[26] if 26 < len(row) else None, int, 0)
            self.wpf = parse_token(row[27] if 27 < len(row) else None, float, 1.0)

class NontransformerBranch:

    def __init__(self):

        self.i = None # no default
        self.j = None # no default
        self.ckt = '1'
        self.r = None # no default
        self.x = None # no default
        self.b = 0.0
        self.ratea = 0.0
        self.rateb = 0.0
        self.ratec = 0.0
        self.gi = 0.0
        self.bi = 0.0
        self.gj = 0.0
        self.bj = 0.0
        self.st = 1
        self.met = 1
        self.len = 0.0
        self.o1 = 1
        self.f1 = 1.0
        self.o2 = 0
        self.f2 = 1.0
        self.o3 = 0
        self.f3 = 1.0
        self.o4 = 0
        self.f4 = 1.0

    def scrub(self):

        if self.ratea <= 0.0:
            '''
            alert(
                {'data_type': 'NontransformerBranch',
                 'error_message': 'adjusting ratea to %f' % default_branch_limit,
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'ckt': self.ckt,
                     'ratea': self.ratea}})
            ''' 
            self.ratea = default_branch_limit
        if self.ratec < self.ratea:
            ''' 
            alert(
                {'data_type': 'NontransformerBranch',
                 'error_message': 'adjusting ratec to ratea',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'ckt': self.ckt,
                     'ratea': self.ratea,
                     'ratec': self.ratec}})
            '''
            self.ratec = self.ratea

    def check(self):

        check_two_char_id_str(self.ckt)
        self.check_ckt_len_1_or_2()
        self.check_r_x_nonzero()
        if do_check_rate_pos:
            self.check_ratea_pos()
            self.check_ratec_pos()
        self.check_ratec_ratea_consistent()
        if do_check_line_i_lt_j:
            self.check_i_lt_j()
        self.check_i_ne_j()
        # need to check i, j in buses

    def check_i_lt_j(self):

        if not (self.i < self.j):
            alert(
                {'data_type': 'NontransformerBranch',
                 'error message': 'fails i < j. Please ensure every nontransformer branch has i < j',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'ckt': self.ckt}})

    def check_i_ne_j(self):

        if self.i == self.j:
            alert(
                {'data_type': 'NontransformerBranch',
                 'error message': 'fails i != j. Please ensure every nontransformer branch has i != j',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'ckt': self.ckt}})
        
    def check_ckt_len_1_or_2(self):

        if not(len(self.ckt) in [1, 2]):
            alert(
                {'data_type': 'NontransformerBranch',
                 'error_message': 'fails ckt string len 1 or 2. Please ensure that the id field of every nontransformer branch is a 1- or 2-character string with no blank characters',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'ckt': self.ckt}})

    def check_r_x_nonzero(self):
        
        if (self.r == 0.0 and self.x == 0.0):
            alert(
                {'data_type': 'NontransformerBranch',
                 'error_message': 'fails r-x nonzero. Please ensure that at least one of the r and x fields of every nontransformer branch is nonzero. The competition formulation uses z = r + j*x, y = 1/z, g = Re(y), b = Im(y). This computation fails if r == 0.0 and x == 0.0.',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'ckt': self.ckt,
                     'r:': self.r,
                     'x:': self.x}})

    def check_ratea_pos(self):
        
        if not (self.ratea > 0.0):
            alert(
                {'data_type': 'NontransformerBranch',
                 'error_message': 'fails ratea positivity. Please ensure that the ratea field of every nontransformer branch is a positive real number.',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'ckt': self.ckt,
                     'ratea': self.ratea}})

    def check_ratec_pos(self):
        
        if not (self.ratec > 0.0):
            alert(
                {'data_type': 'NontransformerBranch',
                 'error_message': 'fails ratec positivity. Please ensure that the ratec field of every nontransformer branch is a positive real number.',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'ckt': self.ckt,
                     'ratec': self.ratec}})

    def check_ratec_ratea_consistent(self):

        global ratec_ratea_2
        if ratec_ratea_2 == True:
            return
        ratec_ratea_2 = True
        
        if self.ratec - self.ratea < 0.0:
            alert(
                {'data_type': 'NontransformerBranch',
                 'error_message': 'fails ratec-ratea consistency. Please ensure that the ratec and ratea fields of every nontransformer branch satisfy ratec - ratea >= 0.0.',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'ckt': self.ckt,
                     'ratec - ratea': self.ratec - self.ratea,
                     'ratec': self.ratec,
                     'ratea': self.ratea}})

    def read_from_row(self, row):

        row = pad_row(row, 24)
        self.i = parse_token(row[0], int, default=None)
        self.j = parse_token(row[1], int, default=None)
        self.ckt = parse_token(row[2], str, default=None).strip()
        self.r = parse_token(row[3], float, default=None)
        self.x = parse_token(row[4], float, default=None)
        self.b = parse_token(row[5], float, default=None)
        self.ratea = parse_token(row[6], float, default=None)
        self.ratec = parse_token(row[8], float, default=None)
        self.st = parse_token(row[13], int, default=None)
        if read_unused_fields:
            self.rateb = parse_token(row[7], float, 0.0)
            self.gi = parse_token(row[9], float, 0.0)
            self.bi = parse_token(row[10], float, 0.0)
            self.gj = parse_token(row[11], float, 0.0)
            self.bj = parse_token(row[12], float, 0.0)
            self.met = parse_token(row[14], int, 1)
            self.len = parse_token(row[15], float, 0.0)
            self.o1 = parse_token(row[16] if 16 < len(row) else None, int, 1)
            self.f1 = parse_token(row[17] if 17 < len(row) else None, float, 1.0)
            self.o2 = parse_token(row[18] if 18 < len(row) else None,int, 0)
            self.f2 = parse_token(row[19] if 19 < len(row) else None, float, 1.0)
            self.o3 = parse_token(row[20] if 20 < len(row) else None, int, 0)
            self.f3 = parse_token(row[21] if 21 < len(row) else None, float, 1.0)
            self.o4 = parse_token(row[22] if 22 < len(row) else None, int, 0)
            self.f4 = parse_token(row[23] if 23 < len(row) else None, float, 1.0)

class Transformer:

    def __init__(self):

        self.i = None # no default
        self.j = None # no default
        self.k = 0
        self.ckt = '1'
        self.cw = 1
        self.cz = 1
        self.cm = 1
        self.mag1 = 0.0
        self.mag2 = 0.0
        self.nmetr = 2
        self.name = 12*' '
        self.stat = 1
        self.o1 = 1
        self.f1 = 1.0
        self.o2 = 0
        self.f2 = 1.0
        self.o3 = 0
        self.f3 = 1.0
        self.o4 = 0
        self.f4 = 1.0
        self.vecgrp = 12*' '
        self.r12 = 0.0
        self.x12 = None # no default allowed
        self.sbase12 = 100.0
        self.windv1 = 1.0
        self.nomv1 = 0.0
        self.ang1 = 0.0
        self.rata1 = 0.0
        self.ratb1 = 0.0
        self.ratc1 = 0.0
        self.cod1 = 0
        self.cont1 = 0
        self.rma1 = 1.1
        self.rmi1 = 0.9
        self.vma1 = 1.1
        self.vmi1 = 0.9
        self.ntp1 = 33
        self.tab1 = 0
        self.cr1 = 0.0
        self.cx1 = 0.0
        self.cnxa1 = 0.0
        self.windv2 = 1.0
        self.nomv2 = 0.0

    def scrub(self):

        if self.rata1 <= 0.0:
            '''
            alert(
                {'data_type': 'Transformer',
                 'error_message': 'adjusting rata1 to 1.0',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'k': self.k,
                     'ckt': self.ckt,
                     'rata1': self.rata1}})
            '''
            self.rata1 = 1.0
        if self.ratc1 < self.rata1:
            '''
            alert(
                {'data_type': 'Transformer',
                 'error_message': 'adjusting ratc1 to rata1',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'k': self.k,
                     'ckt': self.ckt,
                     'rata1': self.rata1,
                     'ratc1': self.ratc1}})
            ''' 
            self.ratc1 = self.rata1
        self.check_tau_theta_init_feas(scrub_mode=True)

    def check(self):

        check_two_char_id_str(self.ckt)
        self.check_ckt_len_1_or_2()
        self.check_cod1_013() # COD1 can be any integer value now - nope, not anymore
        self.check_ntp1_odd_ge_1()
        self.check_r12_x12_nonzero()
        if do_check_rate_pos:
            self.check_rata1_pos()
            self.check_ratc1_pos()
        self.check_ratc1_rata1_consistent()
        self.check_windv1_pos()
        self.check_windv2_pos()
        self.check_windv2_eq_1()
        self.check_k_0()
        if do_check_xfmr_i_lt_j:
            self.check_i_lt_j()
        self.check_i_ne_j()
        # need to check i, j in buses
        self.check_tau_theta_init_feas(scrub_mode=False)

    def check_cod1_013(self):

        if not (self.cod1 in [-3, -1, 0, 1, 3]):
            alert(
                {'data_type': 'Transformer',
                 'error message': 'COD1 not in [-3, -1, 0, 1, 3].',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'k': self.k,
                     'ckt': self.ckt,
                     'cod1': self.cod1}})

    def check_ntp1_odd_ge_1(self):

        if not ((self.ntp1 >= 1) and (self.ntp1 % 2 == 1)):
            alert(
                {'data_type': 'Transformer',
                 'error message': 'NTP1 not in [1, 3, 5, ...].',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'k': self.k,
                     'ckt': self.ckt,
                     'ntp1': self.ntp1}})

    def check_tau_theta_init_feas(self, scrub_mode=False):

        x = compute_xfmr_position(self)
        position = x[0]
        oper_val = x[1]
        oper_val_resulting = x[2]
        resid = x[3]
        mid_val = x[4]
        step_size = x[5]
        max_position = x[6]
        if debug_check_tau_theta_init_feas:
            alert(
                {'data_type': 'Transformer',
                 'error message': 'printing tau/theta init info for debugging.',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'k': self.k,
                     'ckt': self.ckt,
                     'cod1': self.cod1,
                     'stat': self.stat,
                     'ntp1': self.ntp1,
                     'rma1': self.rma1,
                     'rmi1': self.rmi1,
                     'max_position': max_position,
                     'position': position,
                     'oper_val': oper_val,
                     'oper_val_resulting': oper_val_resulting,
                     'resid': resid,
                     'mid_val': mid_val,
                     'step_size': step_size}})
        if scrub_mode:
            if do_fix_xfmr_tau_theta_init:
                #print('scrubbing xfmr tau/theta init value')
                if self.cod1 in [-1,1]:
                    self.windv1 = oper_val_resulting
                    self.windv2 = 1.0
                elif self.cod1 in [-3,3]:
                    self.ang1 = oper_val_resulting
        elif abs(resid) > xfmr_tau_theta_init_tol * abs(oper_val):
            alert(
                {'data_type': 'Transformer',
                 'error message': 'tau/theta init is infeasible.',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'k': self.k,
                     'ckt': self.ckt,
                     'cod1': self.cod1,
                     'stat': self.stat,
                     'ntp1': self.ntp1,
                     'rma1': self.rma1,
                     'rmi1': self.rmi1,
                     'max_position': max_position,
                     'position': position,
                     'oper_val': oper_val,
                     'oper_val_resulting': oper_val_resulting,
                     'resid': resid,
                     'mid_val': mid_val,
                     'step_size': step_size}})
        
    def check_k_0(self):

        if not (self.k == 0):
            alert(
                {'data_type': 'Transformer',
                 'error message': 'fails k = 0. Please ensure every transformer has k = 0. Note, a three-winding transformer can be modeled by a configuration of two-winding transformers.',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'k': self.k,
                     'ckt': self.ckt}})
        
    def check_i_lt_j(self):

        if not (self.i < self.j):
            alert(
                {'data_type': 'Transformer',
                 'error message': 'fails i < j. Please ensure every transformer has i < j',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'ckt': self.ckt}})

    def check_i_ne_j(self):

        if self.i == self.j:
            alert(
                {'data_type': 'Transformer',
                 'error message': 'fails i != j. Please ensure every transformer has i != j',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'ckt': self.ckt}})

    def check_ckt_len_1_or_2(self):

        if not(len(self.ckt) in [1, 2]):
            alert(
                {'data_type': 'Transformer',
                 'error_message': 'fails ckt string len 1 or 2. Please ensure that the ckt field of every transformer is a 1- or 2-character string with no blank characters',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'k': self.k,
                     'ckt': self.ckt}})

    def check_r12_x12_nonzero(self):
        
        if (self.r12 == 0.0 and self.x12 == 0.0):
            alert(
                {'data_type': 'Transformer',
                 'error_message': 'fails r12-x12 nonzero. Please ensure that at least one of the r12 and x12 fields of every transformer is nonzero. The competition formulation uses z = r12 + j*x12, y = 1/z, g = Re(y), b = Im(y). This computation fails if r12 == 0.0 and x12 == 0.0.',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'k': self.k,
                     'ckt': self.ckt,
                     'r12:': self.r12,
                     'x12:': self.x12}})

    def check_rata1_pos(self):
        
        if not (self.rata1 > 0.0):
            alert(
                {'data_type': 'Transformer',
                 'error_message': 'fails rata1 positivity. Please ensure that the rata1 field of every transformer is a positive real number.',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'k': self.k,
                     'ckt': self.ckt,
                     'rata1': self.rata1}})

    def check_ratc1_pos(self):
        
        if not (self.ratc1 > 0.0):
            alert(
                {'data_type': 'Transformer',
                 'error_message': 'fails ratc1 positivity. Please ensure that the ratc1 field of every transformer is a positive real number.',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'k': self.k,
                     'ckt': self.ckt,
                     'ratc1': self.ratc1}})

    def check_ratc1_rata1_consistent(self):

        global ratc1_rata1
        if ratc1_rata1 == True:
            return

        ratc1_rata1 = True
        
        if self.ratc1 - self.rata1 < 0.0:
            alert(
                {'data_type': 'Transformer',
                 'error_message': 'fails ratc1-rata1 consistency. Please ensure that the ratc1 and rata1 fields of every transformer satisfy ratc1 - rata1 >= 0.0.',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'k': self.k,
                     'ckt': self.ckt,
                     'ratc1 - rata1': self.ratc1 - self.rata1,
                     'ratc1': self.ratc1,
                     'rata1': self.rata1}})

    def check_windv1_pos(self):
        
        if not (self.windv1 > 0.0):
            alert(
                {'data_type': 'Transformer',
                 'error_message': 'fails windv1 positivity. Please ensure that the windv1 field of every transformer is a positive real number.',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'k': self.k,
                     'ckt': self.ckt,
                     'windv1': self.windv1}})

    def check_windv2_pos(self):
        
        if not (self.windv2 > 0.0):
            alert(
                {'data_type': 'Transformer',
                 'error_message': 'fails windv2 positivity. Please ensure that the windv2 field of every transformer is a positive real number.',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'k': self.k,
                     'ckt': self.ckt,
                     'windv2': self.windv2}})

    def check_windv2_eq_1(self):
        
        if not(self.windv2 == 1.0):
            alert(
                {'data_type': 'Transformer',
                 'error_message': 'fails windv2 exactly equal to 1.0. Please ensure that the windv2 field of every transformer is equal to 1.0. Transformers not satisfying this property can be converted. This ensures that the formulation used by the Grid Optimization Competition is consistent with the model described in PSSE proprietary documentation',
                 'diagnostics': {
                     'i': self.i,
                     'j': self.j,
                     'k': self.k,
                     'ckt': self.ckt,
                     'windv2': self.windv2}})

    @property
    def num_windings(self):

        num_windings = 0
        if self.k is None:
            num_windings = 0
        elif self.k == 0:
            num_windings = 2
        else:
            num_windings = 3
        return num_windings
    
    def get_num_rows_from_row(self, row):

        num_rows = 0
        k = parse_token(row[2], int, 0)
        if k == 0:
            num_rows = 4
        else:
            num_rows = 5
        return num_rows

    def read_from_rows(self, rows):

        full_rows = self.pad_rows(rows)
        row = self.flatten_rows(full_rows)
        try:
            self.read_from_row(row)
        except Exception as e:
            print("row:")
            print(row)
            raise e
        
    def pad_rows(self, rows):

        return rows
        '''
        rows_new = rows
        if len(rows_new) == 4:
            rows_new.append([])
        rows_len = [len(r) for r in rows_new]
        rows_len_new = [21, 11, 17, 17, 17]
        rows_len_increase = [rows_len_new[i] - rows_len[i] for i in range(5)]
        # check no negatives in increase
        rows_new = [rows_new[i] + rows_len_increase[i]*[''] for i in range(5)]
        return rows_new
        '''

    def flatten_rows(self, rows):

        row = [t for r in rows for t in r]
        return row
    
    def read_from_row(self, row):

        # general (3- or 2-winding, 5- or 4-row)
        '''
        self.i = parse_token(row[0], int, '')
        self.j = parse_token(row[1], int, '')
        self.k = parse_token(row[2], int, 0)
        self.ckt = parse_token(row[3], str, '1')
        self.cw = parse_token(row[4], int, 1)
        self.cz = parse_token(row[5], int, 1)
        self.cm = parse_token(row[6], int, 1)
        self.mag1 = parse_token(row[7], float, 0.0)
        self.mag2 = parse_token(row[8], float, 0.0)
        self.nmetr = parse_token(row[9], int, 2)
        self.name = parse_token(row[10], str, 12*' ')
        self.stat = parse_token(row[11], int, 1)
        self.o1 = parse_token(row[12], int, 0)
        self.f1 = parse_token(row[13], float, 1.0)
        self.o2 = parse_token(row[14], int, 0)
        self.f2 = parse_token(row[15], float, 1.0)
        self.o3 = parse_token(row[16], int, 0)
        self.f3 = parse_token(row[17], float, 1.0)
        self.o4 = parse_token(row[18], int, 0)
        self.f4 = parse_token(row[19], float, 1.0)
        self.vecgrp = parse_token(row[20], str, 12*' ')
        self.r12 = parse_token(row[21], float, 0.0)
        self.x12 = parse_token(row[22], float, 0.0)
        self.sbase12 = parse_token(row[23], float, 0.0)
        self.r23 = parse_token(row[24], float, 0.0)
        self.x23 = parse_token(row[25], float, 0.0)
        self.sbase23 = parse_token(row[26], float, 0.0)
        self.r31 = parse_token(row[27], float, 0.0)
        self.x31 = parse_token(row[28], float, 0.0)
        self.sbase31 = parse_token(row[29], float, 0.0)
        self.vmstar = parse_token(row[30], float, 1.0)
        self.anstar = parse_token(row[31], float, 0.0)
        self.windv1 = parse_token(row[32], float, 1.0)
        self.nomv1 = parse_token(row[33], float, 0.0)
        self.ang1 = parse_token(row[34], float, 0.0)
        self.rata1 = parse_token(row[35], float, 0.0)
        self.ratb1 = parse_token(row[36], float, 0.0)
        self.ratc1 = parse_token(row[37], float, 0.0)
        self.cod1 = parse_token(row[38], int, 0)
        self.cont1 = parse_token(row[39], int, 0)
        self.rma1 = parse_token(row[40], float, 1.1)
        self.rmi1 = parse_token(row[41], float, 0.9)
        self.vma1 = parse_token(row[42], float, 1.1)
        self.vmi1 = parse_token(row[43], float, 0.9)
        self.ntp1 = parse_token(row[44], int, 33)
        self.tab1 = parse_token(row[45], int, 0)
        self.cr1 = parse_token(row[46], float, 0.0)
        self.cx1 = parse_token(row[47], float, 0.0)
        self.cnxa1 = parse_token(row[48], float, 0.0)
        self.windv2 = parse_token(row[49], float, 1.0)
        self.nomv2 = parse_token(row[50], float, 0.0)
        self.ang2 = parse_token(row[51], float, 0.0)
        self.rata2 = parse_token(row[52], float, 0.0)
        self.ratb2 = parse_token(row[53], float, 0.0)
        self.ratc2 = parse_token(row[54], float, 0.0)
        self.cod2 = parse_token(row[55], int, 0)
        self.cont2 = parse_token(row[56], int, 0)
        self.rma2 = parse_token(row[57], float, 1.1)
        self.rmi2 = parse_token(row[58], float, 0.9)
        self.vma2 = parse_token(row[59], float, 1.1)
        self.vmi2 = parse_token(row[60], float, 0.9)
        self.ntp2 = parse_token(row[61], int, 33)
        self.tab2 = parse_token(row[62], int, 0)
        self.cr2 = parse_token(row[63], float, 0.0)
        self.cx2 = parse_token(row[64], float, 0.0)
        self.cnxa2 = parse_token(row[65], float, 0.0)
        self.windv3 = parse_token(row[66], float, 1.0)
        self.nomv3 = parse_token(row[67], float, 0.0)
        self.ang3 = parse_token(row[68], float, 0.0)
        self.rata3 = parse_token(row[69], float, 0.0)
        self.ratb3 = parse_token(row[70], float, 0.0)
        self.ratc3 = parse_token(row[71], float, 0.0)
        self.cod3 = parse_token(row[72], int, 0)
        self.cont3 = parse_token(row[73], int, 0)
        self.rma3 = parse_token(row[74], float, 1.1)
        self.rmi3 = parse_token(row[75], float, 0.9)
        self.vma3 = parse_token(row[76], float, 1.1)
        self.vmi3 = parse_token(row[77], float, 0.9)
        self.ntp3 = parse_token(row[78], int, 33)
        self.tab3 = parse_token(row[79], int, 0)
        self.cr3 = parse_token(row[80], float, 0.0)
        self.cx3 = parse_token(row[81], float, 0.0)
        self.cnxa3 = parse_token(row[82], float, 0.0)
        '''
        
        # check no 3-winding
        self.i = parse_token(row[0], int, default=None)
        self.j = parse_token(row[1], int, default=None)
        k = parse_token(row[2], int, default=None)
        self.ckt = parse_token(row[3], str, default=None).strip()
        if not (k == 0):
            try:
                alert(
                    {'data_type': 'Transformer',
                     'error_message': 'Please model any 3 winding transformer as a configuration of 2 winding transformers',
                     'diagnostics': {
                         'i': self.i,
                         'j': self.j,
                         'k': k,
                         'ckt': self.ckt}})
                raise Exception('3 winding transformers not allowed')
            except Exception as e:
                traceback.print_exc()
                raise e
        # just 2-winding, 4-row
        try:
            if len(row) != 43:
                if len(row) < 43:
                    raise Exception('missing field not allowed')
                elif len(row) > 43:
                    row = remove_end_of_line_comment_from_row(row, '/')
                    if len(row) > new_row_len: # todo: what is new_row_len? need this to handle end of line comments?
                        raise Exception('extra field not allowed')
        except Exception as e:
            traceback.print_exc()
            raise e
        self.mag1 = parse_token(row[7], float, default=None)
        self.mag2 = parse_token(row[8], float, default=None)
        self.stat = parse_token(row[11], int, default=None)
        self.r12 = parse_token(row[21], float, default=None)
        self.x12 = parse_token(row[22], float, default=None)
        self.windv1 = parse_token(row[24], float, default=None)
        self.ang1 = parse_token(row[26], float, default=None)
        self.rata1 = parse_token(row[27], float, default=None)
        self.ratc1 = parse_token(row[29], float, default=None)
        self.windv2 = parse_token(row[41], float, default=None)
        if read_unused_fields:
            self.k = parse_token(row[2], int, 0)
            self.cw = parse_token(row[4], int, 1)
            self.cz = parse_token(row[5], int, 1)
            self.cm = parse_token(row[6], int, 1)
            self.nmetr = parse_token(row[9], int, 2)
            self.name = parse_token(row[10], str, 12*' ')
            self.o1 = parse_token(row[12], int, 1)
            self.f1 = parse_token(row[13], float, 1.0)
            self.o2 = parse_token(row[14], int, 0)
            self.f2 = parse_token(row[15], float, 1.0)
            self.o3 = parse_token(row[16], int, 0)
            self.f3 = parse_token(row[17], float, 1.0)
            self.o4 = parse_token(row[18], int, 0)
            self.f4 = parse_token(row[19], float, 1.0)
            self.vecgrp = parse_token(row[20], str, 12*' ')
            self.sbase12 = parse_token(row[23], float, 0.0)
            self.nomv1 = parse_token(row[25], float, 0.0)
            self.ratb1 = parse_token(row[28], float, 0.0)
            self.cod1 = parse_token(row[30], int, 0)
            self.cont1 = parse_token(row[31], int, 0)
            self.rma1 = parse_token(row[32], float, 1.1)
            self.rmi1 = parse_token(row[33], float, 0.9)
            self.vma1 = parse_token(row[34], float, 1.1)
            self.vmi1 = parse_token(row[35], float, 0.9)
            self.ntp1 = parse_token(row[36], int, 33)
            self.tab1 = parse_token(row[37], int, 0)
            self.cr1 = parse_token(row[38], float, 0.0)
            self.cx1 = parse_token(row[39], float, 0.0)
            self.cnxa1 = parse_token(row[40], float, 0.0)
            self.nomv2 = parse_token(row[42], float, 0.0)

class TransformerImpedanceCorrectionTable:

    def __init__(self):

        self.i = None # no default
        self.t1 = 0.0
        self.f1 = 0.0
        self.t2 = 0.0
        self.f2 = 0.0
        self.t3 = 0.0
        self.f3 = 0.0
        self.t4 = 0.0
        self.f4 = 0.0
        self.t5 = 0.0
        self.f5 = 0.0
        self.t6 = 0.0
        self.f6 = 0.0
        self.t7 = 0.0
        self.f7 = 0.0
        self.t8 = 0.0
        self.f8 = 0.0
        self.t9 = 0.0
        self.f9 = 0.0
        self.t10 = 0.0
        self.f10 = 0.0
        self.t11 = 0.0
        self.f11 = 0.0

        self.t = []
        self.f = []
        self.tict_point_count = 0

    def read_from_row(self, row):

        row = pad_row(row, 23)
        self.i = parse_token(row[0], int, default=None)
        #self.t1 = parse_token(row[1], float, default=0.0)
        #self.f1 = parse_token(row[2], float, default=0.0)
        #self.t2 = parse_token(row[3], float, default=0.0)
        #self.f2 = parse_token(row[4], float, default=0.0)
        self.t1 = parse_token(row[1] if 1 < len(row) else None, float, default=0.0)
        self.f1 = parse_token(row[2] if 2 < len(row) else None, float, default=0.0)
        self.t2 = parse_token(row[3] if 3 < len(row) else None, float, default=0.0)
        self.f2 = parse_token(row[4] if 4 < len(row) else None, float, default=0.0)
        self.t3 = parse_token(row[5] if 5 < len(row) else None, float, default=0.0)
        self.f3 = parse_token(row[6] if 6 < len(row) else None, float, default=0.0)
        self.t4 = parse_token(row[7] if 7 < len(row) else None, float, default=0.0)
        self.f4 = parse_token(row[8] if 8 < len(row) else None, float, default=0.0)
        self.t5 = parse_token(row[9] if 9 < len(row) else None, float, default=0.0)
        self.f5 = parse_token(row[10] if 10 < len(row) else None, float, default=0.0)
        self.t6 = parse_token(row[11] if 11 < len(row) else None, float, default=0.0)
        self.f6 = parse_token(row[12] if 12 < len(row) else None, float, default=0.0)
        self.t7 = parse_token(row[13] if 13 < len(row) else None, float, default=0.0)
        self.f7 = parse_token(row[14] if 14 < len(row) else None, float, default=0.0)
        self.t8 = parse_token(row[15] if 15 < len(row) else None, float, default=0.0)
        self.f8 = parse_token(row[16] if 16 < len(row) else None, float, default=0.0)
        self.t9 = parse_token(row[17] if 17 < len(row) else None, float, default=0.0)
        self.f9 = parse_token(row[18] if 18 < len(row) else None, float, default=0.0)
        self.t10 = parse_token(row[19] if 19 < len(row) else None, float, default=0.0)
        self.f10 = parse_token(row[20] if 20 < len(row) else None, float, default=0.0)
        self.t11 = parse_token(row[21] if 21 < len(row) else None, float, default=0.0)
        self.f11 = parse_token(row[22] if 22 < len(row) else None, float, default=0.0)
        
        #CHALLENGE2
        self.tict_point_count = 11
        for i in range(1,12):
            ti = eval('self.t{}'.format(i))
            fi = eval('self.f{}'.format(i))
            if ti is None or fi is None or fi == 0.0: # note ti==0.0 COULD be valid
                self.tict_point_count = i-1
                break
    
        self.t = [ self.t1, self.t2, self.t3, self.t4, self.t5, self.t6, self.t7, self.t8, self.t9, self.t10, self.t11 ]
        self.f = [ self.f1, self.f2, self.f3, self.f4, self.f5, self.f6, self.f7, self.f8, self.f9, self.f10, self.f11 ]

    def check(self, scrub_mode=False):

        self.check_point_count_ge_2(scrub_mode)
        self.check_i_gt_0(scrub_mode)
        self.check_t_increasing(scrub_mode)

    def check_point_count_ge_2(self, scrub_mode=False):

        if self.tict_point_count < 2:
            alert(
                {'data_type': 'TransformerImpedanceCorrectionTable',
                 'error_message': 'fails num points >= 2. {}'.format(
                        'adding points' if scrub_mode else
                        'scrubber will add points'),
                 'diagnostics': {
                     'i': self.i,
                     't1': self.t1, 'f1': self.f1,
                     't2': self.t2, 'f2': self.f2,
                     't3': self.t3, 'f3': self.f3,
                     't4': self.t4, 'f4': self.f4,
                     't5': self.t5, 'f5': self.f5,
                     't6': self.t6, 'f6': self.f6,
                     't7': self.t7, 'f7': self.f7,
                     't8': self.t8, 'f8': self.f8,
                     't9': self.t9, 'f9': self.f9,
                     't10': self.t10, 'f10': self.f10,
                     't11': self.t11, 'f11': self.f11}})
            if scrub_mode:
                if self.tict_point_count == 1:
                    self.t2 = self.t1 + 1.0
                    self.f2 = self.f1
                    self.tict_point_count = 2
                    self.t[1] = self.t2
                    self.f[1] = self.f2
                else:
                    self.t1 = 0.0
                    self.f1 = 1.0
                    self.t2 = 1.0
                    self.f2 = 1.0
                    self.tict_point_count = 2
                    self.t[0] = self.t1
                    self.f[0] = self.f1
                    self.t[1] = self.t2
                    self.f[1] = self.f2

    def check_i_gt_0(self, scrub_mode=False):

        if self.i <= 0:
            alert(
                {'data_type': 'TransformerImpedanceCorrectionTable',
                 'error_message': 'fails i > 0. scrubber does not fix this',
                'diagnostics': {
                    'i': self.i,
                    't1': self.t1, 'f1': self.f1,
                    't2': self.t2, 'f2': self.f2,
                    't3': self.t3, 'f3': self.f3,
                    't4': self.t4, 'f4': self.f4,
                    't5': self.t5, 'f5': self.f5,
                    't6': self.t6, 'f6': self.f6,
                    't7': self.t7, 'f7': self.f7,
                    't8': self.t8, 'f8': self.f8,
                    't9': self.t9, 'f9': self.f9,
                    't10': self.t10, 'f10': self.f10,
                    't11': self.t11, 'f11': self.f11}})
    
    def check_t_increasing(self, scrub_mode=False):

        for i in range(self.tict_point_count - 1):
            if self.t[i] >= self.t[i + 1]:
                alert(
                    {'data_type': 'TransformerImpedanceCorrectionTable',
                     'error_message': 'fails t increasing. scrubber does not fix this',
                     'diagnostics': {
                            'i': self.i,
                            't1': self.t1, 'f1': self.f1,
                            't2': self.t2, 'f2': self.f2,
                            't3': self.t3, 'f3': self.f3,
                            't4': self.t4, 'f4': self.f4,
                            't5': self.t5, 'f5': self.f5,
                            't6': self.t6, 'f6': self.f6,
                            't7': self.t7, 'f7': self.f7,
                            't8': self.t8, 'f8': self.f8,
                            't9': self.t9, 'f9': self.f9,
                            't10': self.t10, 'f10': self.f10,
                            't11': self.t11, 'f11': self.f11}})
                break
            
class Area:

    def __init__(self):

        self.i = None # no default
        self.isw = 0
        self.pdes = 0.0
        self.ptol = 10.0
        self.arname = 12*' '

    def clean_arname(self):

        self.arname = ''

    def check(self):

        self.check_i_pos()

    def check_i_pos(self):
        
        if not(self.i > 0):
            alert(
                {'data_type': 'Area',
                 'error_message': 'fails i positivity. Please ensure that the i field of every area is a positive integer.',
                 'diagnostics': {
                     'i': self.i}})

    def read_from_row(self, row):

        row = pad_row(row, 5)
        self.i = parse_token(row[0], int, default=None)
        if read_unused_fields:
            self.isw = parse_token(row[1], int, 0)
            self.pdes = parse_token(row[2], float, 0.0)
            self.ptol = parse_token(row[3], float, 10.0)
            self.arname = parse_token(row[4], str, 12*' ')

class Zone:

    def __init__(self):

        self.i = None # no default
        self.zoname = 12*' '

    def clean_zoname(self):

        self.zoname = ''

    def check(self):

        self.check_i_pos()

    def check_i_pos(self):
        
        if not(self.i > 0):
            alert(
                {'data_type': 'Zone',
                 'error_message': 'fails i positivity. Please ensure that the i field of every zone is a positive integer.',
                 'diagnostics': {
                     'i': self.i}})
        
    def read_from_row(self, row):

        row = pad_row(row, 2)
        self.i = parse_token(row[0], int, default=None)
        if read_unused_fields:
            self.zoname = parse_token(row[1], str, 12*' ')

class SwitchedShunt:

    def __init__(self):

        self.i = None # no default
        self.id = "1"
        self.modsw = 1
        self.adjm = 0
        self.status = 1
        self.vswhi = 1.0
        self.vswlo = 1.0
        self.swrem = 0
        self.rmpct = 100.0
        self.rmidnt = 12*' '
        self.binit = 0.0
        self.n1 = 0
        self.b1 = 0.0
        self.n2 = 0
        self.b2 = 0.0
        self.n3 = 0
        self.b3 = 0.0
        self.n4 = 0
        self.b4 = 0.0
        self.n5 = 0
        self.b5 = 0.0
        self.n6 = 0
        self.b6 = 0.0
        self.n7 = 0
        self.b7 = 0.0
        self.n8 = 0
        self.b8 = 0.0
        self.swsh_susc_count = 0

    def scrub(self):

        self.scrub_swrem()
        if do_fix_swsh_binit:
            self.scrub_binit()

    def scrub_binit(self):

        b_min_max = self.compute_bmin_bmax()
        bmin = b_min_max[0]
        bmax = b_min_max[1]
        #tolabs = 1e-8
        #tol = max(abs(self.binit) * tolabs, tolabs)
        if self.binit < bmin:
            self.binit = bmin
        elif self.binit > bmax:
            self.binit = bmax

    def scrub_swrem(self):

        self.swrem = 0

    def clean_rmidnt(self):

        self.rmidnt = ''

    def compute_bmin_bmax(self):

        b_min = 0.0
        b_max = 0.0
        b1 = float(self.n1) * self.b1
        b2 = float(self.n2) * self.b2
        b3 = float(self.n3) * self.b3
        b4 = float(self.n4) * self.b4
        b5 = float(self.n5) * self.b5
        b6 = float(self.n6) * self.b6
        b7 = float(self.n7) * self.b7
        b8 = float(self.n8) * self.b8
        for b in [b1, b2, b3, b4, b5, b6, b7, b8]:
            if b > 0.0:
                b_max += b
            elif b < 0.0:
                b_min += b
            else:
                break
        return (b_min, b_max)

    def check(self):
        '''In contrast to challenge 1,
        in challenge 2 the Grid Optimization competition
        uses a discrete model of shunt switching.
        Every switched shunt can be characterized by the following input data:
          i           - bus number
          stat        - status
          n1...n8     - number of steps available in blocks 1...8 (nh is an integer >= 0)
          b1...b8     - step size in blocks 1...8 (bh is a real number)
          binit       - total susceptance in given operating point priot to base case.

        we need to check that there exist x1...x8 such that:
          xh = x^{st,init}_h is an integer
          0 <= xh <= nh
          binit = sum_{h = 1}^8 bh*xh

        in general this requires a MIP or a cobinatorial algorithm to check.
        We cannot do this check in the data checker without substantial additional coding.
        We do ask that data providers ensure it holds.

        Instead we can at least check the weaker requirement that
          bmin <= binit <= bmax

        where
          bmin = sum_{h = 1}^8 min{0, bh} nh
          bmax = sum_{h = 1}^8 max{0, bh} nh
        '''

        #self.check_b1_b2_opposite_signs()
        #self.check_n1_0_implies_b1_0_n2_0_b2_0()
        #self.check_b1_0_implies_n1_0_n2_0_b2_0()
        self.check_n1_nonneg()
        self.check_n2_nonneg()
        self.check_n3_nonneg()
        self.check_n4_nonneg()
        self.check_n5_nonneg()
        self.check_n6_nonneg()
        self.check_n7_nonneg()
        self.check_n8_nonneg()
        self.check_n1_max()
        self.check_n2_max()
        self.check_n3_max()
        self.check_n4_max()
        self.check_n5_max()
        self.check_n6_max()
        self.check_n7_max()
        self.check_n8_max()

        if do_check_bmin_le_binit_le_bmax:
            self.check_bmin_le_binit_le_bmax()
        if do_check_swrem_zero:
            self.check_swrem_zero()

    def check_swrem_zero(self):

        if self.swrem != 0:
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails swrem==0. For each switched shunt, please ensure that the swrem field contains the value 0.',
                 'diagnostics': {
                     'i': self.i,
                     'swrem': self.swrem}})

    def check_b1_b2_opposite_signs(self):

        if (((self.b1 < 0.0) and (self.b2 < 0.0)) or ((self.b1 > 0.0) and (self.b2 > 0.0))):
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails b1,b2 opposite sign requirement. For each switched shunt, please ensure that the fields b1, b2 are real numbers with opposite signs, i.e. if b1 < 0.0, then b2 >= 0.0, and if b1 > 0.0, then b2 <= 0.0. This is a minimal nonzero data requirement.',
                 'diagnostics': {
                     'i': self.i,
                     'n1': self.n1,
                     'b1': self.b1,
                     'n2': self.n2,
                     'b2': self.b2}})

    def check_n1_0_implies_b1_0_n2_0_b2_0(self):

        if ((self.n1 == 0) and ((self.b1 != 0.0) or (self.n2 != 0) or (self.b2 != 0.0))):
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails ((n1==0)->((b1==0.0)&(n2==0)&(b2==0.0))). For each switched shunt, please ensure that the fields n1, b1, n2, b2 satisfy this logical relation. This is a minimal nonzero data requirement.',
                 'diagnostics': {
                     'i': self.i,
                     'n1': self.n1,
                     'b1': self.b1,
                     'n2': self.n2,
                     'b2': self.b2}})

    def check_b1_0_implies_n1_0_n2_0_b2_0(self):

        if ((self.b1 == 0.0) and ((self.n1 != 0) or (self.n2 != 0) or (self.b2 != 0.0))):
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails ((b1==0.0)->((n1==0)&(n2==0)&(b2==0.0))). For each switched shunt, please ensure that the fields n1, b1, n2, b2 satisfy this logical relation. This is a minimal nonzero data requirement.',
                 'diagnostics': {
                     'i': self.i,
                     'n1': self.n1,
                     'b1': self.b1,
                     'n2': self.n2,
                     'b2': self.b2}})

    def check_n1_nonneg(self):

        if self.n1 < 0:
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n1 nonnegativity. Please ensure that the n1 field of every switched shunt is a nonnegative integer.',
                 'diagnostics': {
                     'i': self.i,
                     'n1': self.n1}})
                                                
    def check_n2_nonneg(self):

        if self.n2 < 0:
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n2 nonnegativity. Please ensure that the n2 field of every switched shunt is a nonnegative integer.',
                 'diagnostics': {
                     'i': self.i,
                     'n2': self.n2}})
    
    def check_n3_nonneg(self):

        if self.n3 < 0:
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n3 nonnegativity. Please ensure that the n3 field of every switched shunt is a nonnegative integer.',
                 'diagnostics': {
                     'i': self.i,
                     'n3': self.n3}})
    
    def check_n4_nonneg(self):

        if self.n4 < 0:
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n4 nonnegativity. Please ensure that the n4 field of every switched shunt is a nonnegative integer.',
                 'diagnostics': {
                     'i': self.i,
                     'n4': self.n4}})
    
    def check_n5_nonneg(self):

        if self.n5 < 0:
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n5 nonnegativity. Please ensure that the n5 field of every switched shunt is a nonnegative integer.',
                 'diagnostics': {
                     'i': self.i,
                     'n5': self.n5}})
    
    def check_n6_nonneg(self):

        if self.n6 < 0:
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n6 nonnegativity. Please ensure that the n6 field of every switched shunt is a nonnegative integer.',
                 'diagnostics': {
                     'i': self.i,
                     'n6': self.n6}})
    
    def check_n7_nonneg(self):

        if self.n7 < 0:
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n7 nonnegativity. Please ensure that the n7 field of every switched shunt is a nonnegative integer.',
                 'diagnostics': {
                     'i': self.i,
                     'n7': self.n7}})
    
    def check_n8_nonneg(self):

        if self.n8 < 0:
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n8 nonnegativity. Please ensure that the n8 field of every switched shunt is a nonnegative integer.',
                 'diagnostics': {
                     'i': self.i,
                     'n8': self.n8}})

    def check_n1_max(self):

        if self.n1 > max_swsh_n:
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n1 max value. Please ensure that the n1 field of every switched shunt is <= the maximum value of %u.' % max_swsh_n,
                 'diagnostics': {
                     'i': self.i,
                     'n1': self.n1}})
                                                
    def check_n2_max(self):

        if self.n2 > max_swsh_n:
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n2 max value. Please ensure that the n2 field of every switched shunt is <= the maximum value of %u.' % max_swsh_n,
                 'diagnostics': {
                     'i': self.i,
                     'n2': self.n2}})
    
    def check_n3_max(self):

        if self.n3 > max_swsh_n:
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n3 max value. Please ensure that the n3 field of every switched shunt is <= the maximum value of %u.' % max_swsh_n,
                 'diagnostics': {
                     'i': self.i,
                     'n3': self.n3}})
    
    def check_n4_max(self):

        if self.n4 > max_swsh_n:
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n4 max value. Please ensure that the n4 field of every switched shunt is <= the maximum value of %u.' % max_swsh_n,
                 'diagnostics': {
                     'i': self.i,
                     'n4': self.n4}})
    
    def check_n5_max(self):

        if self.n5 > max_swsh_n:
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n5 max value. Please ensure that the n5 field of every switched shunt is <= the maximum value of %u.' % max_swsh_n,
                 'diagnostics': {
                     'i': self.i,
                     'n5': self.n5}})
    
    def check_n6_max(self):

        if self.n6 > max_swsh_n:
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n6 max value. Please ensure that the n6 field of every switched shunt is <= the maximum value of %u.' % max_swsh_n,
                 'diagnostics': {
                     'i': self.i,
                     'n6': self.n6}})
    
    def check_n7_max(self):

        if self.n7 > max_swsh_n:
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n7 max value. Please ensure that the n7 field of every switched shunt is <= the maximum value of %u.' % max_swsh_n,
                 'diagnostics': {
                     'i': self.i,
                     'n7': self.n7}})
    
    def check_n8_max(self):

        if self.n8 > max_swsh_n:
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n8 max value. Please ensure that the n8 field of every switched shunt is <= the maximum value of %u.' % max_swsh_n,
                 'diagnostics': {
                     'i': self.i,
                     'n8': self.n8}})
    
    def check_n1_le_1(self):

        if self.n1 > 1:
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n1 at most 1. Please ensure that the n1 field of every switched shunt is an integer <= 1.',
                 'diagnostics': {
                     'i': self.i,
                     'n1': self.n1}})
    

    def check_n2_le_1(self):

        if self.n2 > 1:
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n2 at most 1. Please ensure that the n2 field of every switched shunt is an integer <= 1.',
                 'diagnostics': {
                     'i': self.i,
                     'n2': self.n2}})
    
    def check_n3_zero(self):

        if not (self.n3 == 0):
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n3 exactly equal to 0. Please ensure that the n3 field of every switched shunt is exactly equal to 0. Since the Grid Optimization competition uses a continuous susceptance model of shunt switching, every switched shunt can be expressed using only the i,stat,binit,n1,b1,n2,b2 fields by means of a conversion.',
                 'diagnostics': {
                     'i': self.i,
                     'n3': self.n3}})

    def check_n4_zero(self):

        if not (self.n4 == 0):
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n4 exactly equal to 0. Please ensure that the n4 field of every switched shunt is exactly equal to 0. Since the Grid Optimization competition uses a continuous susceptance model of shunt switching, every switched shunt can be expressed using only the i,stat,binit,n1,b1,n2,b2 fields by means of a conversion.',
                 'diagnostics': {
                     'i': self.i,
                     'n4': self.n4}})

    def check_n5_zero(self):

        if not (self.n5 == 0):
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n5 exactly equal to 0. Please ensure that the n5 field of every switched shunt is exactly equal to 0. Since the Grid Optimization competition uses a continuous susceptance model of shunt switching, every switched shunt can be expressed using only the i,stat,binit,n1,b1,n2,b2 fields by means of a conversion.',
                 'diagnostics': {
                     'i': self.i,
                     'n5': self.n5}})

    def check_n6_zero(self):

        if not (self.n6 == 0):
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n6 exactly equal to 0. Please ensure that the n6 field of every switched shunt is exactly equal to 0. Since the Grid Optimization competition uses a continuous susceptance model of shunt switching, every switched shunt can be expressed using only the i,stat,binit,n1,b1,n2,b2 fields by means of a conversion.',
                 'diagnostics': {
                     'i': self.i,
                     'n6': self.n6}})

    def check_n7_zero(self):

        if not (self.n7 == 0):
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n7 exactly equal to 0. Please ensure that the n7 field of every switched shunt is exactly equal to 0. Since the Grid Optimization competition uses a continuous susceptance model of shunt switching, every switched shunt can be expressed using only the i,stat,binit,n1,b1,n2,b2 fields by means of a conversion.',
                 'diagnostics': {
                     'i': self.i,
                     'n7': self.n7}})

    def check_n8_zero(self):

        if not (self.n8 == 0):
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails n8 exactly equal to 0. Please ensure that the n8 field of every switched shunt is exactly equal to 0. Since the Grid Optimization competition uses a continuous susceptance model of shunt switching, every switched shunt can be expressed using only the i,stat,binit,n1,b1,n2,b2 fields by means of a conversion.',
                 'diagnostics': {
                     'i': self.i,
                     'n8': self.n8}})

    def check_b3_zero(self):

        if not (self.b3 == 0.0):
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails b3 exactly equal to 0.0. Please ensure that the b3 field of every switched shunt is exactly equal to 0.0. Since the Grid Optimization competition uses a continuous susceptance model of shunt switching, every switched shunt can be expressed using only the i,stat,binit,n1,b1,n2,b2 fields by means of a conversion.',
                 'diagnostics': {
                     'i': self.i,
                     'b3': self.b3}})

    def check_b4_zero(self):

        if not (self.b4 == 0.0):
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails b4 exactly equal to 0.0. Please ensure that the b4 field of every switched shunt is exactly equal to 0.0. Since the Grid Optimization competition uses a continuous susceptance model of shunt switching, every switched shunt can be expressed using only the i,stat,binit,n1,b1,n2,b2 fields by means of a conversion.',
                 'diagnostics': {
                     'i': self.i,
                     'b4': self.b4}})

    def check_b5_zero(self):

        if not (self.b5 == 0.0):
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails b5 exactly equal to 0.0. Please ensure that the b5 field of every switched shunt is exactly equal to 0.0. Since the Grid Optimization competition uses a continuous susceptance model of shunt switching, every switched shunt can be expressed using only the i,stat,binit,n1,b1,n2,b2 fields by means of a conversion.',
                 'diagnostics': {
                     'i': self.i,
                     'b5': self.b5}})

    def check_b6_zero(self):

        if not (self.b6 == 0.0):
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails b6 exactly equal to 0.0. Please ensure that the b6 field of every switched shunt is exactly equal to 0.0. Since the Grid Optimization competition uses a continuous susceptance model of shunt switching, every switched shunt can be expressed using only the i,stat,binit,n1,b1,n2,b2 fields by means of a conversion.',
                 'diagnostics': {
                     'i': self.i,
                     'b6': self.b6}})

    def check_b7_zero(self):

        if not (self.b7 == 0.0):
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails b7 exactly equal to 0.0. Please ensure that the b7 field of every switched shunt is exactly equal to 0.0. Since the Grid Optimization competition uses a continuous susceptance model of shunt switching, every switched shunt can be expressed using only the i,stat,binit,n1,b1,n2,b2 fields by means of a conversion.',
                 'diagnostics': {
                     'i': self.i,
                     'b7': self.b7}})

    def check_b8_zero(self):

        if not (self.b8 == 0.0):
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails b8 exactly equal to 0.0. Please ensure that the b8 field of every switched shunt is exactly equal to 0.0. Since the Grid Optimization competition uses a continuous susceptance model of shunt switching, every switched shunt can be expressed using only the i,stat,binit,n1,b1,n2,b2 fields by means of a conversion.',
                 'diagnostics': {
                     'i': self.i,
                     'b8': self.b8}})

    def check_bmin_le_binit_le_bmax(self):

        b_min_max = self.compute_bmin_bmax()
        bmin = b_min_max[0]
        bmax = b_min_max[1]
        tol = max(abs(self.binit) * swsh_bmin_bmax_tol, swsh_bmin_bmax_tol)
        if bmin - tol > self.binit:
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails bmin <= binit. Please ensure that bmin <= binit, where bmin is derived from b1, n1, ..., b8, n8 as described in the formulation.',
                 'diagnostics': {
                     'i': self.i,
                     'bmin': bmin,
                     'binit': self.binit,
                     'b1': self.b1,
                     'n1': self.n1,
                     'b2': self.b2,
                     'n2': self.n2,
                     'b3': self.b3,
                     'n3': self.n3,
                     'b4': self.b4,
                     'n4': self.n4,
                     'b5': self.b5,
                     'n5': self.n5,
                     'b6': self.b6,
                     'n6': self.n6,
                     'b7': self.b7,
                     'n7': self.n7,
                     'b8': self.b8,
                     'n8': self.n8}})
        if self.binit > bmax + tol:
            alert(
                {'data_type': 'SwitchedShunt',
                 'error_message': 'fails binit <= bmax. Please ensure that binit <= bmax, where bmin is derived from b1, n1, ..., b8, n8 as described in the formulation.',
                 'diagnostics': {
                     'i': self.i,
                     'bmax': bmax,
                     'binit': self.binit,
                     'b1': self.b1,
                     'n1': self.n1,
                     'b2': self.b2,
                     'n2': self.n2,
                     'b3': self.b3,
                     'n3': self.n3,
                     'b4': self.b4,
                     'n4': self.n4,
                     'b5': self.b5,
                     'n5': self.n5,
                     'b6': self.b6,
                     'n6': self.n6,
                     'b7': self.b7,
                     'n7': self.n7,
                     'b8': self.b8,
                     'n8': self.n8}})

    def read_from_row(self, row):

        #print(row)
        #if int(row[0]) == 23393:
        #    print(row)
        row = pad_row(row, 26)
        self.i = parse_token(row[0], int, default=None)
        self.stat = parse_token(row[3], int, default=None)
        self.binit = parse_token(row[9], float, default=None)
        #self.n1 = parse_token(row[10], int, default=None) # allow 0 blocks
        #self.b1 = parse_token(row[11], float, default=None)
        self.n1 = parse_token(row[10] , int, default=0)     if 10 < len(row) else 0
        self.b1 = parse_token(row[11] , float, default=0.0) if 11 < len(row) else 0.0
        self.n2 = parse_token(row[12] , int, default=0)     if 12 < len(row) else 0
        self.b2 = parse_token(row[13] , float, default=0.0) if 13 < len(row) else 0.0
        self.n3 = parse_token(row[14] , int, default=0)     if 14 < len(row) else 0
        self.b3 = parse_token(row[15] , float, default=0.0) if 15 < len(row) else 0.0
        self.n4 = parse_token(row[16] , int, default=0)     if 16 < len(row) else 0
        self.b4 = parse_token(row[17] , float, default=0.0) if 17 < len(row) else 0.0
        self.n5 = parse_token(row[18] , int, default=0)     if 18 < len(row) else 0
        self.b5 = parse_token(row[19] , float, default=0.0) if 19 < len(row) else 0.0
        self.n6 = parse_token(row[20] , int, default=0)     if 20 < len(row) else 0
        self.b6 = parse_token(row[21] , float, default=0.0) if 21 < len(row) else 0.0
        self.n7 = parse_token(row[22] , int, default=0)     if 22 < len(row) else 0
        self.b7 = parse_token(row[23] , float, default=0.0) if 23 < len(row) else 0.0
        self.n8 = parse_token(row[24] , int, default=0)     if 24 < len(row) else 0
        self.b8 = parse_token(row[25] , float, default=0.0) if 25 < len(row) else 0.0
        if read_unused_fields:
            self.modsw = parse_token(row[1], int, 1)
            self.adjm = parse_token(row[2], int, 0)
            self.vswhi = parse_token(row[4], float, 1.0)
            self.vswlo = parse_token(row[5], float, 1.0)
            self.swrem = parse_token(row[6], int, 0)
            self.rmpct = parse_token(row[7], float, 100.0)
            self.rmidnt = parse_token(row[8], str, 12*' ')
        
        #CHALLENGE2
        self.swsh_susc_count = 8
        for i in range(1,9):
            ni = eval('self.n{}'.format(i))
            bi = eval('self.b{}'.format(i))
            if ni is None or bi is None or ni == 0 or bi == 0.0:
                self.swsh_susc_count = i-1
                break

class Contingency:

    def __init__(self):

        self.label = ''
        self.branch_out_events = []
        self.generator_out_events = []

    def check(self):

        self.check_label()
        self.check_branch_out_events()
        self.check_generator_out_events()
        self.check_at_most_one_branch_out_event()
        self.check_at_most_one_generator_out_event()
        self.check_at_most_one_branch_or_generator_out_event()
        self.check_at_least_one_branch_or_generator_out_event()
        # need to check that each outaged component is active in the base case

    def clean_label(self):
        '''remove spaces and non-allowed characters
        better to just give each contingency a label that is a positive integer'''

        # todo
        # definitely remove whitespace and quote characters
        # this is currently done implicitly in the Con.read()
        pass

    def check_label(self):
        '''check that there are no spaces or non-allowed characters'''

        # todo
        # at least make sure there are no whitespace or quote characters
        # currently implcit in Con.read()
        pass

    def check_branch_out_events(self):

        for r in self.branch_out_events:
            r.check()

    def check_generator_out_events(self):

        for r in self.generator_out_events:
            r.check()

    def check_at_most_one_branch_out_event(self):

        if len(self.branch_out_events) > 1:
            alert(
                {'data_type': 'Contingency',
                 'error_message': 'fails at most 1 branch out event. Please ensure that each contingency has at most 1 branch out event.',
                 'diagnostics':{
                     'label': self.label,
                     'num branch out events': len(self.branch_out_events)}})

    def check_at_most_one_generator_out_event(self):

        if len(self.generator_out_events) > 1:
            alert(
                {'data_type': 'Contingency',
                 'error_message': 'fails at most 1 generator out event. Please ensure that each contingency has at most 1 generator out event.',
                 'diagnostics':{
                     'label': self.label,
                     'num generator out events': len(self.generator_out_events)}})

    def check_at_most_one_branch_or_generator_out_event(self):

        if len(self.branch_out_events) + len(self.generator_out_events) > 1:
            alert(
                {'data_type': 'Contingency',
                 'error_message': 'fails at most 1 branch or generator out event. Please ensure that each contingency has at most 1 branch or generator out event.',
                 'diagnostics':{
                     'label': self.label,
                     'num branch out events + num generator out events': len(self.branch_out_events) + len(self.generator_out_events)}})

    def check_at_least_one_branch_or_generator_out_event(self):

        if len(self.branch_out_events) + len(self.generator_out_events) < 1:
            alert(
                {'data_type': 'Contingency',
                 'error_message': 'fails at least 1 branch or generator out event. Please ensure that each contingency has at least 1 branch or generator out event.',
                 'diagnostics':{
                     'label': self.label,
                     'num branch out events + num generator out events': len(self.branch_out_events) + len(self.generator_out_events)}})

    def construct_record_rows(self):

        rows = (
            [['CONTINGENCY', self.label]] +
            [r.construct_record_row()
             for r in self.branch_out_events] +
            [r.construct_record_row()
             for r in self.generator_out_events] +
            [['END']])
        return rows

class Point:

    def __init__(self):

        self.x = None
        self.y = None

    def check(self):

        pass

    def read_from_row(self, row):

        row = pad_row(row, 2)
        self.x = parse_token(row[0], float, default=None)
        self.y = parse_token(row[1], float, default=None)

class BranchOutEvent:

    def __init__(self):

        self.i = None
        self.j = None
        self.ckt = None

    def check(self):

        pass
        # need to check (i,j,ckt) is either a line or a transformer
        # need to check that it is active in the base case

    def read_from_row(self, row):

        check_row_missing_fields(row, 10)
        self.i = parse_token(row[4], int, default=None)
        self.j = parse_token(row[7], int, default=None)
        self.ckt = parse_token(row[9], str, default=None).strip()

    def read_from_csv(self, row):

        self.i = parse_token(row[2], int, '')
        self.j = parse_token(row[3], int, '')
        self.ckt = parse_token(row[4], str, '1')

    '''
    def read_three_winding_from_row(self, row):

        row = pad_row(row, 13)
        self.i = parse_token(row[4], int, '')
        self.j = parse_token(row[7], int, '')
        self.k = parse_token(row[10], int, '')
        self.ckt = parse_token(row[12], str, '1')
    '''

    def construct_record_row(self):

        return ['OPEN', 'BRANCH', 'FROM', 'BUS', self.i, 'TO', 'BUS', self.j, 'CIRCUIT', self.ckt]

class GeneratorOutEvent:

    def __init__(self):

        self.i = None
        self.id = None

    def check(self):

        pass
        # need to check that (i,id) is a generator and that it is active in the base case

    def read_from_csv(self, row):

        self.i = parse_token(row[2], int, '')
        self.id = parse_token(row[3], str, '')

    def read_from_row(self, row):

        self.i = parse_token(row[5], int, default=None)
        self.id = parse_token(row[2], str, default=None).strip()

    def construct_record_row(self):

        return ['REMOVE', 'UNIT', self.id, 'FROM', 'BUS', self.i]
