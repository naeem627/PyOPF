""" Methods for solution format checks, feasibility checks, and evaluation

Author: Jesse Holzer, jesse.holzer@pnnl.gov
Author: Arun Veeramany, arun.veeramany@pnnl.gov

Date: 2020-07-23

"""

"""
MPI USAGE
module load python/3.7.0 gcc openmpi
mpirun -np 6 python evaluation.py division "data-path" "solution-path"

NON-MPI USAGE
python evaluation.py division data-path solution-path
"""

import copy
import csv
import functools
import glob
import inspect
import json
import math
import os
import sys
import time
import traceback
# from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse as sp

try:
    import data_utilities.data as data
    from data_utilities.cost_utils import CostEvaluator
    from data_utilities.swsh_utils import solve_py as swsh_solve
except:
    import data
    from cost_utils import CostEvaluator
    from swsh_utils import solve_py as swsh_solve

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO


print = functools.partial(print, flush=True)

# todo - remove
debug = False
stop_after_ctgs = 10

process_rank = 0
process_count = 1
USE_MPI = False

try:
    from mpi4py import MPI
    print("Found mpi4py for MPI")
    comm = MPI.COMM_WORLD
    process_count = comm.Get_size()
    print('Number of processes: {}'.format(process_count))
    process_rank = comm.Get_rank()
    print(f'evaluation.py process rank: {process_rank}')
    if process_count > 1:
        USE_MPI = True
except:
    USE_MPI = False  


print( "Using MPI" if USE_MPI else "Not using MPI"  )


#cost when outside all blocks
max_block_cost = 1000000.0  # USD/MVar-h.

# tolerance on hard constraints (in the units of the constraint, typically pu)
hard_constr_tol = 1e-4

# pandas float precision for reading from solution files
# None: ordinary converter, 'high', 'round_trip'
pandas_float_precision=None
#pandas_float_precision='round_trip'

#stop_on_errors = True
stop_on_errors = False
active_case = "BASECASE"
active_solution_path = ''
log_fileobject = None

swsh_binit_tol = 1e-4

# todo just make some summary classes
summary_keys = [
    'obj',
    'infeas',
    'total_bus_cost',
    'total_bus_real_cost',
    'total_bus_imag_cost',
    'total_load_benefit',
    'total_gen_cost',
    'total_gen_energy_cost',
    'total_gen_on_cost',
    'total_gen_su_cost',
    'total_gen_sd_cost',
    'total_line_cost',
    'total_line_limit_cost',
    'total_line_switch_cost',
    'total_xfmr_cost',
    'total_xfmr_limit_cost',
    'total_xfmr_switch_cost',
    'min_total_load_benefit',
    'max_total_load_benefit',
    'bus_volt_mag_max_viol',
    'bus_volt_mag_min_viol',
    'bus_pow_real_imbalance',
    'bus_pow_imag_imbalance',
    'bus_cost',
    'load_t_max_viol',
    'load_t_min_viol',
    'load_ramp_up_max_viol',
    'load_ramp_down_max_viol',
    'load_benefit',
    'gen_xon_max',
    'gen_xon_min',
    'gen_su_qual',
    'gen_sd_qual',
    'gen_su_sd_not_both',
    'gen_sd_su_not_both',
    'gen_pow_real_max_viol',
    'gen_pow_real_min_viol',
    'gen_pow_imag_max_viol',
    'gen_pow_imag_min_viol',
    'gen_pow_real_0_if_x_0_viol',
    'gen_pow_imag_0_if_x_0_viol',
    'gen_ramp_up_max_viol',
    'gen_ramp_down_max_viol',
    'gen_cost',
    'gen_switch_up_max',
    'gen_switch_up_actual',
    'gen_switch_down_max',
    'gen_switch_down_actual',
    'line_xsw_max',
    'line_xsw_min',
    'line_sw_qual',
    'line_cost',
    'line_switch_up_max',
    'line_switch_up_actual',
    'line_switch_down_max',
    'line_switch_down_actual',
    'xfmr_xsw_max',
    'xfmr_xsw_min',
    'xfmr_sw_qual',
    'xfmr_xst_bounds',
    'xfmr_cost',
    'xfmr_switch_up_max',
    'xfmr_switch_up_actual',
    'xfmr_switch_down_max',
    'xfmr_switch_down_actual',
    'swsh_xst_max',
    'swsh_xst_min',
    'max_bus_pow_real_over',
    'max_bus_pow_real_under',
    'max_bus_pow_imag_over',
    'max_bus_pow_imag_under',
    'sum_bus_pow_real_over',
    'sum_bus_pow_real_under',
    'sum_bus_pow_imag_over',
    'sum_bus_pow_imag_under',
    'bus_volt_mag_delta_to_prior',
    'bus_volt_ang_delta_to_prior',
    'load_pow_real_delta_to_prior',
    'gen_pow_real_delta_to_prior',
    'gen_pow_imag_delta_to_prior',
    'xfmr_tau_delta_to_prior',
    'xfmr_phi_delta_to_prior',
    'swsh_b_delta_to_prior',
    'max_line_viol',
    'max_xfmr_viol',
]
summary_out = {
    'infeas':False,
    'key':None,
    'val':0.0,
}
summary_out_keys = sorted(list(summary_out.keys()))

summary2_keys = [
    "solutions_exist",
    "infeas_cumulative",
    "infeas_all_cases",
    "obj_cumulative",
    "obj_all_cases",
    "obj",
    "infeas",
    "total_bus_cost",
    "total_load_benefit",
    "total_gen_cost",
    "total_line_cost",
    "total_xfmr_cost",
    "min_total_load_benefit",
    "max_total_load_benefit",
    "base_gen_switch_up_actual",
    "base_gen_switch_up_max",
    "base_gen_switch_down_actual",
    "base_gen_switch_down_max",
    "base_line_switch_up_actual",
    "base_line_switch_up_max",
    "base_line_switch_down_actual",
    "base_line_switch_down_max",
    "base_xfmr_switch_up_actual",
    "base_xfmr_switch_up_max",
    "base_xfmr_switch_down_actual",
    "base_xfmr_switch_down_max",
    "ctg_gen_switch_up_actual",
    "ctg_gen_switch_up_max",
    "ctg_gen_switch_down_actual",
    "ctg_gen_switch_down_max",
    "ctg_line_switch_up_actual",
    "ctg_line_switch_up_max",
    "ctg_line_switch_down_actual",
    "ctg_line_switch_down_max",
    "ctg_xfmr_switch_up_actual",
    "ctg_xfmr_switch_up_max",
    "ctg_xfmr_switch_down_actual",
    "ctg_xfmr_switch_down_max",
    "base_gen_switches",
    "base_line_switches",
    "base_xfmr_switches",
    "ctg_gen_switches",
    "ctg_line_switches",
    "ctg_xfmr_switches",
    "base_total_switches",
    "ctg_total_switches",
    "total_switches",
    "base_obj",
    "ctg_obj",
    "base_infeas",
    "ctg_infeas",
    "base_total_bus_cost",
    "ctg_total_bus_cost",
    "base_total_bus_real_cost",
    "ctg_total_bus_real_cost",
    "total_bus_real_cost",
    "base_total_bus_imag_cost",
    "ctg_total_bus_imag_cost",
    "total_bus_imag_cost",
    "base_total_load_benefit",
    "ctg_total_load_benefit",
    "base_total_gen_cost",
    "ctg_total_gen_cost",
    "base_total_gen_energy_cost",
    "ctg_total_gen_energy_cost",
    "total_gen_energy_cost",
    "base_total_gen_on_cost",
    "ctg_total_gen_on_cost",
    "total_gen_on_cost",
    "base_total_gen_su_cost",
    "ctg_total_gen_su_cost",
    "total_gen_su_cost",
    "base_total_gen_sd_cost",
    "ctg_total_gen_sd_cost",
    "total_gen_sd_cost",
    "base_total_line_cost",
    "ctg_total_line_cost",
    "total_line_limit_cost",
    "base_total_line_limit_cost",
    "ctg_total_line_limit_cost",
    "base_total_line_switch_cost",
    "ctg_total_line_switch_cost",
    "total_line_switch_cost",
    "base_total_xfmr_cost",
    "ctg_total_xfmr_cost",
    "base_total_xfmr_limit_cost",
    "ctg_total_xfmr_limit_cost",
    "total_xfmr_limit_cost",
    "base_total_xfmr_switch_cost",
    "ctg_total_xfmr_switch_cost",
    "total_xfmr_switch_cost",
    "base_min_total_load_benefit",
    "ctg_min_total_load_benefit",
    "base_max_total_load_benefit",
    "ctg_max_total_load_benefit",
    "base_max_bus_pow_real_over",
    "ctg_max_bus_pow_real_over",
    "max_bus_pow_real_over",
    "base_max_bus_pow_real_under",
    "ctg_max_bus_pow_real_under",
    "max_bus_pow_real_under",
    "base_max_bus_pow_real",
    "ctg_max_bus_pow_real",
    "max_bus_pow_real",
    "base_max_bus_pow_imag_over",
    "ctg_max_bus_pow_imag_over",
    "max_bus_pow_imag_over",
    "base_max_bus_pow_imag_under",
    "ctg_max_bus_pow_imag_under",
    "max_bus_pow_imag_under",
    "base_max_bus_pow_imag",
    "ctg_max_bus_pow_imag",
    "max_bus_pow_imag",
    "base_max_bus_pow",
    "ctg_max_bus_pow",
    "max_bus_pow",
    "base_sum_bus_pow_real_over",
    "ctg_sum_bus_pow_real_over",
    "sum_bus_pow_real_over",
    "base_sum_bus_pow_real_under",
    "ctg_sum_bus_pow_real_under",
    "sum_bus_pow_real_under",
    "base_sum_bus_pow_real_net",
    "ctg_sum_bus_pow_real_net",
    "sum_bus_pow_real_net",
    "base_sum_bus_pow_imag_over",
    "ctg_sum_bus_pow_imag_over",
    "sum_bus_pow_imag_over",
    "base_sum_bus_pow_imag_under",
    "ctg_sum_bus_pow_imag_under",
    "sum_bus_pow_imag_under",
    "base_sum_bus_pow_imag_net",
    "ctg_sum_bus_pow_imag_net",
    "sum_bus_pow_imag_net",
    "base_bus_volt_mag_max_viol",
    "ctg_bus_volt_mag_max_viol",
    "bus_volt_mag_max_viol",
    "base_bus_volt_mag_min_viol",
    "ctg_bus_volt_mag_min_viol",
    "bus_volt_mag_min_viol",
    "base_load_t_max_viol",
    "ctg_load_t_max_viol",
    "load_t_max_viol",
    "base_load_t_min_viol",
    "ctg_load_t_min_viol",
    "load_t_min_viol",
    "base_load_ramp_up_max_viol",
    "ctg_load_ramp_up_max_viol",
    "load_ramp_up_max_viol",
    "base_load_ramp_down_max_viol",
    "ctg_load_ramp_down_max_viol",
    "load_ramp_down_max_viol",
    "base_gen_pow_real_max_viol",
    "ctg_gen_pow_real_max_viol",
    "gen_pow_real_max_viol",
    "base_gen_pow_real_min_viol",
    "ctg_gen_pow_real_min_viol",
    "gen_pow_real_min_viol",
    "base_gen_pow_imag_max_viol",
    "ctg_gen_pow_imag_max_viol",
    "gen_pow_imag_max_viol",
    "base_gen_pow_imag_min_viol",
    "ctg_gen_pow_imag_min_viol",
    "gen_pow_imag_min_viol",
    "base_gen_pow_real_0_if_x_0_viol",
    "ctg_gen_pow_real_0_if_x_0_viol",
    "gen_pow_real_0_if_x_0_viol",
    "base_gen_pow_imag_0_if_x_0_viol",
    "ctg_gen_pow_imag_0_if_x_0_viol",
    "gen_pow_imag_0_if_x_0_viol",
    "base_gen_ramp_up_max_viol",
    "ctg_gen_ramp_up_max_viol",
    "gen_ramp_up_max_viol",
    "base_gen_ramp_down_max_viol",
    "ctg_gen_ramp_down_max_viol",
    "gen_ramp_down_max_viol",
    "worst_case_ctg_obj",
    "worst_case_obj",
    "num_bus",
    "num_branch",
    "num_line",
    "num_xfmr",
    "num_xfmr_fx_tau_fx_phi",
    "num_xfmr_var_tau_fx_phi",
    "num_xfmr_fx_tau_var_phi",
    "num_xfmr_imp_corr",
    "num_gen",
    "num_load",
    "num_sh",
    "num_fxsh",
    "num_swsh",
    "num_ctg",
    "num_ctg_gen",
    "num_ctg_line",
    "num_ctg_xfmr",
    "num_ctg_branch",
    "prior_point_pow_real_over",
    "prior_point_pow_real_under",
    "prior_point_pow_imag_over",
    "prior_point_pow_imag_under",
    "prior_point_pow_real_net",
    "prior_point_pow_imag_net",
    "base_bus_volt_mag_delta_to_prior",
    "ctg_bus_volt_mag_delta_to_prior",
    "bus_volt_mag_delta_to_prior",
    "base_bus_volt_ang_delta_to_prior",
    "ctg_bus_volt_ang_delta_to_prior",
    "bus_volt_ang_delta_to_prior",
    "base_load_pow_real_delta_to_prior",
    "ctg_load_pow_real_delta_to_prior",
    "load_pow_real_delta_to_prior",
    "base_gen_pow_real_delta_to_prior",
    "ctg_gen_pow_real_delta_to_prior",
    "gen_pow_real_delta_to_prior",
    "base_gen_pow_imag_delta_to_prior",
    "ctg_gen_pow_imag_delta_to_prior",
    "gen_pow_imag_delta_to_prior",
    "base_xfmr_tau_delta_to_prior",
    "ctg_xfmr_tau_delta_to_prior",
    "xfmr_tau_delta_to_prior",
    "base_xfmr_phi_delta_to_prior",
    "ctg_xfmr_phi_delta_to_prior",
    "xfmr_phi_delta_to_prior",
    "base_swsh_b_delta_to_prior",
    "ctg_swsh_b_delta_to_prior",
    "swsh_b_delta_to_prior",
    "base_max_line_viol",
    "ctg_max_line_viol",
    "max_line_viol",
    "base_max_xfmr_viol",
    "ctg_max_xfmr_viol",
    "max_xfmr_viol",
    "prior_max_bus_pow_real_over",
    "prior_sum_bus_pow_real_over",
    "prior_max_bus_pow_real_under",
    "prior_sum_bus_pow_real_under",
    "prior_bus_pow_real_imbalance",
    "prior_max_bus_pow_imag_over",
    "prior_sum_bus_pow_imag_over",
    "prior_max_bus_pow_imag_under",
    "prior_sum_bus_pow_imag_under",
    "prior_bus_pow_imag_imbalance",
]

check_summary_keys = True
#<base/ctg>_<gen/line/xfmr>_switch_<up/down>_<actual/max>

class Configuration:

    def __init__(self):
        
        # current default is to project all integer variables
        # but not cts variables

        self.proj_bus_v = False
        self.proj_load_t = False # tmin/tmax
        self.proj_load_p_ramp = False # ramping constraints on p imply constraints on t
        self.proj_gen_p = False # just pmin/pmax
        self.proj_gen_p_ramp = False # includes ramping constraints: project current p relative to fixed prior p
        self.proj_gen_q = False

        # integer variables are rounded and projected anyway
        # and we do not use the cfg params to control this
        self.proj_gen_x = True
        self.proj_line_x = True
        self.proj_xfmr_x = True
        self.proj_xfmr_xst = True
        self.proj_swsh_xst = True

        # test non-default settings
        # self.proj_bus_v = True
        # self.proj_load_t = True # tmin/tmax
        # self.proj_load_p_ramp = True # ramping constraints on p imply constraints on t
        # self.proj_gen_p = True # just pmin/pmax
        # self.proj_gen_p_ramp = True # includes ramping constraints: project current p relative to fixed prior p
        # self.proj_gen_q = True

def get_summary_keys():

    return summary2_keys

def write_summary_keys(file_name):

    if file_name.endswith('.json'):
        with open(file_name, 'w') as out_file:
            json.dump(summary2_keys, out_file)
    elif file_name.endswith('.csv'):
        #import csv
        with open(file_name, mode='w') as out_file:
            writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(summary2_keys)        
    else:
        revised_file_name = file_name + '.json'
        print('unsupported file type: {}'.format(file_name))
        print('supported output file types are: {}'.format(['csv', 'json']))
        print('writing summary keys to {}'.format(revised_file_name))
        with open(revised_file_name, 'w') as out_file:
            json.dump(summary2_keys, out_file)

def compute_swsh_block_xst(h_b0, ha_n, ha_b):

    num_swsh = ha_n.shape[0]
    num_block = ha_n.shape[1]
    assert(ha_n.shape[0] == ha_b.shape[0])
    assert(ha_n.shape[1] == ha_b.shape[1])
    assert(h_b0.shape[0] == num_swsh)
    ha_x = np.zeros(shape=(num_swsh, num_block), dtype=int)
    h_r = np.zeros(shape=(num_swsh,), dtype=float)
    h_r_abs = np.zeros(shape=(num_swsh,), dtype=float)
    tol = swsh_binit_tol
    swsh_solve(h_b0, ha_n, ha_b, ha_x, h_r, h_r_abs, tol)
    return ha_x

def create_new_summary():
    
    summary = {k: copy.deepcopy(summary_out) for k in summary_keys}
        # k: {
        #     j: None
        #     for j in summary_out_keys}
        # for k in summary_keys}
    return summary

def create_new_summary2():

    summary2 = {k: 0.0 for k in summary2_keys}
    summary2['solutions_exist'] = True
    summary2['obj_all_cases'] = {}
    summary2['infeas_all_cases'] = {}
    return summary2

def flatten_summary(summary):

    flat = flatten_dict(summary)
    keys = flat.keys()
    keys = sorted(list(keys))
    values = [flat[k] for k in keys]
    return {'keys':keys, 'values':values}
    #return summary
    #return flatten_dict(summary)

# source:
# https://www.geeksforgeeks.org/python-convert-nested-dictionary-into-flattened-dictionary/
# accessed 2020-09-28
def flatten_dict(dd, separator ='_', prefix =''): 
    return { prefix + separator + k if prefix else k : v 
             for kk, vv in dd.items() 
             for k, v in flatten_dict(vv, separator, kk).items() 
             } if isinstance(dd, dict) else { prefix : dd }

def uncaught_exception_handler(exc_type, exc_value, exc_traceback):
    # Do not print exception when user cancels the program
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    print_alert("An uncaught exception occurred:", False)
    print_alert("Type: {}".format( exc_type),False)
    print_alert("Value: {}".format( exc_value),False)

    if exc_traceback:
        format_exception = traceback.format_tb(exc_traceback)
        for line in format_exception:
            print_alert(repr(line),False)

sys.excepthook = uncaught_exception_handler

def print_alert(message,  raise_exception = stop_on_errors, check_passed = None, evaluation=None ):
    if active_case == "BASECASE" and process_rank != 0:
        return

    check_status_msg = ""
    if  check_passed == True:
        check_status_msg = "PASS"
    elif  check_passed == False:
        check_status_msg = "FAIL"

    formatted_message = "[{}] [{} -{}]: {}\n".format(
        check_status_msg,
        process_rank, 
        active_case, 
        message)

    # Note: the appearance of this message in the log can be used to trigger the termination of a submission.
    # be careful
    #if raise_exception and check_passed != True:
    #if check_passed != True:
    if check_passed == False:
        formatted_message += "infeasibility:1\n\n"


    global log_fileobject

    try:
        log_fileobject.write(formatted_message)
    except:
        # why might this fail?
        #print(formatted_message)
        #print('passing exception on writing to log_fileobject')
        pass  

    #if raise_exception and check_passed != True:
    if check_passed == False:
        if evaluation is not None:
            if not evaluation.summary_written:
                #evaluation.write_detail(eval_out_path, active_case, detail_csv=True)
                evaluation.write_detail(eval_out_path, active_case, detail_json=True)
                evaluation.summary_written = True

                # todo 1
                # move all this to a shutdown method
                # call shutdown only when returning
                # evaluation.json_to_summary_all_cases(eval_out_path)
                # evaluation.summary_all_cases_to_summary()
                # evaluation.write_summary_json(eval_out_path)
                # evaluation.write_summary_csv(eval_out_path)
                # evaluation.write_detail_all_cases_json(eval_out_path)
                # evaluation.write_detail_all_cases_csv(eval_out_path)

        #print('stop_on_errors: {}, not raising exception')
        #print(formatted_message)        

        #raise Exception(formatted_message)

def print_info(message):
    print(message)
    print_alert(message, raise_exception=False)
    #pass

def timeit(function):
    def timed(*args, **kw):
        start_time = time.time()
        result = function(*args, **kw)
        end_time = time.time()
        print('function: {}, time: {}'.format(function.__name__, end_time - start_time))
        return result
    return timed

#CHALLENGE2
def eval_piecewise_linear_penalty(residual, penalty_block_max, penalty_block_coeff):
    '''residual, penaltyblock_max, penalty_block_coeff are 1-dimensional numpy arrays'''

    r = residual

    num_block = len(penalty_block_coeff)
    num_block_bounded = len(penalty_block_max)
    assert(num_block_bounded  == num_block)
    num_resid = r.size
    abs_resid = np.abs(r)
    remaining_resid = abs_resid
    penalty = np.zeros(num_resid)
    for i in range(num_block):
        if i < num_block - 1:
            block_coeff = penalty_block_coeff[i]
            block_max = penalty_block_max[i]
            penalized_resid = np.minimum(block_max, remaining_resid)
            penalty += block_coeff * penalized_resid
            remaining_resid -= penalized_resid
        else:
            penalty += max_block_cost * remaining_resid
    return penalty

# todo - remove this, make sure it is handled in the scrubber
def clean_string(s):
    #t = s.replace("'","").replace('"','').replace(' ','')
    ###t = str(s).replace("'","").replace('"','').replace(' ','')
    #return t
    return s

def count_lines(file_name):
    return rawgencount(file_name)

def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)

def rawgencount(filename):
    start_time = time.time()
    f = open(filename, 'rb')
    f_gen = _make_gen(f.read) # py2
    #f_gen = _make_gen(f.raw.read) # py3
    count = sum( buf.count(b'/n') for buf in f_gen )
    f.close()
    end_time = time.time()
    print('rawgencount time: %f' % (end_time - start_time))
    return count

class Evaluation:
    '''In per unit convention, i.e. same as the model'''

    def __init__(self):

        self.cfg = Configuration()
        self.check_contingencies = True # set to false to check only the base case and then return
        self.line_switching_allowed = True
        self.xfmr_switching_allowed = True
        self.summary = create_new_summary() # todo refactor: call this detail
        self.summary_all_cases = {} # todo refactor: call this detail_all_cases
        self.summary2 = create_new_summary2() # todo refactor: call this summary
        self.infeas = True
        #self.infeas_cumulative = False
        #self.infeas_all_cases = {}
        self.obj = -float('inf')
        #self.obj_cumulative = 0.0
        #self.obj_all_cases = {}
        self.detail_csv_header_done = False

        '''
        self.base_gen_switch_up_actual = 0.0
        self.base_gen_switch_up_max = 0.0
        self.base_gen_switch_down_actual = 0.0
        self.base_gen_switch_down_max = 0.0
        self.base_line_switch_up_actual = 0.0
        self.base_line_switch_up_max = 0.0
        self.base_line_switch_down_actual = 0.0
        self.base_line_switch_down_max = 0.0
        self.base_xfmr_switch_up_actual = 0.0
        self.base_xfmr_switch_up_max = 0.0
        self.base_xfmr_switch_down_actual = 0.0
        self.base_xfmr_switch_down_max = 0.0
        self.ctg_gen_switch_up_actual = 0.0
        self.ctg_gen_switch_up_max = 0.0
        self.ctg_gen_switch_down_actual = 0.0
        self.ctg_gen_switch_down_max = 0.0
        self.ctg_line_switch_up_actual = 0.0
        self.ctg_line_switch_up_max = 0.0
        self.ctg_line_switch_down_actual = 0.0
        self.ctg_line_switch_down_max = 0.0
        self.ctg_xfmr_switch_up_actual = 0.0
        self.ctg_xfmr_switch_up_max = 0.0
        self.ctg_xfmr_switch_down_actual = 0.0
        self.ctg_xfmr_switch_down_max = 0.0
        '''

    @timeit
    def write_final_summary_and_detail(self, path):

        self.json_to_summary_all_cases(path)
        self.summary_all_cases_to_summary()
        self.write_summary_json(path)
        self.write_summary_csv(path)
        self.write_detail_all_cases_json(path)
        self.write_detail_all_cases_csv(path)
        #self.json_to_csv(path)

    @timeit
    def add_case_info_to_summary(self):
        '''
        num_bus
        num_branch
        etc.
        '''

        self.summary2['num_bus'] = self.num_bus
        self.summary2['num_branch'] = self.num_line + self.num_xfmr
        self.summary2['num_line'] = self.num_line
        self.summary2['num_xfmr'] = self.num_xfmr
        self.summary2['num_xfmr_fx_tau_fx_phi'] = len(self.xfmr_index_fixed_tap_ratio_and_phase_shift)
        self.summary2['num_xfmr_var_tau_fx_phi'] = len(self.xfmr_index_var_tap_ratio)
        self.summary2['num_xfmr_fx_tau_var_phi'] = len(self.xfmr_index_var_phase_shift)
        self.summary2['num_xfmr_imp_corr'] = len(self.xfmr_index_imp_corr)
        self.summary2['num_gen'] = self.num_gen
        self.summary2['num_load'] = self.num_load
        self.summary2['num_sh'] = self.num_fxsh + self.num_swsh
        self.summary2['num_fxsh'] = self.num_fxsh
        self.summary2['num_swsh'] = self.num_swsh
        self.summary2['num_ctg'] = self.num_ctg
        self.summary2['num_ctg_gen'] = len([i for i in range(self.num_ctg) if self.ctg_num_gens_out[i] > 0])
        self.summary2['num_ctg_line'] = len([i for i in range(self.num_ctg) if self.ctg_num_lines_out[i] > 0])
        self.summary2['num_ctg_xfmr'] = len([i for i in range(self.num_ctg) if self.ctg_num_xfmrs_out[i] > 0])
        self.summary2['num_ctg_branch'] = self.summary2['num_ctg_line'] + self.summary2['num_ctg_xfmr']

    @timeit
    def add_prior_point_imbalance_to_summary(self):
        '''
        TODO
        implement this in such a way as to enable the same method to be used in data checker
        move the code from the data checker into a function here
        '''

        self.summary2["prior_point_pow_real_over"] = 0.0
        self.summary2["prior_point_pow_real_under"] = 0.0
        self.summary2["prior_point_pow_imag_over"] = 0.0
        self.summary2["prior_point_pow_imag_under"] = 0.0
        self.summary2["prior_point_pow_real_net"] = 0.0
        self.summary2["prior_point_pow_imag_net"] = 0.0

    @timeit
    def summary_all_cases_to_summary(self):
        '''construct whole problem summary from summary_all_cases'''

        self.add_case_info_to_summary()

        # can only use contingencies represented in the details
        # this should be the same as the set of all contingencies
        # unless terminating early due to infeasibility
        ctg_labels = sorted(list(set(self.summary_all_cases.keys()).difference(['BASECASE'])))

        # top level objective and feasibility
        self.summary2['base_obj'] = self.summary_all_cases['BASECASE']['obj']['val']
        self.summary2['ctg_obj'] = np.sum([self.summary_all_cases[k]['obj']['val'] for k in ctg_labels]) / self.num_ctg
        if len(ctg_labels) > 0:
            self.summary2['worst_case_ctg_obj'] = np.amin([self.summary_all_cases[k]['obj']['val'] for k in ctg_labels])
        else:
            self.summary2['worst_case_ctg_obj'] = 0.0
        self.summary2['obj'] = self.summary2['base_obj'] + self.summary2['ctg_obj']
        self.summary2['worst_case_obj'] = self.summary2['base_obj'] + self.summary2['worst_case_ctg_obj']

        self.summary2['base_infeas'] = (1.0 if self.summary_all_cases['BASECASE']['infeas']['val'] else 0.0)
        self.summary2['ctg_infeas'] = np.sum([1.0 if self.summary_all_cases[k]['infeas']['val'] else 0.0 for k in ctg_labels])
        self.summary2['infeas'] = self.summary2['base_infeas'] + self.summary2['ctg_infeas']

        # objective components
        self.summary2['base_total_bus_cost'] = self.summary_all_cases['BASECASE']['total_bus_cost']['val']
        self.summary2['ctg_total_bus_cost'] = np.sum([self.summary_all_cases[k]['total_bus_cost']['val'] for k in ctg_labels]) / self.num_ctg
        self.summary2['total_bus_cost'] = self.summary2['base_total_bus_cost'] + self.summary2['ctg_total_bus_cost']

        self.summary2['base_total_bus_real_cost'] = self.summary_all_cases['BASECASE']['total_bus_real_cost']['val']
        self.summary2['ctg_total_bus_real_cost'] = np.sum([self.summary_all_cases[k]['total_bus_real_cost']['val'] for k in ctg_labels]) / self.num_ctg
        self.summary2['total_bus_real_cost'] = self.summary2['base_total_bus_real_cost'] + self.summary2['ctg_total_bus_real_cost']

        self.summary2['base_total_bus_imag_cost'] = self.summary_all_cases['BASECASE']['total_bus_imag_cost']['val']
        self.summary2['ctg_total_bus_imag_cost'] = np.sum([self.summary_all_cases[k]['total_bus_imag_cost']['val'] for k in ctg_labels]) / self.num_ctg
        self.summary2['total_bus_imag_cost'] = self.summary2['base_total_bus_imag_cost'] + self.summary2['ctg_total_bus_imag_cost']

        self.summary2['base_total_load_benefit'] = self.summary_all_cases['BASECASE']['total_load_benefit']['val']
        self.summary2['ctg_total_load_benefit'] = np.sum([self.summary_all_cases[k]['total_load_benefit']['val'] for k in ctg_labels]) / self.num_ctg
        self.summary2['total_load_benefit'] = self.summary2['base_total_load_benefit'] + self.summary2['ctg_total_load_benefit']

        self.summary2['base_total_gen_cost'] = self.summary_all_cases['BASECASE']['total_gen_cost']['val']
        self.summary2['ctg_total_gen_cost'] = np.sum([self.summary_all_cases[k]['total_gen_cost']['val'] for k in ctg_labels]) / self.num_ctg
        self.summary2['total_gen_cost'] = self.summary2['base_total_gen_cost'] + self.summary2['ctg_total_gen_cost']

        self.summary2['base_total_gen_energy_cost'] = self.summary_all_cases['BASECASE']['total_gen_energy_cost']['val']
        self.summary2['ctg_total_gen_energy_cost'] = np.sum([self.summary_all_cases[k]['total_gen_energy_cost']['val'] for k in ctg_labels]) / self.num_ctg
        self.summary2['total_gen_energy_cost'] = self.summary2['base_total_gen_energy_cost'] + self.summary2['ctg_total_gen_energy_cost']

        self.summary2['base_total_gen_on_cost'] = self.summary_all_cases['BASECASE']['total_gen_on_cost']['val']
        self.summary2['ctg_total_gen_on_cost'] = np.sum([self.summary_all_cases[k]['total_gen_on_cost']['val'] for k in ctg_labels]) / self.num_ctg
        self.summary2['total_gen_on_cost'] = self.summary2['base_total_gen_on_cost'] + self.summary2['ctg_total_gen_on_cost']

        self.summary2['base_total_gen_su_cost'] = self.summary_all_cases['BASECASE']['total_gen_su_cost']['val']
        self.summary2['ctg_total_gen_su_cost'] = np.sum([self.summary_all_cases[k]['total_gen_su_cost']['val'] for k in ctg_labels]) / self.num_ctg
        self.summary2['total_gen_su_cost'] = self.summary2['base_total_gen_su_cost'] + self.summary2['ctg_total_gen_su_cost']

        self.summary2['base_total_gen_sd_cost'] = self.summary_all_cases['BASECASE']['total_gen_sd_cost']['val']
        self.summary2['ctg_total_gen_sd_cost'] = np.sum([self.summary_all_cases[k]['total_gen_sd_cost']['val'] for k in ctg_labels]) / self.num_ctg
        self.summary2['total_gen_sd_cost'] = self.summary2['base_total_gen_sd_cost'] + self.summary2['ctg_total_gen_sd_cost']

        self.summary2['base_total_line_cost'] = self.summary_all_cases['BASECASE']['total_line_cost']['val']
        self.summary2['ctg_total_line_cost'] = np.sum([self.summary_all_cases[k]['total_line_cost']['val'] for k in ctg_labels]) / self.num_ctg
        self.summary2['total_line_cost'] = self.summary2['base_total_line_cost'] + self.summary2['ctg_total_line_cost']

        self.summary2['base_total_line_limit_cost'] = self.summary_all_cases['BASECASE']['total_line_limit_cost']['val']
        self.summary2['ctg_total_line_limit_cost'] = np.sum([self.summary_all_cases[k]['total_line_limit_cost']['val'] for k in ctg_labels]) / self.num_ctg
        self.summary2['total_line_limit_cost'] = self.summary2['base_total_line_limit_cost'] + self.summary2['ctg_total_line_limit_cost']
        self.summary2['base_total_line_switch_cost'] = self.summary_all_cases['BASECASE']['total_line_switch_cost']['val']
        self.summary2['ctg_total_line_switch_cost'] = np.sum([self.summary_all_cases[k]['total_line_switch_cost']['val'] for k in ctg_labels]) / self.num_ctg
        self.summary2['total_line_switch_cost'] = self.summary2['base_total_line_switch_cost'] + self.summary2['ctg_total_line_switch_cost']

        self.summary2['base_total_xfmr_cost'] = self.summary_all_cases['BASECASE']['total_xfmr_cost']['val']
        self.summary2['ctg_total_xfmr_cost'] = np.sum([self.summary_all_cases[k]['total_xfmr_cost']['val'] for k in ctg_labels]) / self.num_ctg
        self.summary2['total_xfmr_cost'] = self.summary2['base_total_xfmr_cost'] + self.summary2['ctg_total_xfmr_cost']

        self.summary2['base_total_xfmr_limit_cost'] = self.summary_all_cases['BASECASE']['total_xfmr_limit_cost']['val']
        self.summary2['ctg_total_xfmr_limit_cost'] = np.sum([self.summary_all_cases[k]['total_xfmr_limit_cost']['val'] for k in ctg_labels]) / self.num_ctg
        self.summary2['total_xfmr_limit_cost'] = self.summary2['base_total_xfmr_limit_cost'] + self.summary2['ctg_total_xfmr_limit_cost']

        self.summary2['base_total_xfmr_switch_cost'] = self.summary_all_cases['BASECASE']['total_xfmr_switch_cost']['val']
        self.summary2['ctg_total_xfmr_switch_cost'] = np.sum([self.summary_all_cases[k]['total_xfmr_switch_cost']['val'] for k in ctg_labels]) / self.num_ctg
        self.summary2['total_xfmr_switch_cost'] = self.summary2['base_total_xfmr_switch_cost'] + self.summary2['ctg_total_xfmr_switch_cost']

        self.summary2['base_min_total_load_benefit'] = self.summary_all_cases['BASECASE']['min_total_load_benefit']['val']
        self.summary2['ctg_min_total_load_benefit'] = np.sum([self.summary_all_cases[k]['min_total_load_benefit']['val'] for k in ctg_labels]) / self.num_ctg
        self.summary2['min_total_load_benefit'] = self.summary2['base_min_total_load_benefit'] + self.summary2['ctg_min_total_load_benefit']

        self.summary2['base_max_total_load_benefit'] = self.summary_all_cases['BASECASE']['max_total_load_benefit']['val']
        self.summary2['ctg_max_total_load_benefit'] = np.sum([self.summary_all_cases[k]['max_total_load_benefit']['val'] for k in ctg_labels]) / self.num_ctg
        self.summary2['max_total_load_benefit'] = self.summary2['base_max_total_load_benefit'] + self.summary2['ctg_max_total_load_benefit']

        # switching
        self.summary2['base_gen_switch_up_actual'] = self.summary_all_cases['BASECASE']['gen_switch_up_actual']['val']
        self.summary2['base_gen_switch_up_max'] = self.summary_all_cases['BASECASE']['gen_switch_up_max']['val']
        self.summary2['base_gen_switch_down_actual'] = self.summary_all_cases['BASECASE']['gen_switch_down_actual']['val']
        self.summary2['base_gen_switch_down_max'] = self.summary_all_cases['BASECASE']['gen_switch_down_max']['val']
        self.summary2['base_line_switch_up_actual'] = self.summary_all_cases['BASECASE']['line_switch_up_actual']['val']
        self.summary2['base_line_switch_up_max'] = self.summary_all_cases['BASECASE']['line_switch_up_max']['val']
        self.summary2['base_line_switch_down_actual'] = self.summary_all_cases['BASECASE']['line_switch_down_actual']['val']
        self.summary2['base_line_switch_down_max'] = self.summary_all_cases['BASECASE']['line_switch_down_max']['val']
        self.summary2['base_xfmr_switch_up_actual'] = self.summary_all_cases['BASECASE']['xfmr_switch_up_actual']['val']
        self.summary2['base_xfmr_switch_up_max'] = self.summary_all_cases['BASECASE']['xfmr_switch_up_max']['val']
        self.summary2['base_xfmr_switch_down_actual'] = self.summary_all_cases['BASECASE']['xfmr_switch_down_actual']['val']
        self.summary2['base_xfmr_switch_down_max'] = self.summary_all_cases['BASECASE']['xfmr_switch_down_max']['val']

        self.summary2['ctg_gen_switch_up_actual'] = np.sum([self.summary_all_cases[k]['gen_switch_up_actual']['val'] for k in ctg_labels])
        self.summary2['ctg_gen_switch_up_max'] = np.sum([self.summary_all_cases[k]['gen_switch_up_max']['val'] for k in ctg_labels])
        self.summary2['ctg_gen_switch_down_actual'] = np.sum([self.summary_all_cases[k]['gen_switch_down_actual']['val'] for k in ctg_labels])
        self.summary2['ctg_gen_switch_down_max'] = np.sum([self.summary_all_cases[k]['gen_switch_down_max']['val'] for k in ctg_labels])
        self.summary2['ctg_line_switch_up_actual'] = np.sum([self.summary_all_cases[k]['line_switch_up_actual']['val'] for k in ctg_labels])
        self.summary2['ctg_line_switch_up_max'] = np.sum([self.summary_all_cases[k]['line_switch_up_max']['val'] for k in ctg_labels])
        self.summary2['ctg_line_switch_down_actual'] = np.sum([self.summary_all_cases[k]['line_switch_down_actual']['val'] for k in ctg_labels])
        self.summary2['ctg_line_switch_down_max'] = np.sum([self.summary_all_cases[k]['line_switch_down_max']['val'] for k in ctg_labels])
        self.summary2['ctg_xfmr_switch_up_actual'] = np.sum([self.summary_all_cases[k]['xfmr_switch_up_actual']['val'] for k in ctg_labels])
        self.summary2['ctg_xfmr_switch_up_max'] = np.sum([self.summary_all_cases[k]['xfmr_switch_up_max']['val'] for k in ctg_labels])
        self.summary2['ctg_xfmr_switch_down_actual'] = np.sum([self.summary_all_cases[k]['xfmr_switch_down_actual']['val'] for k in ctg_labels])
        self.summary2['ctg_xfmr_switch_down_max'] = np.sum([self.summary_all_cases[k]['xfmr_switch_down_max']['val'] for k in ctg_labels])

        self.summary2['base_gen_switches'] = self.summary2['base_gen_switch_up_actual'] + self.summary2['base_gen_switch_down_actual']
        self.summary2['base_line_switches'] = self.summary2['base_line_switch_up_actual'] + self.summary2['base_line_switch_down_actual']
        self.summary2['base_xfmr_switches'] = self.summary2['base_xfmr_switch_up_actual'] + self.summary2['base_xfmr_switch_down_actual']
        self.summary2['ctg_gen_switches'] = self.summary2['ctg_gen_switch_up_actual'] + self.summary2['ctg_gen_switch_down_actual']
        self.summary2['ctg_line_switches'] = self.summary2['ctg_line_switch_up_actual'] + self.summary2['ctg_line_switch_down_actual']
        self.summary2['ctg_xfmr_switches'] = self.summary2['ctg_xfmr_switch_up_actual'] + self.summary2['ctg_xfmr_switch_down_actual']
        self.summary2['base_total_switches'] = (
            self.summary2['base_gen_switches'] +
            self.summary2['base_line_switches'] +
            self.summary2['base_xfmr_switches'])
        self.summary2['ctg_total_switches'] = (
            self.summary2['ctg_gen_switches'] +
            self.summary2['ctg_line_switches'] +
            self.summary2['ctg_xfmr_switches'])
        self.summary2['total_switches'] = self.summary2['base_total_switches'] + self.summary2['ctg_total_switches']

        # hard constraint violations
        self.summary2['base_bus_volt_mag_max_viol'] = self.summary_all_cases['BASECASE']['bus_volt_mag_max_viol']['val']
        self.summary2['ctg_bus_volt_mag_max_viol'] = np.amax([self.summary_all_cases[k]['bus_volt_mag_max_viol']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['bus_volt_mag_max_viol'] = max(self.summary2['base_bus_volt_mag_max_viol'], self.summary2['ctg_bus_volt_mag_max_viol'])

        self.summary2['base_bus_volt_mag_min_viol'] = self.summary_all_cases['BASECASE']['bus_volt_mag_min_viol']['val']
        self.summary2['ctg_bus_volt_mag_min_viol'] = np.amax([self.summary_all_cases[k]['bus_volt_mag_min_viol']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['bus_volt_mag_min_viol'] = max(self.summary2['base_bus_volt_mag_min_viol'], self.summary2['ctg_bus_volt_mag_min_viol'])

        self.summary2['base_load_t_max_viol'] = self.summary_all_cases['BASECASE']['load_t_max_viol']['val']
        self.summary2['ctg_load_t_max_viol'] = np.amax([self.summary_all_cases[k]['load_t_max_viol']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['load_t_max_viol'] = max(self.summary2['base_load_t_max_viol'], self.summary2['ctg_load_t_max_viol'])

        self.summary2['base_load_t_min_viol'] = self.summary_all_cases['BASECASE']['load_t_min_viol']['val']
        self.summary2['ctg_load_t_min_viol'] = np.amax([self.summary_all_cases[k]['load_t_min_viol']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['load_t_min_viol'] = max(self.summary2['base_load_t_min_viol'], self.summary2['ctg_load_t_min_viol'])

        self.summary2['base_load_ramp_up_max_viol'] = self.summary_all_cases['BASECASE']['load_ramp_up_max_viol']['val']
        self.summary2['ctg_load_ramp_up_max_viol'] = np.amax([self.summary_all_cases[k]['load_ramp_up_max_viol']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['load_ramp_up_max_viol'] = max(self.summary2['base_load_ramp_up_max_viol'], self.summary2['ctg_load_ramp_up_max_viol'])

        self.summary2['base_load_ramp_down_max_viol'] = self.summary_all_cases['BASECASE']['load_ramp_down_max_viol']['val']
        self.summary2['ctg_load_ramp_down_max_viol'] = np.amax([self.summary_all_cases[k]['load_ramp_down_max_viol']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['load_ramp_down_max_viol'] = max(self.summary2['base_load_ramp_down_max_viol'], self.summary2['ctg_load_ramp_down_max_viol'])

        self.summary2['base_gen_pow_real_max_viol'] = self.summary_all_cases['BASECASE']['gen_pow_real_max_viol']['val']
        self.summary2['ctg_gen_pow_real_max_viol'] = np.amax([self.summary_all_cases[k]['gen_pow_real_max_viol']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['gen_pow_real_max_viol'] = max(self.summary2['base_gen_pow_real_max_viol'], self.summary2['ctg_gen_pow_real_max_viol'])

        self.summary2['base_gen_pow_real_min_viol'] = self.summary_all_cases['BASECASE']['gen_pow_real_min_viol']['val']
        self.summary2['ctg_gen_pow_real_min_viol'] = np.amax([self.summary_all_cases[k]['gen_pow_real_min_viol']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['gen_pow_real_min_viol'] = max(self.summary2['base_gen_pow_real_min_viol'], self.summary2['ctg_gen_pow_real_min_viol'])

        self.summary2['base_gen_pow_imag_max_viol'] = self.summary_all_cases['BASECASE']['gen_pow_imag_max_viol']['val']
        self.summary2['ctg_gen_pow_imag_max_viol'] = np.amax([self.summary_all_cases[k]['gen_pow_imag_max_viol']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['gen_pow_imag_max_viol'] = max(self.summary2['base_gen_pow_imag_max_viol'], self.summary2['ctg_gen_pow_imag_max_viol'])

        self.summary2['base_gen_pow_imag_min_viol'] = self.summary_all_cases['BASECASE']['gen_pow_imag_min_viol']['val']
        self.summary2['ctg_gen_pow_imag_min_viol'] = np.amax([self.summary_all_cases[k]['gen_pow_imag_min_viol']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['gen_pow_imag_min_viol'] = max(self.summary2['base_gen_pow_imag_min_viol'], self.summary2['ctg_gen_pow_imag_min_viol'])

        self.summary2['base_gen_pow_real_0_if_x_0_viol'] = self.summary_all_cases['BASECASE']['gen_pow_real_0_if_x_0_viol']['val']
        self.summary2['ctg_gen_pow_real_0_if_x_0_viol'] = np.amax([self.summary_all_cases[k]['gen_pow_real_0_if_x_0_viol']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['gen_pow_real_0_if_x_0_viol'] = max(self.summary2['base_gen_pow_real_0_if_x_0_viol'], self.summary2['ctg_gen_pow_real_0_if_x_0_viol'])

        self.summary2['base_gen_pow_imag_0_if_x_0_viol'] = self.summary_all_cases['BASECASE']['gen_pow_imag_0_if_x_0_viol']['val']
        self.summary2['ctg_gen_pow_imag_0_if_x_0_viol'] = np.amax([self.summary_all_cases[k]['gen_pow_imag_0_if_x_0_viol']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['gen_pow_imag_0_if_x_0_viol'] = max(self.summary2['base_gen_pow_imag_0_if_x_0_viol'], self.summary2['ctg_gen_pow_imag_0_if_x_0_viol'])

        self.summary2['base_gen_ramp_up_max_viol'] = self.summary_all_cases['BASECASE']['gen_ramp_up_max_viol']['val']
        self.summary2['ctg_gen_ramp_up_max_viol'] = np.amax([self.summary_all_cases[k]['gen_ramp_up_max_viol']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['gen_ramp_up_max_viol'] = max(self.summary2['base_gen_ramp_up_max_viol'], self.summary2['ctg_gen_ramp_up_max_viol'])

        self.summary2['base_gen_ramp_down_max_viol'] = self.summary_all_cases['BASECASE']['gen_ramp_down_max_viol']['val']
        self.summary2['ctg_gen_ramp_down_max_viol'] = np.amax([self.summary_all_cases[k]['gen_ramp_down_max_viol']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['gen_ramp_down_max_viol'] = max(self.summary2['base_gen_ramp_down_max_viol'], self.summary2['ctg_gen_ramp_down_max_viol'])

        # maximum penalized bus power balance violations
        self.summary2['base_max_bus_pow_real_over'] = self.summary_all_cases['BASECASE']['max_bus_pow_real_over']['val']
        self.summary2['ctg_max_bus_pow_real_over'] = np.amax([self.summary_all_cases[k]['max_bus_pow_real_over']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['max_bus_pow_real_over'] = max(self.summary2['base_max_bus_pow_real_over'], self.summary2['ctg_max_bus_pow_real_over'])

        self.summary2['base_max_bus_pow_real_under'] = self.summary_all_cases['BASECASE']['max_bus_pow_real_under']['val']
        self.summary2['ctg_max_bus_pow_real_under'] = np.amax([self.summary_all_cases[k]['max_bus_pow_real_under']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['max_bus_pow_real_under'] = max(self.summary2['base_max_bus_pow_real_under'], self.summary2['ctg_max_bus_pow_real_under'])

        self.summary2['base_max_bus_pow_real'] = max(self.summary2['base_max_bus_pow_real_over'], self.summary2['base_max_bus_pow_real_under'])
        self.summary2['ctg_max_bus_pow_real'] = max(self.summary2['ctg_max_bus_pow_real_over'], self.summary2['ctg_max_bus_pow_real_under'])
        self.summary2['max_bus_pow_real'] = max(self.summary2['base_max_bus_pow_real'], self.summary2['ctg_max_bus_pow_real'])

        self.summary2['base_max_bus_pow_imag_over'] = self.summary_all_cases['BASECASE']['max_bus_pow_imag_over']['val']
        self.summary2['ctg_max_bus_pow_imag_over'] = np.amax([self.summary_all_cases[k]['max_bus_pow_imag_over']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['max_bus_pow_imag_over'] = max(self.summary2['base_max_bus_pow_imag_over'], self.summary2['ctg_max_bus_pow_imag_over'])

        self.summary2['base_max_bus_pow_imag_under'] = self.summary_all_cases['BASECASE']['max_bus_pow_imag_under']['val']
        self.summary2['ctg_max_bus_pow_imag_under'] = np.amax([self.summary_all_cases[k]['max_bus_pow_imag_under']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['max_bus_pow_imag_under'] = max(self.summary2['base_max_bus_pow_imag_under'], self.summary2['ctg_max_bus_pow_imag_under'])

        self.summary2['base_max_bus_pow_imag'] = max(self.summary2['base_max_bus_pow_imag_over'], self.summary2['base_max_bus_pow_imag_under'])
        self.summary2['ctg_max_bus_pow_imag'] = max(self.summary2['ctg_max_bus_pow_imag_over'], self.summary2['ctg_max_bus_pow_imag_under'])
        self.summary2['max_bus_pow_imag'] = max(self.summary2['base_max_bus_pow_imag'], self.summary2['ctg_max_bus_pow_imag'])

        self.summary2['base_max_bus_pow'] = max(self.summary2['base_max_bus_pow_real'], self.summary2['base_max_bus_pow_imag'])
        self.summary2['ctg_max_bus_pow'] = max(self.summary2['ctg_max_bus_pow_real'], self.summary2['ctg_max_bus_pow_imag'])
        self.summary2['max_bus_pow'] = max(self.summary2['base_max_bus_pow'], self.summary2['ctg_max_bus_pow'])

        # net penalized bus power balance violations
        self.summary2['base_sum_bus_pow_real_over'] = self.summary_all_cases['BASECASE']['sum_bus_pow_real_over']['val']
        self.summary2['ctg_sum_bus_pow_real_over'] = np.sum([self.summary_all_cases[k]['sum_bus_pow_real_over']['val'] for k in ctg_labels]) / self.num_ctg
        self.summary2['sum_bus_pow_real_over'] = self.summary2['base_sum_bus_pow_real_over'] + self.summary2['ctg_sum_bus_pow_real_over']

        self.summary2['base_sum_bus_pow_real_under'] = self.summary_all_cases['BASECASE']['sum_bus_pow_real_under']['val']
        self.summary2['ctg_sum_bus_pow_real_under'] = np.sum([self.summary_all_cases[k]['sum_bus_pow_real_under']['val'] for k in ctg_labels]) / self.num_ctg
        self.summary2['sum_bus_pow_real_under'] = self.summary2['base_sum_bus_pow_real_under'] + self.summary2['ctg_sum_bus_pow_real_under']

        self.summary2['base_sum_bus_pow_real_net'] = self.summary2['base_sum_bus_pow_real_over'] - self.summary2['base_sum_bus_pow_real_under']
        self.summary2['ctg_sum_bus_pow_real_net'] = self.summary2['ctg_sum_bus_pow_real_over'] - self.summary2['ctg_sum_bus_pow_real_under']
        self.summary2['sum_bus_pow_real_net'] = self.summary2['sum_bus_pow_real_over'] - self.summary2['sum_bus_pow_real_under']

        self.summary2['base_sum_bus_pow_imag_over'] = self.summary_all_cases['BASECASE']['sum_bus_pow_imag_over']['val']
        self.summary2['ctg_sum_bus_pow_imag_over'] = np.sum([self.summary_all_cases[k]['sum_bus_pow_imag_over']['val'] for k in ctg_labels]) / self.num_ctg
        self.summary2['sum_bus_pow_imag_over'] = self.summary2['base_sum_bus_pow_imag_over'] + self.summary2['ctg_sum_bus_pow_imag_over']

        self.summary2['base_sum_bus_pow_imag_under'] = self.summary_all_cases['BASECASE']['sum_bus_pow_imag_under']['val']
        self.summary2['ctg_sum_bus_pow_imag_under'] = np.sum([self.summary_all_cases[k]['sum_bus_pow_imag_under']['val'] for k in ctg_labels]) / self.num_ctg
        self.summary2['sum_bus_pow_imag_under'] = self.summary2['base_sum_bus_pow_imag_under'] + self.summary2['ctg_sum_bus_pow_imag_under']

        self.summary2['base_sum_bus_pow_imag_net'] = self.summary2['base_sum_bus_pow_imag_over'] - self.summary2['base_sum_bus_pow_imag_under']
        self.summary2['ctg_sum_bus_pow_imag_net'] = self.summary2['ctg_sum_bus_pow_imag_over'] - self.summary2['ctg_sum_bus_pow_imag_under']
        self.summary2['sum_bus_pow_imag_net'] = self.summary2['sum_bus_pow_imag_over'] - self.summary2['sum_bus_pow_imag_under']

        # delta to prior
        self.summary2['base_bus_volt_mag_delta_to_prior'] = self.summary_all_cases['BASECASE']['bus_volt_mag_delta_to_prior']['val']
        self.summary2['ctg_bus_volt_mag_delta_to_prior'] = np.amax([self.summary_all_cases[k]['bus_volt_mag_delta_to_prior']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['bus_volt_mag_delta_to_prior'] = max(self.summary2['base_bus_volt_mag_delta_to_prior'], self.summary2['bus_volt_mag_delta_to_prior'])

        self.summary2['base_bus_volt_ang_delta_to_prior'] = self.summary_all_cases['BASECASE']['bus_volt_ang_delta_to_prior']['val']
        self.summary2['ctg_bus_volt_ang_delta_to_prior'] = np.amax([self.summary_all_cases[k]['bus_volt_ang_delta_to_prior']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['bus_volt_ang_delta_to_prior'] = max(self.summary2['base_bus_volt_ang_delta_to_prior'], self.summary2['bus_volt_ang_delta_to_prior'])

        self.summary2['base_load_pow_real_delta_to_prior'] = self.summary_all_cases['BASECASE']['load_pow_real_delta_to_prior']['val']
        self.summary2['ctg_load_pow_real_delta_to_prior'] = np.amax([self.summary_all_cases[k]['load_pow_real_delta_to_prior']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['load_pow_real_delta_to_prior'] = max(self.summary2['base_load_pow_real_delta_to_prior'], self.summary2['load_pow_real_delta_to_prior'])

        self.summary2['base_gen_pow_real_delta_to_prior'] = self.summary_all_cases['BASECASE']['gen_pow_real_delta_to_prior']['val']
        self.summary2['ctg_gen_pow_real_delta_to_prior'] = np.amax([self.summary_all_cases[k]['gen_pow_real_delta_to_prior']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['gen_pow_real_delta_to_prior'] = max(self.summary2['base_gen_pow_real_delta_to_prior'], self.summary2['gen_pow_real_delta_to_prior'])

        self.summary2['base_gen_pow_imag_delta_to_prior'] = self.summary_all_cases['BASECASE']['gen_pow_imag_delta_to_prior']['val']
        self.summary2['ctg_gen_pow_imag_delta_to_prior'] = np.amax([self.summary_all_cases[k]['gen_pow_imag_delta_to_prior']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['gen_pow_imag_delta_to_prior'] = max(self.summary2['base_gen_pow_imag_delta_to_prior'], self.summary2['gen_pow_imag_delta_to_prior'])

        self.summary2['base_xfmr_tau_delta_to_prior'] = self.summary_all_cases['BASECASE']['xfmr_tau_delta_to_prior']['val']
        self.summary2['ctg_xfmr_tau_delta_to_prior'] = np.amax([self.summary_all_cases[k]['xfmr_tau_delta_to_prior']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['xfmr_tau_delta_to_prior'] = max(self.summary2['base_xfmr_tau_delta_to_prior'], self.summary2['xfmr_tau_delta_to_prior'])

        self.summary2['base_xfmr_phi_delta_to_prior'] = self.summary_all_cases['BASECASE']['xfmr_phi_delta_to_prior']['val']
        self.summary2['ctg_xfmr_phi_delta_to_prior'] = np.amax([self.summary_all_cases[k]['xfmr_phi_delta_to_prior']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['xfmr_phi_delta_to_prior'] = max(self.summary2['base_xfmr_phi_delta_to_prior'], self.summary2['xfmr_phi_delta_to_prior'])

        self.summary2['base_swsh_b_delta_to_prior'] = self.summary_all_cases['BASECASE']['swsh_b_delta_to_prior']['val']
        self.summary2['ctg_swsh_b_delta_to_prior'] = np.amax([self.summary_all_cases[k]['swsh_b_delta_to_prior']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['swsh_b_delta_to_prior'] = max(self.summary2['base_swsh_b_delta_to_prior'], self.summary2['swsh_b_delta_to_prior'])

        # maximum penalized line limit violations
        self.summary2['base_max_line_viol'] = self.summary_all_cases['BASECASE']['max_line_viol']['val']
        self.summary2['ctg_max_line_viol'] = np.amax([self.summary_all_cases[k]['max_line_viol']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['max_line_viol'] = max(self.summary2['base_max_line_viol'], self.summary2['ctg_max_line_viol'])

        # maximum penalized xfmr limit violations
        self.summary2['base_max_xfmr_viol'] = self.summary_all_cases['BASECASE']['max_xfmr_viol']['val']
        self.summary2['ctg_max_xfmr_viol'] = np.amax([self.summary_all_cases[k]['max_xfmr_viol']['val'] for k in ctg_labels], initial=0.0)
        self.summary2['max_xfmr_viol'] = max(self.summary2['base_max_xfmr_viol'], self.summary2['ctg_max_xfmr_viol'])

    @timeit
    def json_to_summary_all_cases(self, path):
        '''read the json case summary files and create summary_all_cases from them'''
        # note if we terminate early, we will not have all the contingency files

        json_files = [f for f in glob.glob(str('{}/eval_detail_*.json'.format(path))) if ('eval_detail_' in f) and ('json' in f)]
        json_contingency_files = [f for f in json_files if 'BASECASE' not in f]
        json_base_case_files = [f for f in json_files if 'BASECASE' in f]
        print_alert(
            'Expected 1 BASECASE json eval output file, Encountered {} BASECASE json eval output files, files found: {}'.format(
                len(json_base_case_files), json_base_case_files),
            check_passed=(len(json_base_case_files) == 1))

        json_base_case_file = json_base_case_files[0]
        contingency_labels = [Path(f).resolve().stem.replace("eval_detail_","") for f in json_contingency_files]
        num_contingencies = len(contingency_labels)
        
        # filter out anything not from this run
        contingency_labels = sorted(list(set(contingency_labels).intersection(set(self.ctg_label))))
        num_contingencies = len(contingency_labels)
        json_contingency_files = ["{}/eval_detail_{}.json".format(path, k) for k in contingency_labels]
        
        with open(json_base_case_file, 'r') as f:
            s = json.load(f)
        self.summary_all_cases['BASECASE'] = s
        for i in range(num_contingencies):
            json_contingency_file = json_contingency_files[i]
            contingency_label = contingency_labels[i]
            with open(json_contingency_file, 'r') as f:
                s = json.load(f)
            self.summary_all_cases[contingency_label] = s

    @timeit
    def write_detail_all_cases_csv(self, path):

        contingency_labels = sorted(list(set(self.summary_all_cases.keys()).difference(set(['BASECASE']))))
        num_contingencies = len(contingency_labels)

        with open('{}/eval_detail.csv'.format(path), mode='w') as detail_csv_file:
            detail_csv_writer = csv.writer(detail_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            s = self.summary_all_cases['BASECASE']
            detail_csv_writer.writerow(['case_label'] + flatten_summary(s)['keys']) # field names come from keys of base case summary
            detail_csv_writer.writerow(['BASECASE'] + flatten_summary(s)['values']) # values of base case summary
            for i in range(num_contingencies):
                contingency_label = contingency_labels[i]
                s = self.summary_all_cases[contingency_label]
                detail_csv_writer.writerow([contingency_label] + flatten_summary(s)['values']) # values of contingency summary

    @timeit
    def write_detail_all_cases_json(self, path):

        with open('{}/eval_detail.json'.format(path), mode='w') as outfile:
        #with open(f'{eval_out_path}/eval_out.json', 'w') as outfile:
            json.dump(self.summary_all_cases, outfile, indent=4, sort_keys=True)

    @timeit
    def json_to_csv(self, path):
        ''' '''
        # todo split this up
        # create_summary_all_cases(path)
        #     read the case summary json files into a dict structure
        # write_summary_all_cases(path)
        #     write the csv file as below
        #     write a json version of it
    
        json_files = [f for f in glob.glob(str('{}/eval_detail_*.json'.format(path))) if ('eval_detail_' in f) and ('json' in f)]
        json_contingency_files = [f for f in json_files if 'BASECASE' not in f]
        json_base_case_files = [f for f in json_files if 'BASECASE' in f]
        print_alert(
            'Expected 1 BASECASE json eval output file, Encountered {} BASECASE json eval output files, files found: {}'.format(
                len(json_base_case_files), json_base_case_files),
            check_passed=(len(json_base_case_files) == 1))

        json_base_case_file = json_base_case_files[0]
        contingency_labels = [Path(f).resolve().stem.replace("eval_detail_","") for f in json_contingency_files]
        num_contingencies = len(contingency_labels)
        
        with open('{}/eval_detail.csv'.format(eval_out_path), mode='w') as detail_csv_file:
            detail_csv_writer = csv.writer(detail_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            with open(json_base_case_file, 'r') as f:
                s = json.load(f)
            detail_csv_writer.writerow(['case_label'] + flatten_summary(s)['keys']) # field names come from keys of base case summary
            detail_csv_writer.writerow(['BASECASE'] + flatten_summary(s)['values']) # values of base case summary
            for i in range(num_contingencies):
                json_contingency_file = json_contingency_files[i]
                contingency_label = contingency_labels[i]
                with open(json_contingency_file, 'r') as f:
                    s = json.load(f)
                detail_csv_writer.writerow([contingency_label] + flatten_summary(s)['values']) # values of contingency summary

    def write_summary_json(self, path):

        with open('{}/eval_summary.json'.format(path), 'w') as outfile:
            json.dump(self.summary2,
                # {'infeas_cumulative': self.infeas_cumulative,
                #  'obj_cumulative': self.obj_cumulative,
                #  'infeas_all_cases': self.infeas_all_cases,
                #  'obj_all_cases': self.obj_all_cases,
                #  'base_gen_switch_up_actual': self.base_gen_switch_up_actual,
                #  'base_gen_switch_up_max': self.base_gen_switch_up_max,
                #  'base_gen_switch_down_actual': self.base_gen_switch_down_actual,
                #  'base_gen_switch_down_max': self.base_gen_switch_down_max,
                #  'base_line_switch_up_actual': self.base_line_switch_up_actual,
                #  'base_line_switch_up_max': self.base_line_switch_up_max,
                #  'base_line_switch_down_actual': self.base_line_switch_down_actual,
                #  'base_line_switch_down_max': self.base_line_switch_down_max,
                #  'base_xfmr_switch_up_actual': self.base_xfmr_switch_up_actual,
                #  'base_xfmr_switch_up_max': self.base_xfmr_switch_up_max,
                #  'base_xfmr_switch_down_actual': self.base_xfmr_switch_down_actual,
                #  'base_xfmr_switch_down_max': self.base_xfmr_switch_down_max,
                #  'ctg_gen_switch_up_actual': self.ctg_gen_switch_up_actual,
                #  'ctg_gen_switch_up_max': self.ctg_gen_switch_up_max,
                #  'ctg_gen_switch_down_actual': self.ctg_gen_switch_down_actual,
                #  'ctg_gen_switch_down_max': self.ctg_gen_switch_down_max,
                #  'ctg_line_switch_up_actual': self.ctg_line_switch_up_actual,
                #  'ctg_line_switch_up_max': self.ctg_line_switch_up_max,
                #  'ctg_line_switch_down_actual': self.ctg_line_switch_down_actual,
                #  'ctg_line_switch_down_max': self.ctg_line_switch_down_max,
                #  'ctg_xfmr_switch_up_actual': self.ctg_xfmr_switch_up_actual,
                #  'ctg_xfmr_switch_up_max': self.ctg_xfmr_switch_up_max,
                #  'ctg_xfmr_switch_down_actual': self.ctg_xfmr_switch_down_actual,
                #  'ctg_xfmr_switch_down_max': self.ctg_xfmr_switch_down_max,
                #  }, # todo a bunch of others here
                outfile, indent=4, sort_keys=False)

    def write_summary_csv(self, path):

        with open('{}/eval_summary.csv'.format(path), 'w') as outfile:
            writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for k in summary2_keys:
                writer.writerow([k, self.summary2[k]])

    def write_detail(self, path, case, detail_csv=False, detail_json=False):

        if detail_csv:
            if self.detail_csv_header_done:
                with open('{}/eval_detail.csv'.format(path), mode='a') as detail_csv_file:
                    detail_csv_writer = csv.writer(detail_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    detail_csv_writer.writerow([case] + flatten_summary(self.summary)['values'])
            else:
                self.detail_csv_header_done = True
                with open('{}/eval_detail.csv'.format(path), mode='w') as detail_csv_file:
                    detail_csv_writer = csv.writer(detail_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    detail_csv_writer.writerow(['case_label'] + flatten_summary(self.summary)['keys'])
                    detail_csv_writer.writerow([case] + flatten_summary(self.summary)['values'])
                
        if detail_json:
            with open('{}/eval_detail_{}.json'.format(path, case), 'w') as outfile:
                json.dump(self.summary, outfile, indent=4, sort_keys=True)

    def summarize(self, summary_key, values, keys=None, tol=None):

        '''adds to the evaluation single case summary
        a dictionary that serves as a summary of
        a numpy array of constraint violations'''
        # todo timing?

        infeas = False
        key = None
        val = -float('inf')
        if keys is None:
            # assume values is a scalar
            val = values
        elif values.size > 0:
            arg_max = np.argmax(values)
            key = keys[arg_max]
            val = values[arg_max]
        if tol is not None:
            if val > tol:
                infeas= True
        out = {
            'infeas': infeas,
            'key': key,
            'val': val}
        if check_summary_keys:
            assert summary_key in self.summary.keys(), 'unregistered summary key. new key: {}, existing keys: {}'.format(summary_key, list(self.summary.keys()))
            assert len(set(out.keys()).difference(set(summary_out_keys))) == 0, 'unregisered summary out key. new keys: {}, existing keys: {}'.format(list(out.keys), summary_out_keys) 
        self.summary[summary_key] = out
        if infeas:
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            function = calframe[1][3]
            self.summary['infeas'] = {'infeas': infeas, 'key': function, 'val': True}
            print_alert('fcn: {}, key: {}, val: {}'.format(function, key, val), check_passed=(not infeas), evaluation=self)

    @timeit
    def set_data_for_base(self):
        '''run this once before evaluating the base case'''

        self.delta = self.delta_base
        self.delta_r = self.delta_r_base
        self.bus_volt_mag_max = self.bus_volt_mag_max_base
        self.bus_volt_mag_min = self.bus_volt_mag_min_base
        self.gen_su_qual = self.gen_su_qual_base
        self.gen_sd_qual = self.gen_sd_qual_base
        self.line_curr_mag_max = self.line_curr_mag_max_base
        self.xfmr_pow_mag_max = self.xfmr_pow_mag_max_base
        self.load_ramp_up_max = self.load_ramp_up_max_base
        self.load_ramp_down_max = self.load_ramp_down_max_base
        self.gen_ramp_up_max = self.gen_ramp_up_max_base
        self.gen_ramp_down_max = self.gen_ramp_down_max_base
        self.line_cost_evaluator = self.line_cost_evaluator_base
        self.xfmr_cost_evaluator = self.xfmr_cost_evaluator_base

    @timeit
    def set_data_for_ctgs(self):
        '''run this once after evaluating the base case and before
        looping over contingencies'''

        self.delta = self.delta_ctg
        self.delta_r = self.delta_r_ctg
        self.bus_volt_mag_max = self.bus_volt_mag_max_ctg
        self.bus_volt_mag_min = self.bus_volt_mag_min_ctg
        self.gen_su_qual = self.gen_su_qual_ctg
        self.gen_sd_qual = self.gen_sd_qual_ctg
        self.line_curr_mag_max = self.line_curr_mag_max_ctg
        self.xfmr_pow_mag_max = self.xfmr_pow_mag_max_ctg
        self.load_ramp_up_max = self.load_ramp_up_max_ctg
        self.load_ramp_down_max = self.load_ramp_down_max_ctg
        self.gen_ramp_up_max = self.gen_ramp_up_max_ctg
        self.gen_ramp_down_max = self.gen_ramp_down_max_ctg
        self.line_cost_evaluator = self.line_cost_evaluator_ctg
        self.xfmr_cost_evaluator = self.xfmr_cost_evaluator_ctg

    @timeit
    def set_data_scalars(self, data):

        self.base_mva = data.raw.case_identification.sbase
        self.delta_base = self.data.sup.sup_jsonobj['systemparameters']['delta']
        self.delta_r_base = self.data.sup.sup_jsonobj['systemparameters']['deltar']
        self.delta_ctg = self.data.sup.sup_jsonobj['systemparameters']['deltactg']
        self.delta_r_ctg = self.data.sup.sup_jsonobj['systemparameters']['deltarctg']
        self.epsilon = hard_constr_tol
        self.num_swsh_block = 8
        self.num_xfmr_imp_corr_pts = 11

    @timeit
    def set_data_bus_params(self, data):

        buses = list(data.raw.buses.values())
        self.num_bus = len(buses)
        self.bus_i = [r.i for r in buses]
        self.bus_key = self.bus_i
        self.bus_map = {self.bus_i[i]:i for i in range(len(self.bus_i))}
        self.bus_volt_mag_0 = np.array([r.vm for r in buses])
        self.bus_volt_ang_0 = np.array([np.pi * r.va / 180.0 for r in buses])
        self.bus_volt_mag_max_base = np.array([r.nvhi for r in buses])
        self.bus_volt_mag_min_base = np.array([r.nvlo for r in buses])
        self.bus_volt_mag_max_ctg = np.array([r.evhi for r in buses])
        self.bus_volt_mag_min_ctg = np.array([r.evlo for r in buses])

    @timeit
    def set_data_load_params(self, data):

        loads = [r for r in data.raw.loads.values() if r.status == 1]
        self.num_load = len(loads)
        self.load_i = [r.i for r in loads]
        self.load_id = [r.id for r in loads]
        self.load_key = [(r.i, r.id) for r in loads]
        self.load_bus = [self.bus_map[self.load_i[i]] for i in range(self.num_load)]
        self.load_map = {(self.load_i[i], self.load_id[i]):i for i in range(self.num_load)}

        # real and reactive power consumption of the load in the
        # given operating point prior to the base case
        self.load_pow_real_0 = np.array([r.pl for r in loads]) / self.base_mva
        self.load_pow_imag_0 = np.array([r.ql for r in loads]) / self.base_mva

        self.load_t_min = np.array(
            [self.data.sup.loads[k]['tmin'] for k in self.load_key])
        self.load_t_max = np.array(
            [self.data.sup.loads[k]['tmax'] for k in self.load_key])
        self.load_ramp_up_max_base = np.array(
            [self.data.sup.loads[k]['prumax'] for k in self.load_key]) / self.base_mva
        self.load_ramp_down_max_base = np.array(
            [self.data.sup.loads[k]['prdmax'] for k in self.load_key]) / self.base_mva
        self.load_ramp_up_max_ctg = np.array(
            [self.data.sup.loads[k]['prumaxctg'] for k in self.load_key]) / self.base_mva
        self.load_ramp_down_max_ctg = np.array(
            [self.data.sup.loads[k]['prdmaxctg'] for k in self.load_key]) / self.base_mva

        self.bus_load_matrix = sp.csr_matrix(
            ([1.0 for i in range(self.num_load)],
             (self.load_bus, list(range(self.num_load)))),
            (self.num_bus, self.num_load))

    @timeit
    def set_data_fxsh_params(self, data):

        fxshs = [r for r in data.raw.fixed_shunts.values() if r.status == 1]
        self.num_fxsh = len(fxshs)
        self.fxsh_i = [r.i for r in fxshs]
        self.fxsh_id = [r.id for r in fxshs]
        self.fxsh_key = [(r.i, r.id) for r in fxshs]
        self.fxsh_bus = [self.bus_map[self.fxsh_i[i]] for i in range(self.num_fxsh)]
        self.fxsh_map = {(self.fxsh_i[i], self.fxsh_id[i]):i for i in range(self.num_fxsh)}
        self.fxsh_adm_real = np.array([r.gl / self.base_mva for r in fxshs])
        self.fxsh_adm_imag = np.array([r.bl / self.base_mva for r in fxshs])
        self.bus_fxsh_matrix = sp.csr_matrix(
            ([1.0 for i in range(self.num_fxsh)],
             (self.fxsh_bus, list(range(self.num_fxsh)))),
            (self.num_bus, self.num_fxsh))
        #self.bus_fxsh_adm_real = self.bus_fxsh_matrix.dot(self.fxsh_adm_real)
        #self.bus_fxsh_adm_imag = self.bus_fxsh_matrix.dot(self.fxsh_adm_imag)
        #print_info('fxsh_adm_imag:')
        #print_info(self.fxsh_adm_imag)

    @timeit
    def set_data_gen_params(self, data):
    
        gens = list(data.raw.generators.values()) # note here we take all generators regardless of status in RAW
        self.num_gen = len(gens)
        self.gen_i = [r.i for r in gens]
        self.gen_id = [r.id for r in gens]
        self.gen_key = [(r.i, r.id) for r in gens]
        self.gen_bus = [self.bus_map[self.gen_i[i]] for i in range(self.num_gen)]
        self.gen_map = {(self.gen_i[i], self.gen_id[i]):i for i in range(self.num_gen)}

        # commitment status in given operating point prior to the base case
        self.gen_xon_0 = np.array([r.stat for r in gens])

        self.gen_pow_imag_max = np.array([r.qt for r in gens]) / self.base_mva
        self.gen_pow_imag_min = np.array([r.qb for r in gens]) / self.base_mva
        self.gen_pow_real_max = np.array([r.pt for r in gens]) / self.base_mva
        self.gen_pow_real_min = np.array([r.pb for r in gens]) / self.base_mva
        self.gen_ramp_up_max_base = np.array([data.sup.generators[k]['prumax'] for k in self.gen_key]) / self.base_mva
        self.gen_ramp_down_max_base = np.array([data.sup.generators[k]['prdmax'] for k in self.gen_key]) / self.base_mva
        self.gen_ramp_up_max_ctg = np.array([data.sup.generators[k]['prumaxctg'] for k in self.gen_key]) / self.base_mva
        self.gen_ramp_down_max_ctg = np.array([data.sup.generators[k]['prdmaxctg'] for k in self.gen_key]) / self.base_mva
        self.gen_on_cost = np.array([data.sup.generators[k]['oncost'] for k in self.gen_key])
        self.gen_su_cost = np.array([data.sup.generators[k]['sucost'] for k in self.gen_key])
        self.gen_sd_cost = np.array([data.sup.generators[k]['sdcost'] for k in self.gen_key])
        self.gen_su_qual_base = np.array([data.sup.generators[k]['suqual'] for k in self.gen_key])
        self.gen_sd_qual_base = np.array([data.sup.generators[k]['sdqual'] for k in self.gen_key])
        self.gen_su_qual_ctg = np.array([data.sup.generators[k]['suqualctg'] for k in self.gen_key])
        self.gen_sd_qual_ctg = np.array([data.sup.generators[k]['sdqualctg'] for k in self.gen_key])

        # real power output in given operating point prior to the base case
        self.gen_pow_real_0 = np.array([r.pg for r in gens]) / self.base_mva
        self.gen_pow_imag_0 = np.array([r.qg for r in gens]) / self.base_mva

        self.gen_service_status = np.ones(shape=(self.num_gen,))
        self.bus_gen_matrix = sp.csr_matrix(
            ([1.0 for i in range(self.num_gen)],
             (self.gen_bus, list(range(self.num_gen)))),
            (self.num_bus, self.num_gen))

    @timeit
    def set_data_line_params(self, data):
        
        lines = list(data.raw.nontransformer_branches.values()) # note here we take all lines regardless of status in RAW
        self.num_line = len(lines)
        self.line_i = [r.i for r in lines]
        self.line_j = [r.j for r in lines]
        self.line_ckt = [r.ckt for r in lines]
        self.line_key = [(r.i, r.j, r.ckt) for r in lines]
        self.line_orig_bus = [self.bus_map[self.line_i[i]] for i in range(self.num_line)]
        self.line_dest_bus = [self.bus_map[self.line_j[i]] for i in range(self.num_line)]
        self.line_map = {(self.line_i[i], self.line_j[i], self.line_ckt[i]):i for i in range(self.num_line)}

        # closed-open status in operating point prior to base case
        self.line_xsw_0 = np.array([r.st for r in lines])
        
        self.line_adm_real = np.array([r.r / (r.r**2.0 + r.x**2.0) for r in lines])
        self.line_adm_imag = np.array([-r.x / (r.r**2.0 + r.x**2.0) for r in lines])
        self.line_adm_ch_imag = np.array([r.b for r in lines])
        self.line_adm_total_imag = self.line_adm_imag + 0.5 * self.line_adm_ch_imag
        self.line_curr_mag_max_base = np.array([r.ratea for r in lines]) / self.base_mva # todo - normalize by bus base kv?
        self.line_curr_mag_max_ctg = np.array([r.ratec for r in lines]) / self.base_mva # todo - normalize by bus base kv?
        self.bus_line_orig_matrix = sp.csr_matrix(
            ([1.0 for i in range(self.num_line)],
             (self.line_orig_bus, list(range(self.num_line)))),
            (self.num_bus, self.num_line))
        self.bus_line_dest_matrix = sp.csr_matrix(
            ([1.0 for i in range(self.num_line)],
             (self.line_dest_bus, list(range(self.num_line)))),
            (self.num_bus, self.num_line))
        self.line_sw_cost = np.array([data.sup.lines[k]['csw'] for k in self.line_key])
        if self.line_switching_allowed:
            self.line_sw_qual = np.array([data.sup.lines[k]['swqual'] for k in self.line_key])
        else:
            self.line_sw_qual = np.zeros(shape=(self.num_line))
        self.line_service_status = np.ones(shape=(self.num_line,))

    @timeit
    def set_data_xfmr_params(self, data):

        xfmrs = list(data.raw.transformers.values()) # note here we take all xfmrs, regardless of status in RAW
        self.num_xfmr = len(xfmrs)
        self.xfmr_i = [r.i for r in xfmrs]
        self.xfmr_j = [r.j for r in xfmrs]
        self.xfmr_ckt = [r.ckt for r in xfmrs]
        self.xfmr_key = [(r.i, r.j, r.ckt) for r in xfmrs] # do we really need the '0'? - not anymore
        self.xfmr_orig_bus = [self.bus_map[self.xfmr_i[i]] for i in range(self.num_xfmr)]
        self.xfmr_dest_bus = [self.bus_map[self.xfmr_j[i]] for i in range(self.num_xfmr)]
        self.xfmr_map = {(self.xfmr_i[i], self.xfmr_j[i], self.xfmr_ckt[i]):i for i in range(self.num_xfmr)}

        # closed-open status in operating point prior to base case
        self.xfmr_xsw_0 = np.array([r.stat for r in xfmrs])
        
        # series admittance (conductance and susceptance) from data
        # impedance correction divides these by impedance correction factor
        self.xfmr_adm_real_0 = np.array([r.r12 / (r.r12**2.0 + r.x12**2.0) for r in xfmrs])
        self.xfmr_adm_imag_0 = np.array([-r.x12 / (r.r12**2.0 + r.x12**2.0) for r in xfmrs])
        
        # magnetizing admittance
        self.xfmr_adm_mag_real = np.array([r.mag1 for r in xfmrs]) # todo normalize?
        self.xfmr_adm_mag_imag = np.array([r.mag2 for r in xfmrs]) # todo normalize?

        # tap magnitude and angle from data
        # control mode and xst variable may modify these into
        # xfmr_tap_mag and xfmr_tap_ang
        self.xfmr_tap_mag_0 = np.array([(r.windv1 / r.windv2) for r in xfmrs])
        self.xfmr_tap_ang_0 = np.array([r.ang1 * math.pi / 180.0 for r in xfmrs])

        # flow limits in base case and contingencies
        # select one or the other of these for
        # xfmr_pow_mag_max
        self.xfmr_pow_mag_max_base = np.array([r.rata1 for r in xfmrs]) / self.base_mva # todo check normalization
        self.xfmr_pow_mag_max_ctg = np.array([r.ratc1 for r in xfmrs]) / self.base_mva # todo check normalization
        
        # max num steps on each side of control range
        self.xfmr_xst_max = np.array([round(0.5 * (r.ntp1 - 1.0)) if (r.cod1 in [-3, -1, 1, 3]) else 0 for r in xfmrs], dtype=int)

        # midpoint of control range
        self.xfmr_tap_mag_mid = np.array(
            [(0.5 * (r.rma1 + r.rmi1)) if (r.cod1 in [-1, 1]) else (r.windv1 / r.windv2) for r in xfmrs])
        self.xfmr_tap_ang_mid = np.array(
            [(0.5 * (r.rma1 + r.rmi1) * math.pi / 180.0) if (r.cod1 in [-3, 3]) else (r.ang1 * math.pi / 180.0) for r in xfmrs])
        self.xfmr_mid = np.array(
            [(0.5 * (r.rma1 + r.rmi1)) if (r.cod1 in [-1, 1]) else
             (0.5 * (r.rma1 + r.rmi1) * math.pi / 180.0) if (r.cod1 in [-3, 3]) else
             0.0
             for r in xfmrs])

        # control step size
        self.xfmr_tap_mag_step_size = np.array(
            [((r.rma1 - r.rmi1) / (r.ntp1 - 1.0)) if (r.cod1 in [-1, 1]) else 0.0 for r in xfmrs])
        self.xfmr_tap_ang_step_size = np.array(
            [((r.rma1 - r.rmi1) * math.pi / 180.0 / (r.ntp1 - 1.0)) if (r.cod1 in [-3, 3]) else 0.0 for r in xfmrs])
        self.xfmr_step_size = np.array(
            [((r.rma1 - r.rmi1) / (r.ntp1 - 1.0)) if (r.cod1 in [-1, 1]) else
             ((r.rma1 - r.rmi1) * math.pi / 180.0 / (r.ntp1 - 1.0)) if (r.cod1 in [-3, 3]) else
             0.0
             for r in xfmrs])

        # to facilitate division by step size
        self.xfmr_xst_max[self.xfmr_step_size == 0.0] = 0
        self.xfmr_step_size[self.xfmr_step_size == 0.0] = 1.0

        # topology and switching
        self.bus_xfmr_orig_matrix = sp.csr_matrix(
            ([1.0 for i in range(self.num_xfmr)],
             (self.xfmr_orig_bus, list(range(self.num_xfmr)))),
            (self.num_bus, self.num_xfmr))
        self.bus_xfmr_dest_matrix = sp.csr_matrix(
            ([1.0 for i in range(self.num_xfmr)],
             (self.xfmr_dest_bus, list(range(self.num_xfmr)))),
            (self.num_bus, self.num_xfmr))
        self.xfmr_sw_cost = np.array([data.sup.transformers[k]['csw'] for k in self.xfmr_key])
        if self.xfmr_switching_allowed:
            self.xfmr_sw_qual = np.array([data.sup.transformers[k]['swqual'] for k in self.xfmr_key])
        else:
            self.xfmr_sw_qual = np.zeros(shape=(self.num_xfmr))
        self.xfmr_service_status = np.ones(shape=(self.num_xfmr,))

        # todo transformer impedance correction
        # todo which of these 2 options is best?
        self.xfmr_index_imp_corr = [ind for ind in range(self.num_xfmr) if (xfmrs[ind].tab1 > 0 and xfmrs[ind].cod1 in [-3, -1, 1, 3])]
        #self.xfmr_index_imp_corr = [ind for ind in range(self.num_xfmr) if xfmrs[ind].tab1 > 0]
        self.xfmr_index_fixed_tap_ratio_and_phase_shift = [ind for ind in range(self.num_xfmr) if xfmrs[ind].cod1 == 0]
        #self.xfmr_index_fixed_tap_ratio_and_phase_shift = [ind for ind in range(self.num_xfmr) if xfmrs[ind].cod1 not in [-3, -1, 1, 3]]
        self.xfmr_index_var_tap_ratio = [ind for ind in range(self.num_xfmr) if xfmrs[ind].cod1 in [-1, 1]]
        self.xfmr_index_var_phase_shift = [ind for ind in range(self.num_xfmr) if xfmrs[ind].cod1 in [-3, 3]]
        self.xfmr_index_imp_corr_var_tap_ratio = sorted(
            list(set(self.xfmr_index_imp_corr).intersection(
                    set(self.xfmr_index_var_tap_ratio))))
        self.xfmr_index_imp_corr_var_phase_shift = sorted(
            list(set(self.xfmr_index_imp_corr).intersection(
                    set(self.xfmr_index_var_phase_shift))))

        self.compute_xfmr_xst_0_from_tau_phi_0()

    def compute_xfmr_xst_0_from_tau_phi_0(self):

        #print('xfmr_xst_max:')
        #print(self.data.xfmr_xst_max)
        #
        self.xfmr_temp = np.zeros(shape=(self.num_xfmr,))
        self.xfmr_temp[self.xfmr_index_var_tap_ratio] = self.xfmr_tap_mag_0[self.xfmr_index_var_tap_ratio]
        self.xfmr_temp[self.xfmr_index_var_phase_shift] = self.xfmr_tap_ang_0[self.xfmr_index_var_phase_shift]
        np.subtract(self.xfmr_temp, self.xfmr_mid, out=self.xfmr_temp)
        np.divide(self.xfmr_temp, self.xfmr_step_size, out=self.xfmr_temp)
        np.round(self.xfmr_temp, out=self.xfmr_temp)
        np.minimum(self.xfmr_temp, self.xfmr_xst_max, out=self.xfmr_temp)
        np.negative(self.xfmr_temp, out=self.xfmr_temp)
        np.minimum(self.xfmr_temp, self.xfmr_xst_max, out=self.xfmr_temp)
        np.negative(self.xfmr_temp, out=self.xfmr_temp)
        self.xfmr_xst_0 = self.xfmr_temp # should be 0 on fixed tap, fixed phase
        #self.xfmr_xst[:] = 0
        #self.xfmr_xst[self.data.xfmr_index_var_tap_ratio] = self.xfmr_temp[self.data.xfmr_index_var_tap_ratio]
        #self.xfmr_xst[self.data.xfmr_index_var_phase_shift] = self.xfmr_temp[self.data.xfmr_index_var_phase_shift]

    @timeit
    def set_data_swsh_params(self, data):

        swshs = [r for r in data.raw.switched_shunts.values() if r.stat == 1]
        self.num_swsh = len(swshs)
        self.swsh_i = [r.i for r in swshs]
        self.swsh_key = self.swsh_i
        self.swsh_bus = [self.bus_map[self.swsh_i[i]] for i in range(self.num_swsh)]
        self.swsh_map = {self.swsh_i[i]:i for i in range(self.num_swsh)}
        self.swsh_b_0 = np.array([r.binit for r in swshs]) / self.base_mva # ?????)
        #print(self.swsh_b_0)
        self.swsh_block_adm_imag = np.array(
            [[r.b1, r.b2, r.b3, r.b4, r.b5, r.b6, r.b7, r.b8]
             for r in swshs]) / self.base_mva
        self.swsh_block_num_steps = np.array(
            [[r.n1, r.n2, r.n3, r.n4, r.n5, r.n6, r.n7, r.n8]
             for r in swshs])
        if self.num_swsh == 0:
            self.swsh_block_adm_imag = np.zeros(shape=(0, 8))
            self.swsh_block_num_steps = np.zeros(shape=(0, 8))
        #print(self.swsh_block_adm_imag)
        #print(self.swsh_block_num_steps)
        self.swsh_block_xst_0 = compute_swsh_block_xst(
            self.swsh_b_0, self.swsh_block_num_steps, self.swsh_block_adm_imag)
        #print(self.swsh_block_xst_0)
        self.swsh_num_blocks = np.array(
            [r.swsh_susc_count for r in swshs])
        self.bus_swsh_matrix = sp.csr_matrix(
            ([1.0 for i in range(self.num_swsh)],
             (self.swsh_bus, list(range(self.num_swsh)))),
            (self.num_bus, self.num_swsh))

    @timeit
    def set_data_gen_cost_params(self, data, convert_units=True):

        #print_info('pcblocks:')
        #print_info(self.data.sup.sup_jsonobj['pcblocks'])
        if convert_units:
            data.sup.convert_generator_cblock_units(self.base_mva)
            data.sup.convert_load_cblock_units(self.base_mva)
            data.sup.convert_pcblock_units(self.base_mva)
            data.sup.convert_qcblock_units(self.base_mva)
            data.sup.convert_scblock_units(self.base_mva)
        #print_info('pcblocks:')
        #print_info(self.data.sup.sup_jsonobj['pcblocks'])

    @timeit
    def set_data_ctg_params(self, data):
        # contingency records
        # this section was pretty long (40 s) - much reduced now, < 1 s (see below)
        # todo what is needed? can it be more efficient?

        ctgs = data.con.contingencies.values()
        self.num_ctg = len(ctgs)
        self.ctg_label = [r.label for r in ctgs]
        self.ctg_map = dict(zip(self.ctg_label, range(self.num_ctg)))
        line_keys = set(self.line_key)
        xfmr_keys = set(self.xfmr_key)
        ctg_gen_keys_out = {
            r.label:set([(e.i, e.id) for e in r.generator_out_events])
            for r in ctgs}
        ctg_branch_keys_out = {
            r.label:set([(e.i, e.j, e.ckt) for e in r.branch_out_events])
            for r in ctgs}
        ctg_branch_xfmr_keys_out = {
            r.label:set([(e.i, e.j, e.ckt) for e in r.branch_out_events])
            for r in ctgs}

        ctg_line_keys_out = {k:(v & line_keys) for k,v in ctg_branch_keys_out.items()}
        ctg_xfmr_keys_out = {k:(v & xfmr_keys) for k,v in ctg_branch_xfmr_keys_out.items()}

        self.ctg_gens_out = [
            [self.gen_map[k] for k in ctg_gen_keys_out[self.ctg_label[i]]]
            for i in range(self.num_ctg)]
        self.ctg_lines_out = [
            [self.line_map[k] for k in ctg_line_keys_out[self.ctg_label[i]]]
            for i in range(self.num_ctg)]
        self.ctg_xfmrs_out = [
            [self.xfmr_map[k] for k in ctg_xfmr_keys_out[self.ctg_label[i]]]
            for i in range(self.num_ctg)]

        self.ctg_num_gens_out = [len(self.ctg_gens_out[i]) for i in range(self.num_ctg)]
        self.ctg_num_lines_out = [len(self.ctg_lines_out[i]) for i in range(self.num_ctg)]
        self.ctg_num_xfmrs_out = [len(self.ctg_xfmrs_out[i]) for i in range(self.num_ctg)]

    @timeit
    def set_data(self, data, convert_units=True):
        ''' set values from the data object
        convert to per unit (p.u.) convention'''

        self.data = data
        self.set_data_scalars(data)
        self.set_data_bus_params(data)
        self.set_data_load_params(data)
        self.set_data_fxsh_params(data)
        self.set_data_gen_params(data)
        self.set_data_line_params(data)
        self.set_data_xfmr_params(data)
        self.set_data_swsh_params(data)
        self.set_data_gen_cost_params(data, convert_units=convert_units)
        self.set_cost_evaluators()
        self.set_data_ctg_params(data)

    @timeit
    def set_cost_evaluators(self):
        '''add cost evaluators and set them up with cost function data'''

        # construct cost evaluators
        self.bus_real_cost_evaluator = CostEvaluator('pmax', 'c')
        self.bus_imag_cost_evaluator = CostEvaluator('qmax', 'c')
        self.load_cost_evaluator = CostEvaluator('pmax', 'c')
        self.gen_cost_evaluator = CostEvaluator('pmax', 'c')
        self.line_cost_evaluator_base = CostEvaluator('smax', 'c')
        self.line_cost_evaluator_ctg = CostEvaluator('smax', 'c')
        self.xfmr_cost_evaluator_base = CostEvaluator('smax', 'c')
        self.xfmr_cost_evaluator_ctg = CostEvaluator('smax', 'c')

        # add cost function data
        self.bus_real_cost_evaluator.setup(self.num_bus, [self.data.sup.sup_jsonobj['pcblocks']])
        self.bus_imag_cost_evaluator.setup(self.num_bus, [self.data.sup.sup_jsonobj['qcblocks']])
        self.load_cost_evaluator.setup(self.num_load, [self.data.sup.loads[k]['cblocks'] for k in self.load_key])
        self.gen_cost_evaluator.setup(self.num_gen, [self.data.sup.generators[k]['cblocks'] for k in self.gen_key])
        self.line_cost_evaluator_base.setup(
            self.num_line,
            [[{'smax': (b['tmax'] * self.line_curr_mag_max_base[i]),
               'c': b['c']}
              for b in self.data.sup.sup_jsonobj['scblocks']]
             for i in range(self.num_line)])
        self.line_cost_evaluator_ctg.setup(
            self.num_line,
            [[{'smax': (b['tmax'] * self.line_curr_mag_max_ctg[i]),
               'c': b['c']}
              for b in self.data.sup.sup_jsonobj['scblocks']]
             for i in range(self.num_line)])
        self.xfmr_cost_evaluator_base.setup(
            self.num_xfmr,
            [[{'smax': (b['tmax'] * self.xfmr_pow_mag_max_base[i]),
               'c': b['c']}
              for b in self.data.sup.sup_jsonobj['scblocks']]
             for i in range(self.num_xfmr)])
        self.xfmr_cost_evaluator_ctg.setup(
            self.num_xfmr,
            [[{'smax': (b['tmax'] * self.xfmr_pow_mag_max_ctg[i]),
               'c': b['c']}
              for b in self.data.sup.sup_jsonobj['scblocks']]
             for i in range(self.num_xfmr)])

        # extra stuff to translate between load benefit and cost
        self.load_cost_evaluator.compute_f_z_at_x_max()

    @timeit
    def set_sol_initialize(self):
        '''Call this once to allocate the numpy arrays needed to store the solution and intermediate calculations'''

        # buses
        self.bus_temp = np.zeros(shape=(self.num_bus,))
        self.bus_volt_mag = np.zeros(shape=(self.num_bus,))
        self.bus_volt_ang = np.zeros(shape=(self.num_bus,))
        self.bus_volt_mag_prior = np.zeros(shape=(self.num_bus,))
        self.bus_volt_ang_prior = np.zeros(shape=(self.num_bus,))
        self.bus_pow_real_imbalance = np.zeros(shape=(self.num_bus,))
        self.bus_pow_imag_imbalance = np.zeros(shape=(self.num_bus,))
        self.bus_cost = np.zeros(shape=(self.num_bus,))

        # loads
        self.load_temp = np.zeros(shape=(self.num_load,))
        self.load_t = np.zeros(shape=(self.num_load,))
        self.load_pow_real_prior = np.zeros(shape=(self.num_load,))
        self.load_pow_real = np.zeros(shape=(self.num_load,))
        self.load_pow_imag = np.zeros(shape=(self.num_load,))
        self.load_benefit = np.zeros(shape=(self.num_load,))

        # fixed shunts
        self.fxsh_temp = np.zeros(shape=(self.num_fxsh,))
        self.fxsh_pow_real = np.zeros(shape=(self.num_fxsh,))
        self.fxsh_pow_imag = np.zeros(shape=(self.num_fxsh,))

        # generators
        self.gen_temp = np.zeros(shape=(self.num_gen,))
        self.gen_temp2 = np.zeros(shape=(self.num_gen,))
        self.gen_xon_prior = np.zeros(shape=(self.num_gen,))
        self.gen_xon = np.zeros(shape=(self.num_gen,))
        self.gen_xsu_prior = np.zeros(shape=(self.num_gen,))
        self.gen_xsu = np.zeros(shape=(self.num_gen,))
        self.gen_xsd_prior = np.zeros(shape=(self.num_gen,))
        self.gen_xsd = np.zeros(shape=(self.num_gen,))
        self.gen_pow_real_prior = np.zeros(shape=(self.num_gen,))
        self.gen_pow_imag_prior = np.zeros(shape=(self.num_gen,))
        self.gen_pow_real = np.zeros(shape=(self.num_gen,))
        self.gen_pow_imag = np.zeros(shape=(self.num_gen,))
        self.gen_cost = np.zeros(shape=(self.num_gen,))

        # lines
        self.line_temp = np.zeros(shape=(self.num_line,))
        self.line_xsu = np.zeros(shape=(self.num_line,))
        self.line_xsd = np.zeros(shape=(self.num_line,))
        self.line_xsw_prior = np.zeros(shape=(self.num_line,))
        self.line_xsw = np.zeros(shape=(self.num_line,))
        self.line_orig_volt_mag = np.zeros(shape=(self.num_line,))
        self.line_dest_volt_mag = np.zeros(shape=(self.num_line,))
        self.line_orig_volt_ang = np.zeros(shape=(self.num_line,))
        self.line_dest_volt_ang = np.zeros(shape=(self.num_line,))
        self.line_orig_pow_real = np.zeros(shape=(self.num_line,))
        self.line_dest_pow_real = np.zeros(shape=(self.num_line,))
        self.line_orig_pow_imag = np.zeros(shape=(self.num_line,))
        self.line_dest_pow_imag = np.zeros(shape=(self.num_line,))
        self.line_pow_mag_max_viol = np.zeros(shape=(self.num_line,))
        self.line_cost = np.zeros(shape=(self.num_line,))

        # transformers
        self.xfmr_temp = np.zeros(shape=(self.num_xfmr,))
        self.xfmr_xsu = np.zeros(shape=(self.num_xfmr,))
        self.xfmr_xsd = np.zeros(shape=(self.num_xfmr,))
        self.xfmr_xsw_prior = np.zeros(shape=(self.num_xfmr,))
        self.xfmr_xsw = np.zeros(shape=(self.num_xfmr,))
        self.xfmr_xst = np.zeros(shape=(self.num_xfmr,))
        self.xfmr_tap_mag = np.ones(shape=(self.num_xfmr,))
        self.xfmr_tap_ang = np.zeros(shape=(self.num_xfmr,))
        self.xfmr_tap_mag_prior = np.ones(shape=(self.num_xfmr,))
        self.xfmr_tap_ang_prior = np.zeros(shape=(self.num_xfmr,))
        self.xfmr_imp_corr = np.ones(shape=(self.num_xfmr,))
        self.xfmr_adm_real = np.zeros(shape=(self.num_xfmr,))
        self.xfmr_adm_imag = np.zeros(shape=(self.num_xfmr,))
        self.xfmr_orig_volt_mag = np.zeros(shape=(self.num_xfmr,))
        self.xfmr_dest_volt_mag = np.zeros(shape=(self.num_xfmr,))
        self.xfmr_orig_volt_ang = np.zeros(shape=(self.num_xfmr,))
        self.xfmr_dest_volt_ang = np.zeros(shape=(self.num_xfmr,))
        self.xfmr_orig_pow_real = np.zeros(shape=(self.num_xfmr,))
        self.xfmr_dest_pow_real = np.zeros(shape=(self.num_xfmr,))
        self.xfmr_orig_pow_imag = np.zeros(shape=(self.num_xfmr,))
        self.xfmr_dest_pow_imag = np.zeros(shape=(self.num_xfmr,))
        self.xfmr_pow_mag_max_viol = np.zeros(shape=(self.num_xfmr,))
        self.xfmr_cost = np.zeros(shape=(self.num_xfmr,))

        # switched shunts
        self.swsh_temp = np.zeros(shape=(self.num_swsh,))
        self.swsh_block_temp = np.zeros(shape=(self.num_swsh, self.num_swsh_block))
        self.swsh_block_xst = np.zeros(shape=(self.num_swsh, self.num_swsh_block))
        self.swsh_adm_imag = np.zeros(shape=(self.num_swsh,))
        self.swsh_adm_imag_prior = np.zeros(shape=(self.num_swsh,))
        self.swsh_pow_imag = np.zeros(shape=(self.num_swsh,))

    @timeit
    def set_prior_from_data_for_base(self):

        self.bus_volt_ang_prior[:] = self.bus_volt_ang_0
        self.bus_volt_mag_prior[:] = self.bus_volt_mag_0
        self.load_pow_real_prior[:] = self.load_pow_real_0
        self.gen_pow_real_prior[:] = self.gen_pow_real_0
        self.gen_pow_imag_prior[:] = self.gen_pow_imag_0
        self.gen_xon_prior[:] = self.gen_xon_0
        self.gen_xsu_prior[:] = 0.0
        self.gen_xsd_prior[:] = 0.0
        self.line_xsw_prior[:] = self.line_xsw_0
        self.xfmr_xsw_prior[:] = self.xfmr_xsw_0
        self.xfmr_tap_mag_prior[:] = self.xfmr_tap_mag_0
        self.xfmr_tap_ang_prior[:] = self.xfmr_tap_ang_0
        self.swsh_adm_imag_prior[:] = self.swsh_b_0

    @timeit
    def set_prior_from_base_for_ctgs(self):
        '''Call this one time after evaluating the base case and before looping over contingencies.
        Do not call before each contigency.
        '''

        self.bus_volt_ang_prior[:] = self.bus_volt_ang
        self.bus_volt_mag_prior[:] = self.bus_volt_mag
        self.load_pow_real_prior[:] = self.load_pow_real
        self.gen_pow_real_prior[:] = self.gen_pow_real
        self.gen_pow_imag_prior[:] = self.gen_pow_imag
        self.gen_xon_prior[:] = self.gen_xon
        self.gen_xsu_prior[:] = self.gen_xsu
        self.gen_xsd_prior[:] = self.gen_xsd
        self.line_xsw_prior[:] = self.line_xsw
        self.xfmr_xsw_prior[:] = self.xfmr_xsw
        self.xfmr_tap_mag_prior[:] = self.xfmr_tap_mag
        self.xfmr_tap_ang_prior[:] = self.xfmr_tap_ang
        self.swsh_adm_imag_prior[:] = self.swsh_adm_imag

    @timeit
    def set_sol(self, sol):
        #C2 A1 S2
        #Read solution input variables xon gk, xsw ek , xsw fk, xst hak, xst fk, vik, θik, pgk, qgk, tjk from solution ﬁles. 

        self.bus_volt_mag[:] = sol.bus_volt_mag
        self.bus_volt_ang[:] = sol.bus_volt_ang
        self.load_t[:] = sol.load_t
        self.gen_pow_real[:] = sol.gen_pow_real
        self.gen_pow_imag[:] = sol.gen_pow_imag
        self.gen_xon[:] = sol.gen_xon
        # todo int?
        #print_info('gen xon:')
        #print_info(sol.gen_xon)
        self.line_xsw[:] = sol.line_xsw
        self.xfmr_xsw[:] = sol.xfmr_xsw
        self.xfmr_xst[:] = sol.xfmr_xst
        self.swsh_block_xst[:] = sol.swsh_xst

    @timeit
    def set_sol_from_data(self): #????????

        self.bus_volt_mag[:] = self.bus_volt_mag_0
        self.bus_volt_ang[:] = self.bus_volt_ang_0
        self.load_t[:] = 1.0
        self.gen_pow_real[:] = self.gen_pow_real_0
        self.gen_pow_imag[:] = self.gen_pow_imag_0
        self.gen_xon[:] = self.gen_xon_0
        self.line_xsw[:] = self.line_xsw_0
        self.xfmr_xsw[:] = self.xfmr_xsw_0
        self.xfmr_xst[:] = self.xfmr_xst_0
        self.swsh_block_xst[:] = self.swsh_block_xst_0

        # for debugging
        # print('bus volt mag: {}'.format(self.bus_volt_mag))
        # print('bus volt ang: {}'.format(self.bus_volt_ang))
        # print('load t: {}'.format(self.load_t))
        # print('gen pow real: {}'.format(self.gen_pow_real))
        # print('gen pow imag: {}'.format(self.gen_pow_imag))
        # print('gen xon: {}'.format(self.gen_xon))
        # print('line xsw: {}'.format(self.line_xsw))
        # print('xfmr xsw: {}'.format(self.xfmr_xsw))
        # print('xfmr xst: {}'.format(self.xfmr_xst))
        # print('swsh block xst: {}'.format(self.swsh_block_xst))

    @timeit
    def round_sol(self):
        #The commitment variables xon gk, the start up indicators xsu gk, and the shut down indicators xsd gk, are binary variables
        # C2 A1 S4  #77
        #x_gk^on
        ##Round integer input variables C1 A1 S3

        # todo: do we use this?
        # should we store these as ints or floats?
        np.around(self.gen_xon, out=self.gen_xon)
        np.around(self.line_xsw, out=self.line_xsw)
        np.around(self.xfmr_xsw, out=self.xfmr_xsw)
        np.around(self.xfmr_xst, out=self.xfmr_xst)
        np.around(self.swsh_block_xst, out=self.swsh_block_xst)
        # todo int?
        #print_info('gen xon:')
        #print_info(self.gen_xon)

    # todo remove, but note the checking routines/coding
    def set_solution1(self, solution1):
        # set values from the solution objects
        #convert to per unit (p.u.) convention

        self.solution1 = solution1

        # insert columns from solution bus data frame
        #sol_bus_vm = solution1.bus_df.vm.values
        #sol_bus_va = solution1.bus_df.va.values
        #self.bus_volt_mag

        #CHALLENGE2 CHECK BOUNDS
        self.check_swsh_bounds(solution1)

        start_time = time.time()
        sol_bus_i = solution1.bus_df.i.values
        sol_gen_i = solution1.gen_df.i.values
        sol_gen_id = map(clean_string, list(solution1.gen_df.id.values))

        # which is faster? do the same for gens
        #sol_bus_map = {sol_bus_i[i]:i for i in range(self.num_bus)}
        sol_bus_map = dict(zip(sol_bus_i, list(range(self.num_bus))))

        #sol_gen_map = {(sol_gen_i[i], sol_gen_id[i]):i for i in range(self.num_gen)}
        sol_gen_key = zip(sol_gen_i, sol_gen_id)
        self.base_gen_map = dict(zip(sol_gen_key, list(range(self.num_gen))))
        #print([sol_gen_i, sol_gen_id, sol_gen_key, sol_gen_map, self.gen_key])
        # up through here is fast enough ~ 0.001 s

        #Take all the generators from solution
        #Number them 0...n
        #Map sol keys against 0..n
        #Consider order of generators in data as the basis
        #Get indices of solution generators in the order of data generators
        #Get array of observations based on this list

        
        # which is faster?
        #bus_permutation = [sol_bus_map[self.bus_i[r]] for r in range(self.num_bus)] # this line is slow ~0.015s. is there a faster python-y way to do it?
        bus_permutation = [sol_bus_map[k] for k in self.bus_i]
        gen_permutation = [self.base_gen_map[k] for k in self.gen_key]
        # need it to handle arbitrary bus order in solution files
        # is there a faster way to do it?
        # maybe arrange all the bus-indexed vectors in a matrix - not much time left to save though
        # multiplication by a permutation matrix?
        self.bus_volt_mag = solution1.bus_df.vm.values[bus_permutation]                         #CHALLENGE2 algorithm1 STEP 2 V
        self.bus_volt_ang = solution1.bus_df.va.values[bus_permutation] * (math.pi / 180.0)     #CHALLENGE2 ALGORITHM1 STEP 2 THETA
        

        #SCALING RULES
        # ALL POWER VALUES FROM RAW NEED TO BE SCALED
        # ALL BLOCKS FROM JSON NEED TO BE ADJUSTED
        # NOTHING FROM SOLUTION SHOULD BE SCALED

        self.gen_pow_real = solution1.gen_df.p.values[gen_permutation]         #CHALLENGE2 ALGORITHM1 STEP 2 P
        self.gen_pow_imag = solution1.gen_df.q.values[gen_permutation]         #CHALLENGE2 ALGORITHM1 STEP 2 Q
        #self.gen_bus_volt_mag = self.bus_volt_mag[self.gen_bus]
        # up through here is fast enough ~ 0.02 s (only 0.005 s from previous point)


        #CHALLENGE2

        self.gen_base_x = np.around( solution1.gen_df.x.values[gen_permutation] )        #commitment status

        check0 = (x == 0 or x == 1. for x in self.gen_base_x)
        check = all(check0)
        f = np.argmax( check0 == False)
        eg = self.gen_key [  f    ] if f > 0 else ""
        print_alert("gen commitment status indicator must be zero or one. {}".format(eg), check_passed=check)

        self.gen_base_commitment_changes = self.gen_base_x - self.gen_init_x

        #startup from 0 to 1, sd - otherwise
        self.base_su =  np.array( [ 1 if p[0]==0 and p[1]==1  else 0    for p in zip(self.gen_init_x,self.gen_base_x)])
        self.base_sd =  np.array( [ 1 if p[0]==1 and p[1]==0  else 0     for p in zip(self.gen_init_x,self.gen_base_x)])

        base_su_cost =  np.array(  [  self.data.sup.generators[k]['sucost']  for k in self.gen_key])
        base_sd_cost =  np.array(  [  self.data.sup.generators[k]['sdcost']  for k in self.gen_key])
        base_on_cost =  np.array(  [  self.data.sup.generators[k]['oncost']  for k in self.gen_key])

        self.gen_base_su_cost = base_su_cost * self.base_su 
        self.gen_base_sd_cost = base_sd_cost * self.base_sd
        self.gen_base_on_cost = base_on_cost * self.gen_base_x
 
        #78
        lhs = self.gen_base_commitment_changes
        rhs = self.base_su - self.base_sd
        check0 = ( lhs == rhs)
        check = all(check0)
        f = np.argmax( check0 == False)
        eg = self.gen_key [  f    ] if f > 0 else ""        
        print_alert(f'The start up and shut down indicators are inconsistent. {eg}', check_passed=check)

        #80
        check0 = [s <= 1 for s in  self.base_su + self.base_sd ]
        check = all (check0 )
        f = np.argmax( check0 == False)
        eg = self.gen_key [  f    ] if f > 0 else ""                
        print_alert(f'no generator may simultaneously start up and shut down. {eg}', check_passed = check)


        #Only generators in Gsu may start up    
        #C2 A1 S6 #85
        sup_gen_suqual = np.array( [ self.data.sup.generators[k]['suqual']   for k in self.gen_key ])
        check0 = [(p[0] == 1 and p[1] == 1) or p[0] == 0      for p in zip(self.base_su, sup_gen_suqual)  ]
        check = all (  check0     )
        f = np.argmax( check0 == False)
        eg = self.gen_key [  f    ] if f > 0 else ""                        
        print_alert(f'Only qualified generators may start up. {eg}', check_passed=check)

        #Only generators in Gsd may shutdown
        #C2 A1 S6 #86
        sup_gen_sdqual = np.array( [ self.data.sup.generators[k]['sdqual']   for k in self.gen_key ])
        check0 =[ (p[0] == 1 and p[1] == 1) or p[0] == 0      for p in zip(self.base_sd, sup_gen_sdqual)  ]
        check = all (  check0     )
        f = np.argmax( check0 == False)
        eg = self.gen_key [  f    ] if f > 0 else ""
        print_alert('Only qualified generators may shutdown.{}'.format(eg),check_passed=check)   


        # Energy Bounds
        # The real power output pgk of a committed generator in case k is subject to energy bounds, while decommitted generators have 0 real power output:
        # C2 A1 S9 #81
        gen_raw_max_pow_real = np.array( [ self.data.raw.generators[k].pt   for k in self.gen_key ]) / self.base_mva
        gen_raw_min_pow_real = np.array( [ self.data.raw.generators[k].pb   for k in self.gen_key ]) / self.base_mva
        check0_lb = np.array([ p[0] <= p[1] for p in zip(  (gen_raw_min_pow_real * self.gen_base_x), self.gen_pow_real)])
        check0_ub = np.array( [ p[0] <= p[1] for p in zip(   self.gen_pow_real, (gen_raw_max_pow_real * self.gen_base_x))])
        check0 = check0_lb & check0_ub 
        check = all (check0)
        f = np.argmax( check0 == False)
        eg = self.gen_key [  f    ] if f > 0 else ""
        print_alert('The real power output pgk of a committed generator in case k is subject to energy bounds, while decommitted generators have 0 real power output. {}'.format(eg), check_passed=check)

        # print_alert(gen_raw_min_pow_real)
        # print_alert(gen_raw_max_pow_real)
        # print_alert(self.gen_pow_real, raise_exception=False)
        # print_alert(check0_lb)
        # print_alert(check0_ub)

        # Energy Bounds
        # The reactive power output qgk of a committed generator in case k is subject to energy bounds, while decommitted generators have 0 reactive power output:
        # C2 A1 S9 #82
        gen_raw_max_pow_imag = np.array( [ self.data.raw.generators[k].qt   for k in self.gen_key ]) / self.base_mva
        gen_raw_min_pow_imag = np.array( [ self.data.raw.generators[k].qb   for k in self.gen_key ]) / self.base_mva
        check0_lb = np.array( [p[0] <= p[1] for p in zip( gen_raw_min_pow_imag * self.gen_base_x, self.gen_pow_imag)])
        check0_ub = np.array( [p[0] <= p[1] for p in zip( self.gen_pow_imag,  (gen_raw_max_pow_imag * self.gen_base_x))])
        check0 = ( check0_lb & check0_ub  )
        check = all (check0)
        f = np.argmax( check0 == False)
        eg = self.gen_key [  f    ] if f > 0 else ""
        print_alert('The reactive power output qgk of a committed generator in case k is subject to energy bounds, while decommitted generators have 0 reactive power output.{}'.format(eg),  check_passed=check)

        # Each generator g in a case k is subject to ramp rate constraints linking the real power output pgk to the prior real power output: 
        # C2 A1 S9 #83
        deltar = self.data.sup.sup_jsonobj['systemparameters']['deltar']
        prumax = np.array( [  self.data.sup.generators[k]['prumax'] for k in self.gen_key ]) / self.base_mva
        prdmax = np.array( [  self.data.sup.generators[k]['prdmax'] for k in self.gen_key ]) / self.base_mva
        prior_real_pow = np.array( [ self.data.raw.generators[k].pg   for k in self.gen_key ])/self.base_mva      #operating point
        delta_real_pow = self.gen_pow_real - prior_real_pow
        check0_lb = np.array( [ p[0] <= p[1] for p in zip( -(prdmax * deltar ),  delta_real_pow) ])
        check0_ub = np.array( [  p[0] <= p[1] for p in zip( delta_real_pow, (prumax * deltar )) ])
        check0 = ( check0_lb  & check0_ub  )
        check = all (check0)
        f = np.argmax( check0 == False)
        eg = self.gen_key [  f    ] if f > 0 else ""        
        print_alert(f'Each generator g in a case k is subject to ramp rate constraints linking the real power output pgk to the prior real power output. {eg}', check_passed=check)

        #C2 A1 
        # CHECK Pgk > Pmax  ()  #already done in #81

        #x_ek_sw
        sol_line_iorig = solution1.line_df.iorig.values
        sol_line_idest = solution1.line_df.idest.values
        sol_line_id = map(clean_string, list(solution1.line_df.id.values))
        sol_line_key = zip(sol_line_iorig, sol_line_idest, sol_line_id)
        self.base_line_map = dict(zip(sol_line_key, list(range(self.num_line))))
        #line_permutation = [self.base_line_map[k] for k in self.data.raw.nontransformer_branches.keys()]
        line_permutation = [self.base_line_map[k] for k in self.line_key]

        #Round integer input variables  
        self.line_base_x = np.around( solution1.line_df.x.values[line_permutation] )        #line open/closed

        # Check domains of integer input variables,     #44
        # Each line may be closed or open, and the closed-open status is indicated by a binary variable
        # C2 A1 S4 #44
        check0 = [x == 0 or x == 1. for x in self.line_base_x]
        line_x_binary_only = all(check0)
        f = np.argmax( check0 == False)
        eg = self.line_key [  f    ] if f > 0 else ""                
        print_alert(f"Line Closed-open status indicator must be zero or one.{eg}", check_passed=line_x_binary_only)

        # Check domains of integer input variables  #45
        #The closed-open status for lines that are not qualiﬁed to switch must remain to the value in the given operating point prior to the base case
        # C2 A1 S4 #45
        data_line_status = np.array([ self.data.raw.nontransformer_branches[line_key].st for line_key in self.line_key])
        swqual = np.array([ self.data.sup.lines[line_key]['swqual'] for line_key in self.line_key])
        check0 = np.array([ (p[0] == p[1]) if p[2] == 0 else True for p in zip(data_line_status,self.line_base_x,swqual)])
        check = all( check0 == True)
        f = np.argmax( check0 == False)
        eg = self.line_key [  f    ] if f > 0 else ""        
        print_alert("The closed-open status for lines that are not qualified to switch must remain to the value in the given operating point prior to the base case. {}".format(eg), check_passed=check)

        #x_fk^sw, x_fk^st
        sol_xfmr_iorig = solution1.xfmr_df.iorig.values
        sol_xfmr_idest = solution1.xfmr_df.idest.values
        sol_xfmr_id = map(clean_string, list(solution1.xfmr_df.id.values))
        sol_xfmr_key = zip(sol_xfmr_iorig, sol_xfmr_idest, sol_xfmr_id)  #CHALLENGE2
        self.base_xfmr_map = dict(zip(sol_xfmr_key, list(range(self.num_xfmr))))
        #xfmr_permutation =  [self.base_xfmr_map[k] for k in self.data.raw.transformers.keys()] 
        xfmr_permutation =  [self.base_xfmr_map[k] for k in self.xfmr_key] 

        self.xfmr_base_x = np.around( solution1.xfmr_df.x.values[xfmr_permutation] )        #open/closed
        self.xfmr_base_xst = np.around( solution1.xfmr_df.xst.values[xfmr_permutation] )        #open/closed

        #Each transformer may be closed or open, and the closed-open status is indicated by a binary variable
        #Check domains of integer input variables  #55 - 57
        # C2 A1 S4 #55
        check0 = [x == 0 or x == 1. for x in self.xfmr_base_x]
        check = all(check0)
        f = np.argmax( check0 == False)
        eg = self.xfmr_key [  f    ] if f > 0 else ""                
        print_alert("Xfmr Closed-open status indicator must be zero or one.", check_passed=check)

        # Check domains of integer input variables  #56
        #The closed-open status for transformers that are not qualiﬁed to switch must remain to the value in the given operating point prior to the base case                
        # C2 A1 S4 #56
        data_xfmr_status = np.array([ self.data.raw.transformers[xfmr_key].stat for xfmr_key in self.xfmr_key])
        swqual = np.array([ self.data.sup.transformers[xfmr_key]['swqual'] for xfmr_key in self.xfmr_key])
        swcost = np.array([ self.data.sup.transformers[xfmr_key]['csw'] for xfmr_key in self.xfmr_key])

        check0 = np.array([ (p[0] == p[1]) if p[2] == 0 else True for p in zip(data_xfmr_status,self.xfmr_base_x,swqual)])
        check = all( check0 == True)
        f = np.argmax( check0 == False)
        eg = self.line_key [  f    ] if f > 0 else ""        
        print_alert("The closed-open status for transformers that are not qualified to switch must remain to the value in the given operating point prior to the base case. {}".format(eg), check_passed=check)

        xfmr_base_swstatus = abs(data_xfmr_status - self.xfmr_base_x)
        self.xfmr_base_swcost = xfmr_base_swstatus *  swcost

        # xstf_ub = (NTP1 - 1) / 2 - upper bound  
        # xst comes from solution
        # tau ( rma, rmi) = (rma - rmi) / (NTP1-1)
        # tau_mid  ( rma + rmi) / 2 - midpoint
        # no differentiation between cod1 and cod2
        # COD1 - whether xfmr has fixed or variable tap ratio or phase shift
        # tau0 = raw WINDV1 / WINDV2 = tap ratio 
        # 
        # S15 compute imbalance - get pcblocks
        # S17 partition as low/medium/high - #cost blocks
        #  xst = position number, bounds come from (ntp-)/2
        #  look for xfmr_tau0  between (T1,F1) for the impedence correction table
        #  delta:  (xfmr_tau0 - Tk )/(Tk+1 - Tk)
        #  F = Fk + delta * (F(k+1) - Fk)
        # (xfmr_tau0, F) ---> eta_fk
        # if tab1 <> 0, has impedence correction
        # whether tap ratio or phase shift, depends on COD1 - consider either tau or theta for the search

        ntp1 =  np.array(  [   self.data.raw.transformers[key].ntp1  for key in self.xfmr_key ])
        rma =  np.array( [  self.data.raw.transformers[key].rma1   for key in self.xfmr_key ])
        rmi =  np.array( [  self.data.raw.transformers[key].rmi1  for key in self.xfmr_key ])
        windv1 =  np.array( [  self.data.raw.transformers[key].windv1  for key in self.xfmr_key ])
        windv2 =  np.array( [  self.data.raw.transformers[key].windv2  for key in self.xfmr_key ])
        cod1 =  np.array( [  self.data.raw.transformers[key].cod1  for key in self.xfmr_key ])
        ang1 =  np.array( [  self.data.raw.transformers[key].ang1  for key in self.xfmr_key ])
        tab1 =  np.array( [  self.data.raw.transformers[key].tab1  for key in self.xfmr_key ])

        xstf_ub     = (ntp1 -1)/2
        tau_mid     = ( rma + rmi) / 2
        xfmr_tau0   = windv1 / windv2       #tap ratio
        tau_ub      = np.array( [ p[0] if p[1]==1 else p[2]  for p in zip(rma, cod1, xfmr_tau0   )]   )
        tau_lb      = np.array( [ p[0] if p[1]==1 else p[2]  for p in zip(rmi, cod1, xfmr_tau0   )]   )
        xfmr_tau_st = (tau_ub - tau_lb)/(2 * xstf_ub)

        theta_mid     = (math.pi/180.0) * ( rma + rmi) / 2
        xfmr_theta0   =  ang1 * math.pi / 180.0 
        theta_ub      = np.array( [ p[0] if p[1]==3 else p[2]  for p in zip(rma, cod1, xfmr_theta0   )]   )
        theta_lb      = np.array( [ p[0] if p[1]==3 else p[2]  for p in zip(rmi, cod1, xfmr_theta0   )]   )
        xfmr_theta_st = (theta_ub - theta_lb)/(2 * xstf_ub)

        #Each transformer has a set of positions, and the selected position is indicated by an integer variable bounded by a minimum position number and a maximum position number
        # C2 A1 S4 #57
        check0_lb = np.array( [p[0] <= p[1] for p in zip( -xstf_ub, self.xfmr_base_xst )])
        check0_ub = np.array([p[0] <= p[1] for p in zip( self.xfmr_base_xst, xstf_ub  )])
        check0 = ( check0_lb  & check0_ub  )
        check = all (check0)
        f = np.argmax( check0 == False)
        eg = self.xfmr_key [  f    ] if f > 0 else ""        
        print_alert(f'Each transformer has a set of positions, and the selected position is indicated by an integer variable bounded by a minimum position number and a maximum position number. {eg}', check_passed = check)

        #The tap ratio of a variable tap ratio transformer depends on the position selected
        # cod1 == 1: variable
        # C2 A1 S4 #58 #59
        self.xfmr_tau = np.array( [ p[0] + p[1] * p[2]  if p[3]==1 else p[4]    for p in zip(tau_mid, xfmr_tau_st, self.xfmr_base_xst, cod1, xfmr_tau0)  ])

        #The phase shift of a variable phase shift transformer depends on the position selected
        # C2 A1 S11 #60 #61
        self.xfmr_theta = np.array( [ p[0] + p[1] * p[2]  if p[3]==3 else p[4]    for p in zip(theta_mid, xfmr_theta_st, self.xfmr_base_xst, cod1, xfmr_theta0)  ])

        # Compute transformer series conductance and susceptance variables gfk, bfk 
        # The impedance correction factor ηfk of a transformer f with impedance correction in case k is used to modify the conductance gf and susceptance bf by the constraints 
        # XFMR.TAB1 --> IMPEDENCE CORRECTION TABLE --> I, T1, F1
        # self.xfmr_adm_real     CONDUCTANCE
        # self.xfmr_adm_imag     SUSCEPTANCE

        # CHALLENGE2 REVIEW
        # TAB1      IMPEDENCE CORRECTION IF TAB1 > 0
        # COD1      0 - FIXED TAP AND FIXED PHASE; 1 - VARIABLE TAP AND FIXED PHASE; 3 - FIXED TAP AND VARIABLE PHASE


        # C2 A1 S13 #66 #67
        # (tau, eta), (theta, eta)
        #eta_tau = np.zeros(len(self.xfmr_key))
        #eta_theta = np.zeros(len(self.xfmr_key))
        eta = np.zeros(len(self.xfmr_key)) # just one eta
        for k, key in enumerate(self.xfmr_key):
            tab = tab1[k]                   # HAS IMPEDENCE CORRECTION?
            if tab > 0: # set t,f at this level #HAS IMPEDENCE CORRECTION
                t = self.data.raw.transformer_impedance_correction_tables[tab].t
                f = self.data.raw.transformer_impedance_correction_tables[tab].f

                cod = cod1[k]                   # TAP OR PHASE?
                if cod == 1:        #HAS IMPEDENCE CORRECTION AND CONTROL MODE IS A TAP RATIO
                    #tau_k = xfmr_tau0[k]
                    tau_k = xfmr_tau[k] # resulting tau in the case (base or ctg) not tau in the prior operating point (tau0), same for theta
                    print('tab',tab)
                    print('t',t)
                    print('f',f)
                    print('tauk',tau_k)
                    for i in range(len(t)-1):
                        if t[i] is not None and t[i+1] is not None and t[i] <= tau_k and tau_k <= t[i + 1]:
                            #eta_tau[k] = f[k] + (tau_k - t[i]) * (f[k+1] - f[k])
                            #eta[k] = f[k] + (tau_k - t[i]) * (f[k+1] - f[k]) # just one eta
                            #eta[k] = f[i] + (tau_k - t[i]) * (f[i+1] - f[i]) # i not k
                            eta[k] = f[i] + (tau_k - t[i]) * (f[i+1] - f[i]) / (t[i+1] - t[i]) # need denom
                            break # just in case tau lands exactly on t[i]
                if cod == 3:        #HAS IMPEDENCE CORRECTION AND CONTROL MODE IS A PHASE SHIFT
                    t = [ti * math.pi / 180.0 if ti is not None else None for ti in t] # data has angles in degrees
                    theta_k = xfmr_theta[k]
                    for i in range(len(t)-1):
                        if t[i] is not None and t[i+1] is not None and t[i] <= theta_k and theta_k <= t[i + 1]:
                            eta[k] = f[i] + (theta_k - t[i]) * (f[i+1] - f[i]) / (t[i+1] - t[i])
                            break


        # C2 A1 S13 #62 #64 CONDUCTANCE
        print('tab1',tab1)
        print('self.data.raw.transformer_impedance_correction_tables',self.data.raw.transformer_impedance_correction_tables[1].t)
        print('cod1',cod1)
        #print('eta tau', eta_tau)
        print('eta', eta) # just one eta
        print('self.xfmr_adm_real',self.xfmr_adm_real)
        #self.g_base = np.array( [  p[0]/p[1] if p[2]!=0 and p[3] == 1 else p[0]    for p in zip(self.xfmr_adm_real,eta_tau, tab1, cod1) ]   )
        self.g_base = np.array( [  p[0]/p[1] if p[2]!=0 and p[3] == 1 else p[0]    for p in zip(self.xfmr_adm_real,eta, tab1, cod1) ]   ) # just one eta
        print('gbase',self.g_base)

        '''
        =======
                for i in range(len(t)-1):
                    if t[i] is not None and t[i+1] is not None and t[i] <= tau_k and tau_k <= t[i + 1]:
                        eta_tau[k] = f[k] + (tau_k - t[i]) * (f[k+1] - f[k])

            if tab > 0 and cod == 3:        #HAS IMPEDENCE CORRECTION AND CONTROL MODE IS A PHASE SHIFT
                theta_k = xfmr_theta0[k]
                for i in range(len(t)-1):
                    if t[i] is not None and t[i+1] is not None and t[i] <= theta_k and theta_k <= t[i + 1]:
                        eta_theta[k] = f[k] + (tau_k - t[i]) * (f[k+1] - f[k])


        # C2 A1 S13 #62 #64 CONDUCTANCE
        self.g_base = np.array( [  p[0]/p[1] if p[2]!=0 and p[3] == 1 else p[0]    for p in zip(self.xfmr_adm_real,eta_tau, tab1, cod1) ]   )

        '''

        # C2 A1 S13 #63 #65 SUSCEPTANCE
        #self.b_base = np.array( [  p[0]/p[1] if p[2]!=0 and p[3] == 3  else p[0]    for p in zip(self.xfmr_adm_imag,eta_theta, tab1, cod1) ]     )
        self.b_base = np.array( [  p[0]/p[1] if p[2]!=0 and p[3] == 3  else p[0]    for p in zip(self.xfmr_adm_imag,eta, tab1, cod1) ]     ) # just one eta


        #x_hak^st
        sol_swsh_i = solution1.swsh_df.i.values
        self.base_swsh_map = dict(zip(sol_swsh_i, list(range(self.num_swsh))))

        swsh_permutation = [self.base_swsh_map[k[0]] for k in self.data.raw.active_swsh.keys()] #tuple (11,) to 11
        self.swsh_base_xst1 = np.around( solution1.swsh_df.xst1.values[swsh_permutation] )        #steps activated
        self.swsh_base_xst2 = np.around( solution1.swsh_df.xst2.values[swsh_permutation] )        #steps activated
        self.swsh_base_xst3 = np.around( solution1.swsh_df.xst3.values[swsh_permutation] )        #steps activated
        self.swsh_base_xst4 = np.around( solution1.swsh_df.xst4.values[swsh_permutation] )        #steps activated
        self.swsh_base_xst5 = np.around( solution1.swsh_df.xst5.values[swsh_permutation] )        #steps activated
        self.swsh_base_xst6 = np.around( solution1.swsh_df.xst6.values[swsh_permutation] )        #steps activated
        self.swsh_base_xst7 = np.around( solution1.swsh_df.xst7.values[swsh_permutation] )        #steps activated
        self.swsh_base_xst8 = np.around( solution1.swsh_df.xst8.values[swsh_permutation] )        #steps activated

        sol_load_i = solution1.load_df.i.values
        sol_load_id = solution1.load_df.id.values
        sol_load_id = [txt.strip() for txt in sol_load_id] #AT-0724
        sol_load_key = zip(sol_load_i, sol_load_id )
        self.base_load_map = dict(zip(sol_load_key, list(range(self.num_load))))
        load_permutation = [self.base_load_map[k] for k in self.data.raw.active_loads.keys()]

        # C2 A1 S7
        # load that is cleared is subject to bounds #37
        # Check simple bounds on continuous input variables tjk, i.e. Equations (37). If any violation >  is found, or any tjk < 0, then the solution is deemed infeasible. 
        self.load_base_t = solution1.load_df.t.values[load_permutation]        #cleared fraction


        sup_load_tmin = np.array( [ self.data.sup.loads[k]['tmin']   for k in self.data.raw.active_loads.keys() ])
        sup_load_tmax = np.array( [ self.data.sup.loads[k]['tmax']   for k in self.data.raw.active_loads.keys() ])
        check0_lb =   sup_load_tmin <= self.load_base_t
        check0_ub =    self.load_base_t <= sup_load_tmax
        check0 =   check0_lb  & check0_ub 
        check = all (check0)
        f = np.argmax( check0 == False)
        eg = self.data.raw.active_loads.keys() [  f    ] if f > 0 else ""        
        print_alert(f'load that is cleared is subject to bounds. {eg}', check_passed = check)

        # Each load is subject to ramping limits
        # C2 A1 S9 #40
        deltar = self.data.sup.sup_jsonobj['systemparameters']['deltar']
        prumax = np.array( [  self.data.sup.loads[k]['prumax'] for k in self.data.raw.active_loads.keys() ])
        prdmax = np.array( [  self.data.sup.loads[k]['prdmax'] for k in self.data.raw.active_loads.keys() ])
        p0 =  np.array( [ self.data.raw.loads[k].pl   for k in self.data.raw.active_loads.keys() ]) / self.base_mva
        #CONVERT FRACTION TO REAL POWER
        self.load_base_p = p0 * solution1.load_df.t.values[load_permutation]        #CHALLENGE2 fraction to power
        min_p = -prdmax * deltar
        max_p = prumax *  deltar
        delta_load = self.load_base_p - p0
        check0_lb = np.array( [p[0] <= p[1]  for  p in zip(min_p, delta_load)])
        check0_ub = np.array( [(p[0] <= p[1]) for p in zip(delta_load, max_p) ])
        check0 = ( check0_lb  & check0_ub  )
        check = all (check0)
        f = np.argmax( check0 == False)
        eg = self.data.raw.active_loads.keys() [  f    ] if f > 0 else ""                
        print_alert(f'Each load is subject to ramping limits. {eg}', check_passed=check)


        # Check simple bounds on continuous input variables vik
        # C2 A1 S7
        bus_base_vm = np.around( solution1.bus_df.vm.values[bus_permutation] )        #commitment status
        data_bus_vmin = np.array( [ self.data.raw.buses[k].evlo   for k in self.data.raw.buses.keys() ])
        data_bus_vmax = np.array( [ self.data.raw.buses[k].evhi   for k in self.data.raw.buses.keys() ])
        check0_lb = data_bus_vmin <= bus_base_vm
        check0_ub = bus_base_vm <= data_bus_vmax
        check0 = ( check0_lb  & check0_ub  )
        check = all (check0)
        f = np.argmax( check0 == False)
        eg = self.data.raw.buses.keys() [  f    ] if f > 0 else ""                
        print_alert(f'bus voltages must be within evlo and evhi. {eg}', check_passed = check)

        # self.load_pow_real =  self.load_const_pow_real * cleared fraction
        # use self.load_base_p
        # remove "const"
        self.bus_load_const_pow_real = self.bus_load_matrix.dot(self.load_const_pow_real * self.load_base_p)   # (buses x loads) x (loads x 1)
        self.bus_load_const_pow_imag = self.bus_load_matrix.dot(self.load_const_pow_imag)

    # todo remove, but note checking and coding
    def set_solution2(self, solution2):
        #set values from the solution objects
        #convert to per unit (p.u.) convention

        self.solution2 = solution2

        #CHALLENGE2 CHECK BOUNDS
        self.check_swsh_bounds(solution2)


        self.ctg_current = self.ctg_map[clean_string(solution2.ctg_label)]
        sol_bus_i = solution2.bus_df.i.values
        sol_gen_i = solution2.gen_df.i.values
        #sol_gen_id = solution2.gen_df.id.values
        sol_gen_id = map(clean_string, list(solution2.gen_df.id.values))
        self.ctg_bus_map = dict(zip(sol_bus_i, list(range(self.num_bus))))
        sol_gen_key = zip(sol_gen_i, sol_gen_id)
        self.ctg_gen_map = dict(zip(sol_gen_key, list(range(self.num_gen))))
        #bus_permutation = list(itemgetter(*(self.bus_i))(sol_bus_map))
        #gen_permutation = list(itemgetter(*(self.gen_key))(sol_gen_map))
        bus_permutation = [self.ctg_bus_map[k] for k in self.bus_i] 
        gen_permutation = [self.ctg_gen_map[k] for k in self.gen_key]      #CHALLENGE2
        self.ctg_bus_volt_mag = solution2.bus_df.vm.values[bus_permutation]
        self.ctg_bus_volt_ang = solution2.bus_df.va.values[bus_permutation] * (math.pi / 180.0)
        
        #CHALLENGE2
        self.bus_permutation_swsh = [self.ctg_bus_map[k] for k in self.bus_i if k in self.data.raw.switched_shunts] #CHALLENGE2
  
        self.ctg_gen_pow_real = solution2.gen_df.p.values[gen_permutation]
        self.ctg_gen_pow_imag = solution2.gen_df.q.values[gen_permutation]
        #self.ctg_pow_real_change = solution2.delta / self.base_mva
        #self.ctg_gen_bus_volt_mag = self.ctg_bus_volt_mag[self.gen_bus]


        # Read solution input variables xon gk, xsw ek , xsw fk, xst hak, xst fk, vik, θik, pgk, qgk, tjk from solution ﬁles. 
        # C2 A1 S2

        #CHALLENGE2 - COMMITMENT STATUS FROM SOLUTION

        self.gen_ctg_x = np.around(solution2.gen_df.x.values[gen_permutation] )       #commitment status

        #The commitment variables xon gk, the start up indicators xsu gk, and the shut down indicators xsd gk, are binary variables
        # C2 A1 S4  #77
        check0 =[ x == 0 or x == 1. for x in self.gen_ctg_x]
        check = all(check0)
        f = np.argmax( check0 == False)
        eg = self.gen_key [  f    ] if f > 0 else ""        
        print_alert(f"gen commitment status indicator must be zero or one. {eg}", check_passed=check )


        #Init           STAT generator status, binary 
        #Base           commitment status xon gk 
        #Contingency    commitment status xon gk 
        #The start up and shut down indicators xsu gk, xsd gk, are deﬁned by changes in the commitment status xon gk relative to the prior commitment status
        #C2 A1 S5 #78 TO #80
        base_gen_permutation = [self.base_gen_map[k] for k in self.gen_key]      #CHALLENGE2 REVIEW
        gen_base_x = np.around(self.solution1.gen_df.x.values[base_gen_permutation] )
        self.gen_ctg_commitment_changes = self.gen_ctg_x - gen_base_x

        #startup from 0 to 1, sd - otherwise   
        self.ctg_su =  np.array( [ 1 if p[0]==0 and p[1]==1  else 0    for p in zip(self.gen_base_x,self.gen_ctg_x)])
        self.ctg_sd =  np.array( [ 1 if p[0]==1 and p[1]==0  else 0    for p in zip(self.gen_base_x,self.gen_ctg_x)])

        ctg_su_cost =  np.array(  [  self.data.sup.generators[k]['sucost']  for k in self.gen_key])
        ctg_sd_cost =  np.array(  [  self.data.sup.generators[k]['sdcost']  for k in self.gen_key])
        ctg_on_cost =  np.array(  [  self.data.sup.generators[k]['oncost']  for k in self.gen_key])

        self.gen_ctg_su_cost = ctg_su_cost * self.ctg_su 
        self.gen_ctg_sd_cost = ctg_sd_cost * self.ctg_sd
        self.gen_ctg_on_cost = ctg_on_cost * self.gen_ctg_x


        #79
        lhs = self.gen_ctg_commitment_changes
        rhs = self.ctg_su - self.ctg_sd
        check0 = lhs == rhs
        check = all( check0)
        f = np.argmax( check0 == False)
        eg = self.gen_key [  f    ] if f > 0 else ""        
        print_alert(f'The start up and shut down indicators are inconsistent. {eg}', check_passed=check)

        #80
        check0 =  [s <= 1 for s in  self.ctg_su + self.ctg_sd]
        check = all (check0 )
        f = np.argmax( check0 == False)
        eg = self.gen_key [  f    ] if f > 0 else ""                    
        print_alert(f'no generator may simultaneously start up and shut down. {eg}', check_passed=check)


        #Only generators in Gsu may start up    
        #C2 A1 S6 #85
        sup_gen_suqual = np.array( [ self.data.sup.generators[k]['suqualctg']   for k in self.gen_key ])
        check0 = [ (p[0] == 1 and p[1] == 1) or p[0] == 0      for p in zip(self.ctg_su, sup_gen_suqual) ]
        check = all ( check0      )
        f = np.argmax( check0 == False)
        eg = self.gen_key [  f    ] if f > 0 else ""             
        print_alert(f'Only qualified generators may start up. {eg}', check_passed=check)

        #Only generators in Gsd may shutdown
        #C2 A1 S6 #86
        sup_gen_sdqual = np.array( [ self.data.sup.generators[k]['sdqualctg']   for k in self.gen_key ])
        check0 = [ (p[0] == 1 and p[1] == 1) or p[0] == 0      for p in zip(self.ctg_sd, sup_gen_sdqual) ]
        check = all ( check0      )
        f = np.argmax( check0 == False)
        eg = self.gen_key [  f    ] if f > 0 else ""               
        print_alert(f'Only qualified generators may shutdown. {eg}', check_passed=check)        

        #No generator may start up in the base case and then shut down in a contingency or shut down in the base case and then start up in a contingency:         
        #C2 A1 S6 #87 #88
        check0_lb = np.array( [ not (p[0] == 1 and p[1] == 1)    for p in zip(self.base_su, self.ctg_sd) ])
        check0_ub = np.array( [ not( p[0] == 1 and p[1] == 1 )   for p in zip(self.base_sd, self.ctg_su)])
        check0 =  (check0_lb & check0_ub     )
        check = all(check0)
        f = np.argmax( check0 == False)
        eg = self.gen_key [  f    ] if f > 0 else ""            
        print_alert(f'No generator may start up in the base case and then shut down in a contingency or shut down in the base case and then start up in a contingency', check_passed=check)

        # Energy Bounds
        # The real power output pgk of a committed generator in case k is subject to energy bounds, while decommitted generators have 0 real power output:
        # C2 A1 S9 #81
        gen_raw_max_pow_real = np.array( [ self.data.raw.generators[k].pt   for k in self.gen_key ])/self.base_mva
        gen_raw_min_pow_real = np.array( [ self.data.raw.generators[k].pb   for k in self.gen_key ])/self.base_mva
        check0_lb = np.array( [ p[0] <= p[1] for p in zip( (gen_raw_min_pow_real * self.gen_ctg_x), self.gen_pow_real)])
        check0_ub = np.array( [ p[0] <= p[1] for p in zip(self.gen_pow_real , (gen_raw_max_pow_real * self.gen_ctg_x))])
        check0 = check0_lb & check0_ub     
        check = all(check0)
        f = np.argmax( check0 == False)
        eg = self.gen_key [  f    ] if f > 0 else ""            
        print_alert(f'The real power output pgk of a committed generator in case k is subject to energy bounds, while decommitted generators have 0 real power output. {eg}', check_passed=check)


        # Energy Bounds
        # The reactive power output qgk of a committed generator in case k is subject to energy bounds, while decommitted generators have 0 reactive power output:
        # C2 A1 S9 #82
        gen_raw_max_pow_imag = np.array( [ self.data.raw.generators[k].qt   for k in self.gen_key ])/self.base_mva
        gen_raw_min_pow_imag = np.array( [ self.data.raw.generators[k].qb   for k in self.gen_key ])/self.base_mva
        check0_lb = np.array( [ p[0] <= p[1] for p in zip( (gen_raw_min_pow_imag * self.gen_ctg_x) , self.gen_pow_imag)])
        check0_ub = np.array( [ p[0] <= p[1] for p in zip( self.gen_pow_imag,  (gen_raw_max_pow_imag * self.gen_ctg_x))])
        check0 = check0_lb & check0_ub     
        check = all(check0)
        f = np.argmax( check0 == False)
        eg = self.gen_key [  f    ] if f > 0 else ""            
        print_alert(f'The reactive power output qgk of a committed generator in case k is subject to energy bounds, while decommitted generators have 0 reactive power output. {eg}', check_passed=check)

        # Each generator g in a case k is subject to ramp rate constraints linking the real power output pgk to the prior real power output: 
        # C2 A1 S9 #84
        deltarctg = self.data.sup.sup_jsonobj['systemparameters']['deltarctg']
        prumax = np.array( [  self.data.sup.generators[k]['prumax'] for k in self.gen_key ]) / self.base_mva
        prdmax = np.array( [  self.data.sup.generators[k]['prdmax'] for k in self.gen_key ]) / self.base_mva
        real_pow_base = self.solution1.gen_df.p.values[base_gen_permutation]
        
        delta_real_pow = self.ctg_gen_pow_real - real_pow_base
        check0_lb = np.array( [ p[0] <= p[1] for p in zip( -(prdmax * deltarctg ), delta_real_pow)])
        check0_ub = np.array( [ p[0] <= p[1] for p in zip( delta_real_pow, (prumax * deltarctg ))])
        check0 = check0_lb & check0_ub     
        check = all(check0)
        f = np.argmax( check0 == False)
        eg = self.gen_key [  f    ] if not check else ""            
        print_alert(f'Each generator g in a case k is subject to ramp rate constraints linking the real power output pgk to the prior real power output. {eg}', check_passed=check)


        sol_line_iorig = solution2.line_df.iorig.values
        sol_line_idest = solution2.line_df.idest.values
        sol_line_id = map(clean_string, list(solution2.line_df.id.values))
        sol_line_key = zip(sol_line_iorig, sol_line_idest, sol_line_id)
        self.ctg_line_map = dict(zip(sol_line_key, list(range(self.num_line))))
        line_permutation = [self.ctg_line_map[k] for k in self.line_key]
        self.line_ctg_x = np.around( solution2.line_df.x.values[line_permutation] )        #line open/closed

        # Check domains of integer input variables,     #44
        #Each line may be closed or open, and the closed-open status is indicated by a binary variable
        # C2 A1 S4 #44
        check0 = [x == 0 or x == 1. for x in self.line_ctg_x]
        check = all(check0)
        f = np.argmax( check0 == False)
        eg = self.active_lines [  f    ] if f > 0 else ""            
        print_alert(f"Line Closed-open status indicator must be zero or one. {eg}", check_passed=check)

        # Check domains of integer input variables  #45
        #The closed-open status for lines that are not qualiﬁed to switch must remain to the value in the given operating point prior to the base cas
        # C2 A1 S4 #45
        data_line_status = np.array([ self.data.raw.nontransformer_branches[line_key].st for line_key in self.active_lines])
        swqual = np.array([ self.data.sup.lines[line_key]['swqual'] for line_key in self.active_lines])
        check0 = np.array([ (p[0] == p[1]) if p[2] == 0 else True for p in zip(data_line_status,self.line_ctg_x,swqual)])
        check = all( check0 == True)
        f = np.argmax( check0 == False)
        eg = self.line_key [  f    ] if f > 0 else ""        
        print_alert("The closed-open status for lines that are not qualified to switch must remain to the value in the given operating point prior to the base case. {}".format(eg), check_passed=check)


        sol_xfmr_iorig = solution2.xfmr_df.iorig.values
        sol_xfmr_idest = solution2.xfmr_df.idest.values
        sol_xfmr_id = map(clean_string, list(solution2.xfmr_df.id.values))
        sol_xfmr_key = zip(sol_xfmr_iorig, sol_xfmr_idest, sol_xfmr_id)
        self.ctg_xfmr_map = dict(zip(sol_xfmr_key, list(range(self.num_xfmr))))
        xfmr_permutation = [self.ctg_xfmr_map[k] for k in self.xfmr_key]

        self.xfmr_ctg_x = np.around(solution2.xfmr_df.x.values[xfmr_permutation])        #open/closed
        self.xfmr_ctg_xst = np.around (solution2.xfmr_df.xst.values[xfmr_permutation]  )      #open/closed


        #Each transformer may be closed or open, and the closed-open status is indicated by a binary variable
        #Check domains of integer input variables  #55 - 57
        # C2 A1 S4 #55 - 57
        check0 = [x == 0 or x == 1. for x in self.xfmr_ctg_x]
        check = all(check0)
        f = np.argmax( check0 == False)
        eg = self.active_xfmrs [  f    ] if f > 0 else ""            
        print_alert(f"Xfmr Closed-open status indicator must be zero or one. {eg}", check_passed=check)

        # Check domains of integer input variables  #56
        #The closed-open status for trandformer that are not qualiﬁed to switch must remain to the value in the given operating point prior to the base cas
        # C2 A1 S4 #56
        data_xfmr_status = np.array([ self.data.raw.transformers[xfmr_key].stat for xfmr_key in self.xfmr_key])
        swqual = np.array([ self.data.sup.transformers[xfmr_key]['swqual'] for xfmr_key in self.xfmr_key])
        swcost = np.array([ self.data.sup.transformers[xfmr_key]['csw'] for xfmr_key in self.xfmr_key])

        check0 = np.array([ (p[0] == p[1]) if p[2] == 0 else True for p in zip(data_xfmr_status,self.xfmr_ctg_x,swqual)])
        check = all( check0 == True)
        f = np.argmax( check0 == False)
        eg = self.line_key [  f    ] if f > 0 else ""        
        print_alert("The closed-open status for transformers that are not qualified to switch must remain to the value in the given operating point prior to the base case. {}".format(eg), check_passed=check)

        xfmr_ctg_swstatus = abs(self.xfmr_ctg_x - self.xfmr_base_x)
        self.xfmr_ctg_swcost = xfmr_ctg_swstatus *  swcost        

        # xstf_ub = (NTP1 - 1) / 2 - upper bound  
        # xst comes from solution
        # tau ( rma, rmi) = (rma - rmi) / (NTP1-1)
        # tau_mid  ( rma + rmi) / 2 - midpoint
        #no differentiation between cod1 and cod2
        #COD1 - whether xfmr has fixed or variable tap ratio or phase shift
        #tau0 = raw WINDV1 / WINDV2 = tap ratio 
        # 
        # S15 compute imbalance - get pcblocks
        # S17 partition as low/medium/high - #cost blocks
        #  
        ntp1 =  np.array(  [   self.data.raw.transformers[key].ntp1  for key in self.xfmr_key ])
        rma =  np.array( [  self.data.raw.transformers[key].rma1   for key in self.xfmr_key ])
        rmi =  np.array( [  self.data.raw.transformers[key].rmi1  for key in self.xfmr_key ])
        windv1 =  np.array( [  self.data.raw.transformers[key].windv1  for key in self.xfmr_key ])
        windv2 =  np.array( [  self.data.raw.transformers[key].windv2  for key in self.xfmr_key ])
        cod1 =  np.array( [  self.data.raw.transformers[key].cod1  for key in self.xfmr_key ])
        ang1 =  np.array( [  self.data.raw.transformers[key].ang1  for key in self.xfmr_key ])
        tab1 =  np.array( [  self.data.raw.transformers[key].tab1  for key in self.xfmr_key ])

        xstf_ub     = (ntp1 -1)/2
        tau_mid     = ( rma + rmi) / 2
        xfmr_tau0   = windv1 / windv2
        tau_ub      = np.array( [ p[0] if p[1]==1 else p[2]  for p in zip(rma, cod1, xfmr_tau0   )]   )
        tau_lb      = np.array( [ p[0] if p[1]==1 else p[2]  for p in zip(rmi, cod1, xfmr_tau0   )]   )
        xfmr_tau_st = (tau_ub - tau_lb)/(2 * xstf_ub)

        theta_mid     = (math.pi/180.0) * ( rma + rmi) / 2
        xfmr_theta0   =  ang1 * math.pi / 180.0 
        theta_ub      = np.array( [ p[0] if p[1]==3 else p[2]  for p in zip(rma, cod1, xfmr_theta0   )]   )
        theta_lb      = np.array( [ p[0] if p[1]==3 else p[2]  for p in zip(rmi, cod1, xfmr_theta0   )]   )
        xfmr_theta_st = (theta_ub - theta_lb)/(2 * xstf_ub)

        #Each transformer has a set of positions, and the selected position is indicated by an integer variable bounded by a minimum position number and a maximum position number
        # C2 A1 S4 #57
        check0_lb = np.array([  p[0] <= p[1] for p in zip( -xstf_ub, self.xfmr_ctg_xst ) ])
        check0_ub = np.array([  p[0] <= p[1] for p in zip( self.xfmr_ctg_xst, xstf_ub  ) ])
        check0 = check0_lb & check0_ub     
        check = all(check0)
        f = np.argmax( check0 == False)
        eg = self.xfmr_key [  f    ] if not check else ""            
        print_alert(f'Each transformer has a set of positions, and the selected position is indicated by an integer variable bounded by a minimum position number and a maximum position number. {eg}', check_passed = check)
        if not check:
            print_info((self.xfmr_ctg_xst, xstf_ub))

        #The tap ratio of a variable tap ratio transformer depends on the position selected
        # cod1 == 1: variable
        # C2 A1 S4 #58 #59
        self.xfmr_tau = np.array( [ p[0] + p[1] * p[2]  if p[3]==1 else p[4]    for p in zip(tau_mid, xfmr_tau_st, self.xfmr_ctg_xst, cod1, xfmr_tau0)  ])

        #The phase shift of a variable phase shift transformer depends on the position selected
        # C2 A1 S11 #60 #61
        self.xfmr_theta = np.array( [ p[0] + p[1] * p[2]  if p[3]==3 else p[4]    for p in zip(theta_mid, xfmr_theta_st, self.xfmr_ctg_xst, cod1, xfmr_theta0)  ])



        eta_tau = np.zeros(len(self.xfmr_key))
        eta_theta = np.zeros(len(self.xfmr_key))
        for k, key in enumerate(self.xfmr_key):
            tab = tab1[k]
            cod = cod1[k]
            if tab > 0 and cod == 1:        #HAS IMPEDENCE CORRECTION AND CONTROL MODE IS A TAP RATIO
                tau_k = xfmr_tau0[k]
                t = self.data.raw.transformer_impedance_correction_tables[tab].t
                f = self.data.raw.transformer_impedance_correction_tables[tab].f
                for i in range(len(t)-1):
                    if t[i] is not None and t[i+1] is not None and t[i] <= tau_k and tau_k <= t[i + 1]:
                        eta_tau[k] = f[k] + (tau_k - t[i]) * (f[k+1] - f[k])

            if tab > 0 and cod == 3:        #HAS IMPEDENCE CORRECTION AND CONTROL MODE IS A PHASE SHIFT
                theta_k = xfmr_theta0[k]
                for i in range(len(t)-1):
                    if t[i] is not None and t[i+1] is not None and t[i] <= theta_k and theta_k <= t[i + 1]:
                        eta_theta[k] = f[k] + (tau_k - t[i]) * (f[k+1] - f[k])


        # C2 A1 S13 #62 #64 CONDUCTANCE
        self.g_ctg = np.array( [  p[0]/p[1] if p[2]!=0 and p[3] == 1 else p[0]    for p in zip(self.xfmr_adm_real,eta_tau, tab1, cod1) ]   )

        # C2 A1 S13 #63 #65 SUSCEPTANCE
        self.b_ctg = np.array( [  p[0]/p[1] if p[2]!=0 and p[3] == 3  else p[0]    for p in zip(self.xfmr_adm_imag,eta_theta, tab1, cod1) ]     )



        sol_swsh_i = solution2.swsh_df.i.values
        self.ctg_swsh_map = dict(zip(sol_swsh_i, list(range(self.num_swsh))))
        swsh_permutation = [self.ctg_swsh_map[k[0]] for k in self.data.raw.active_swsh.keys()]
        self.swsh_ctg_xst1 = np.around( solution2.swsh_df.xst1.values[swsh_permutation] )        #activated steps
        self.swsh_ctg_xst2 = np.around( solution2.swsh_df.xst2.values[swsh_permutation] )        #activated steps
        self.swsh_ctg_xst3 = np.around( solution2.swsh_df.xst3.values[swsh_permutation] )        #activated steps
        self.swsh_ctg_xst4 = np.around( solution2.swsh_df.xst4.values[swsh_permutation] )        #activated steps
        self.swsh_ctg_xst5 = np.around( solution2.swsh_df.xst5.values[swsh_permutation] )        #activated steps
        self.swsh_ctg_xst6 = np.around( solution2.swsh_df.xst6.values[swsh_permutation] )        #activated steps
        self.swsh_ctg_xst7 = np.around( solution2.swsh_df.xst7.values[swsh_permutation] )        #activated steps
        self.swsh_ctg_xst8 = np.around( solution2.swsh_df.xst8.values[swsh_permutation] )        #activated steps

        sol_load_i = solution2.load_df.i.values
        sol_load_id = solution2.load_df.id.values
        sol_load_key = zip(sol_load_i, sol_load_id )
        self.ctg_load_map = dict(zip(sol_load_key, list(range(self.num_load))))
        load_permutation = [self.ctg_load_map[k] for k in self.data.raw.active_loads.keys()]

        self.load_ctg_t = solution2.load_df.t.values[load_permutation]        #steps activated

        '''
        #CHALLENGE2 SUM OF CLREARED FRACTION AT EACH LOAD OF A BUS
        self.bus_load_ctg_t = np.zeros(self.num_bus)       
        for load in solution2.load_df.itertuples():           
            bus = load.i
            np_bus_index = self.ctg_bus_map[ bus ]
            self.bus_load_ctg_t += load.t
        '''


        # C2 A1 S7
        # load that is cleared is subject to bounds #37
        # Check simple bounds on continuous input variables tjk, i.e. Equations (37). If any violation >  is found, or any tjk < 0, then the solution is deemed infeasible. 
        sup_load_tmin = np.array( [ self.data.sup.loads[k]['tmin']   for k in self.data.raw.active_loads.keys() ])
        sup_load_tmax = np.array( [ self.data.sup.loads[k]['tmax']   for k in self.data.raw.active_loads.keys() ])

        check_lb0 = sup_load_tmin <= self.load_ctg_t
        check_ub0 = self.load_ctg_t <= sup_load_tmax
        check0 = check0_lb & check0_ub     
        check = all(check0)
        f = np.argmax( check0 == False)
        loads = list(self.data.raw.active_loads.keys())
        eg = loads [  f    ] if f > 0 else ""            
        print_alert(f'load that is cleared is subject to bounds. {eg}', check_passed=check)


        # Each load is subject to ramping limits
        # C2 A1 S9 #41
        deltarctg = self.data.sup.sup_jsonobj['systemparameters']['deltarctg']
        prumax = np.array( [  self.data.sup.loads[k]['prumax'] for k in self.data.raw.loads.keys() ])/self.base_mva
        prdmax = np.array( [  self.data.sup.loads[k]['prdmax'] for k in self.data.raw.loads.keys() ])/self.base_mva
        p0 =  np.array( [ self.data.raw.loads[k].pl   for k in self.data.raw.loads.keys() ])/self.base_mva
        #CONVERT FRACTION TO REAL POWER
        load_base_p = p0 * self.solution1.load_df.t.values[load_permutation]        #CHALLENGE2 
        self.load_ctg_p = p0 * solution2.load_df.t.values[load_permutation]        #CHALLENGE2 
        min_p = -prdmax * deltarctg
        max_p = prumax *  deltarctg
        delta_load = self.load_ctg_p - load_base_p
        check0_lb = np.array( [ p[0] <= p[1]  for  p in zip(min_p, delta_load)])
        check0_ub = np.array([ (p[0] <= p[1]) for p in zip(delta_load, max_p) ])
        check0 = check0_lb & check0_ub     
        check = all(check0)
        f = np.argmax( check0 == False)
        eg = self.data.raw.active_loads.keys() [  f    ] if f > 0 else ""            
        print_alert(f'Each load is subject to ramping limits. {eg}', check_passed=check)



        # Check simple bounds on continuous input variables vik
        # C2 A1 S7
        bus_ctg_vm = np.around( solution2.bus_df.vm.values[bus_permutation] )        #commitment status
        data_bus_vmin = np.array( [ self.data.raw.buses[k].evlo   for k in self.data.raw.buses.keys() ])
        data_bus_vmax = np.array( [ self.data.raw.buses[k].evhi   for k in self.data.raw.buses.keys() ])
        check_lb0 = data_bus_vmin <= bus_ctg_vm
        check_ub0 = bus_ctg_vm <= data_bus_vmax
        check0 = check0_lb & check0_ub     
        check = all(check0)
        f = np.argmax( check0 == False)
        eg = self.data.raw.buses.keys() [  f    ] if f > 0 else ""            
        print_alert(f'bus voltages must be within evlo and evhi. {eg}', check_passed=check)

        # self.load_pow_real =  self.load_const_pow_real * cleared fraction
        # use self.load_base_p
        # remove "const"
        # bus_load_pow_real 
        self.bus_load_const_pow_real = self.bus_load_matrix.dot(self.load_const_pow_real * self.load_ctg_p)   # (buses x loads) x (loads x 1)
        self.bus_load_const_pow_imag = self.bus_load_matrix.dot(self.load_const_pow_imag)

        '''
        self.ctg_gen_bus = [self.bus_map[self.gen_i[i]] for i in range(self.ctg_num_gen)]
        self.ctg_bus_gen_matrix = sp.csr_matrix(
            ([1.0 for i in range(self.ctg_num_gen)],
             (self.ctg_gen_bus, list(range(self.ctg_num_gen)))),
            (self.num_bus, self.ctg_num_gen))
        '''

    @timeit
    def set_service_status_for_ctg(self):
        # do this before each contingency.
        # it sets the vector of inservice statuses for each component

        # this is not a significant time cost

        # C2 todo: vector of in service statuses for each class of grid element
        self.gen_service_status[:] = 1.0 # potentially could save a tiny amount of time by only resetting the entry that went out of service
        self.gen_service_status[self.ctg_gens_out[self.ctg_current]] = 0.0
        self.line_service_status[:] = 1.0
        self.line_service_status[self.ctg_lines_out[self.ctg_current]] = 0.0
        self.xfmr_service_status[:] = 1.0
        self.xfmr_service_status[self.ctg_xfmrs_out[self.ctg_current]] = 0.0
        
        # print_info('num_gen: {}'.format(self.num_gen))
        # print_info('ctg_current: {}'.format(self.ctg_current))
        # #print('ctg_current: ')
        # #print(self.ctg_current)
        # print_info('ctg_gens_out (all ctgs): {}'.format(self.ctg_gens_out))
        # print_info('ctg_gens_out: {}'.format(self.ctg_gens_out[self.ctg_current]))
        # print_info('gen_service_status: {}'.format(self.gen_service_status))
        # print_info('ctg_gen_service_status: {}'.format(self.ctg_gen_service_status))

    # todo abstract these to another class, taking json input
    def write_header(self, det_name):
        """write header line for detailed output
        the detailed output file has a header row, then a row for the base case, then a row for each contingency.
        each row is a comma separated list of fields.

        A short description of each field is given in the end of line comment after each field name in the code below.
        The description gives references to the relevant equations in the formulation,
        specified by the equation numbers in the formulation document in parentheses.
        Formulation numbers in the document may change as they are generated automatically,
        but reasonable efforts are made to keep the descriptions here consistent with the document.
        Most of the fields come in (idx, val) pairs, where idx refers to the index or key of the maximum violation in
        a class of constraints, and val refers to the value of the maximum violation.
        Most of the fields apply to the individual contingency or base case specified in the current row.
        The one exception is the 'obj' field, which gives the cumulative objective value for the base
        case and all contingencies up through the current row.
        """

        with open(det_name, 'w') as out:
        #with open(det_name, 'w', newline='') as out:
        #with open(det_name, 'w', newline='', encoding='utf-8') as out:
        #with open(det_name, 'wb') as out:
            csv_writer = csv.writer(out, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(
                ['ctg', # contingency label for the current contingency, empty if base case
                 'infeas', # binary indicator of infeasibility for the base case or contingency of the current row - 1 indicates infeasible
                 'pen', # penalty value on soft constraint violations in the base case or current contingency (1,6-31)
                 'cost', # generator cost (base case only), 0 in contingencies (1-5)
                 'obj', # obj, = pen + cost, cumulative (i.e. base case + all ctgs through the current row) (1)
                 'vmax-idx', # bus number of maximum violation of bus voltage magnitude upper bounds (32,58)
                 'vmax-val', # value of maximum violation of bus voltage magnitude upper bounds (32,58)
                 'vmin-idx', # bus number of maximum violation of bus voltage magnitude lower bounds (32,58)
                 'vmin-val', # value of maximum violation of bus voltage magnitude lower bounds (32,58)
                 'bmax-idx', # bus number of maximum violation of switched shunt susceptance upper bounds (37,63)
                 'bmax-val', # value of maximum violation of switched shunt susceptance upper bounds (37,63)
                 'bmin-idx', # bus number of maximum violation of switched shunt susceptance lower bounds (37,63)
                 'bmin-val', # value of maximum violation of switched shunt susceptance lower bounds (37,63)
                 'pbal-idx', # bus number of maximum violation of real power balance contraints (46-48,72-74)
                 'pbal-val', # value of maximum violation of real power balance contraints (46-48,72-74)
                 'qbal-idx', # bus number of maximum violation of reactive power balance contraints (49-51,75-77)
                 'qbal-val', # value of maximum violation of reactive power balance contraints (49-51,75-77)
                 'pgmax-idx', # bus and unit id of maximum violation of generator real power upper bounds (33,34)
                 'pgmax-val', # value of maximum violation of generator real power upper bounds (33,34)
                 'pgmin-idx', # bus and unit id of maximum violation of generator real power lower bounds (33,34)
                 'pgmin-val', # value of maximum violation of generator real power lower bounds (33,34)
                 'qgmax-idx', # bus and unit id of maximum violation of generator reactive power upper bounds (35,36,61,62)
                 'qgmax-val', # value of maximum violation of generator reactive power upper bounds (35,36,61,62)
                 'qgmin-idx', # bus and unit id of maximum violation of generator reactive power lower bounds (35,36,61,62)
                 'qgmin-val', # value of maximum violation of generator reactive power lower bounds (35,36,61,62)
                 'qvg1-idx', # bus and unit id of maximum violation of generator pv/pq switching constraints of type 1 (undervoltage -> reactive power at max) (94)
                 'qvg1-val', # value of maximum violation of generator pv/pq switching constraints of type 1 (undervoltage -> reactive power at max)  (94)
                 'qvg2-idx', # bus and unit id of maximum violation of generator pv/pq switching constraints of type 2 (overvoltage -> reactive power at min) (95)
                 'qvg2-val', # value of maximum violation of generator pv/pq switching constraints of type 2 (overvoltage -> reactive power at min) (95)
                 'lineomax-idx', # origin destination and circuit id of maximum violation of line origin flow bounds (52,53,78,79)
                 'lineomax-val', # value of maximum violation of line origin flow bounds (52,53,78,79)
                 'linedmax-idx', # origin destination and circuit id of maximum violation of line destination flow bounds (53,54,79,80)
                 'linedmax-val', # value of maximum violation of line destination flow bounds (53,54,79,80)
                 'xfmromax-idx', # origin destination and circuit id of maximum violation of transformer origin flow bounds (55,56,81,82)
                 'xfmromax-val', # value of maximum violation of transformer origin flow bounds (55,56,81,82)
                 'xfmrdmax-idx', # origin destination and circuit id of maximum violation of transformer destination flow bounds (56,57,82,83)
                 'xfmrdmax-val', # value of maximum violation of transformer destination flow bounds (56,57,82,83)
            ])
            #'''

    # todo abstract these to another class, taking json input
    def write_base(self, det_name):
        """write detail of base case evaluation"""

        with open(det_name, 'a') as out:
        #with open(det_name, 'a', newline='') as out:
        #with open(det_name, 'ab') as out:
            csv_writer = csv.writer(out, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(
                ['', self.infeas, self.obj, self.cost, self.obj,
                 self.max_bus_volt_mag_max_viol[0],
                 self.max_bus_volt_mag_max_viol[1],
                 self.max_bus_volt_mag_min_viol[0],
                 self.max_bus_volt_mag_min_viol[1],
                 self.max_bus_pow_balance_real_viol[0],
                 self.max_bus_pow_balance_real_viol[1],
                 self.max_bus_pow_balance_imag_viol[0],
                 self.max_bus_pow_balance_imag_viol[1],
                 self.max_gen_pow_real_max_viol[0],
                 self.max_gen_pow_real_max_viol[1],
                 self.max_gen_pow_real_min_viol[0],
                 self.max_gen_pow_real_min_viol[1],
                 self.max_gen_pow_imag_max_viol[0],
                 self.max_gen_pow_imag_max_viol[1],
                 self.max_gen_pow_imag_min_viol[0],
                 self.max_gen_pow_imag_min_viol[1],
                 None,
                 0.0,
                 None,
                 0.0,
                 self.max_line_curr_orig_mag_max_viol[0],
                 self.max_line_curr_orig_mag_max_viol[1],
                 self.max_line_curr_dest_mag_max_viol[0],
                 self.max_line_curr_dest_mag_max_viol[1],
                 self.max_xfmr_pow_orig_mag_max_viol[0],
                 self.max_xfmr_pow_orig_mag_max_viol[1],
                 self.max_xfmr_pow_dest_mag_max_viol[0],
                 self.max_xfmr_pow_dest_mag_max_viol[1],
                 ])

    # todo abstract these to another class, taking json input
    def print_base(self):
        """print out summary info on the base case"""

        print_alert(
            "base case summary info: {infeasibility: %s, obj: %s, cost: %s, objective: %s}" % (
                self.infeas, self.obj, self.cost, self.obj), raise_exception=False)        

    # todo abstract these to another class, taking json input
    def write_ctg(self, det_name):
        """write detail of ctg evaluation"""        

        with open(det_name, 'a') as out:
        #with open(det_name, 'a', newline='') as out:
        #with open(det_name, 'ab') as out:
            csv_writer = csv.writer(out, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(
                [self.ctg_label[self.ctg_current], self.ctg_infeas, self.ctg_obj, 0.0, self.obj,
                 self.ctg_max_bus_volt_mag_max_viol[0],
                 self.ctg_max_bus_volt_mag_max_viol[1],
                 self.ctg_max_bus_volt_mag_min_viol[0],
                 self.ctg_max_bus_volt_mag_min_viol[1],
                 self.ctg_max_bus_pow_balance_real_viol[0],
                 self.ctg_max_bus_pow_balance_real_viol[1],
                 self.ctg_max_bus_pow_balance_imag_viol[0],
                 self.ctg_max_bus_pow_balance_imag_viol[1],
                 #self.ctg_max_gen_pow_real_max_viol[0],
                 #self.ctg_max_gen_pow_real_max_viol[1],
                 #self.ctg_max_gen_pow_real_min_viol[0],
                 #self.ctg_max_gen_pow_real_min_viol[1],
                 None,
                 0.0,
                 None,
                 0.0,
                 self.ctg_max_gen_pow_imag_max_viol[0],
                 self.ctg_max_gen_pow_imag_max_viol[1],
                 self.ctg_max_gen_pow_imag_min_viol[0],
                 self.ctg_max_gen_pow_imag_min_viol[1],
                 # self.ctg_max_gen_pvpq1_viol[0],
                 # self.ctg_max_gen_pvpq1_viol[1],
                 # self.ctg_max_gen_pvpq2_viol[0],
                 # self.ctg_max_gen_pvpq2_viol[1],
                 self.ctg_max_line_curr_orig_mag_max_viol[0],
                 self.ctg_max_line_curr_orig_mag_max_viol[1],
                 self.ctg_max_line_curr_dest_mag_max_viol[0],
                 self.ctg_max_line_curr_dest_mag_max_viol[1],
                 self.ctg_max_xfmr_pow_orig_mag_max_viol[0],
                 self.ctg_max_xfmr_pow_orig_mag_max_viol[1],
                 self.ctg_max_xfmr_pow_dest_mag_max_viol[0],
                 self.ctg_max_xfmr_pow_dest_mag_max_viol[1],
                 ])

    @timeit
    def eval_gen_xsusd(self):
        #Init           STAT generator status, binary 
        #Base           commitment status xon gk 
        #Contingency    commitment status xon gk 
        #The start up and shut down indicators xsu gk, xsd gk, are deﬁned by changes in the commitment status xon gk relative to the prior commitment status
        #C2 A1 S5 #78 TO #80

        np.subtract(self.gen_xon, self.gen_xon_prior, out=self.gen_temp) # diff to prior
        np.multiply(self.gen_temp, self.gen_service_status, out=self.gen_temp) # 0 if out of service anyway
        np.clip(self.gen_temp, a_min=0.0, a_max=None, out=self.gen_xsu) # positive part
        np.negative(self.gen_temp, out=self.gen_temp)
        np.clip(self.gen_temp, a_min=0.0, a_max=None, out=self.gen_xsd) # negative part

        self.summarize('gen_switch_up_actual', self.gen_xsu.sum())
        self.summarize('gen_switch_down_actual', self.gen_xsd.sum())

        #su_max = su_qual * (1 - x_prior) * stat
        np.subtract(1, self.gen_xon_prior, out=self.gen_temp)
        np.multiply(self.gen_temp, self.gen_su_qual, out=self.gen_temp)
        np.multiply(self.gen_temp, self.gen_service_status, out=self.gen_temp)
        self.summarize('gen_switch_up_max', self.gen_temp.sum())

        #sd_max = sd_qual * x_prior * stat
        np.multiply(self.gen_xon_prior, self.gen_sd_qual, out=self.gen_temp)
        np.multiply(self.gen_temp, self.gen_service_status, out=self.gen_temp)
        self.summarize('gen_switch_down_max', self.gen_temp.sum())

    @timeit
    def eval_gen_xsusd_qual(self):
        
        #Only generators in Gsu may start up    
        #C2 A1 S6 #85
        np.subtract(self.gen_xsu, self.gen_su_qual, out=self.gen_temp)
        self.summarize('gen_su_qual', self.gen_temp, self.gen_key, self.epsilon)
        #Only generators in Gsd may shutdown
        #C2 A1 S6 #86
        np.subtract(self.gen_xsd, self.gen_sd_qual, out=self.gen_temp)
        self.summarize('gen_sd_qual', self.gen_temp, self.gen_key, self.epsilon)
            
    @timeit
    def eval_gen_xsusd_not_both(self):
        
        #No generator may start up in the base case and then shut down in a contingency or shut down in the base case and then start up in a contingency:         
        #C2 A1 S6 #87 #88
        np.add(self.gen_xsu_prior, self.gen_xsd, out=self.gen_temp)
        np.subtract(self.gen_temp, 1.0, out=self.gen_temp)
        self.summarize('gen_su_sd_not_both', self.gen_temp, self.gen_key, self.epsilon)
        np.add(self.gen_xsd_prior, self.gen_xsu, out=self.gen_temp)
        np.subtract(self.gen_temp, 1.0, out=self.gen_temp)
        self.summarize('gen_sd_su_not_both', self.gen_temp, self.gen_key, self.epsilon)
            
    @timeit
    def eval_line_xsw_qual(self):
        # C2 A1 S6 #85

        #print_info('debug: {}'.format(self.line_xsw))
        #print_info('debug: {}'.format(self.line_xsw_prior))
        #print_info('debug: {}'.format(self.line_sw_qual))

        np.subtract(self.line_xsw, self.line_xsw_prior, out=self.line_temp)
        np.multiply(self.line_temp, self.line_service_status, out=self.line_temp) # 0 if out of service anyway
        np.clip(self.line_temp, a_min=0.0, a_max=None, out=self.line_xsu) # positive part
        np.negative(self.line_temp, out=self.line_temp)
        np.clip(self.line_temp, a_min=0.0, a_max=None, out=self.line_xsd) # negative part
        #print(self.line_xsu)
        #print(self.line_xsd)

        # debug
        # for i in range(self.num_line):
        #     if self.line_xsu[i] > 0.0 or self.line_xsd[i] > 0.0:
        #         print('line switching: [index: {}, i: {}, j: {}, ckt: {}, x: {}, x_prior: {}, stat: {}, xsu: {}, xsd: {}, sw_qual: {}, sw_allowed: {}]'.format(
        #                 i, self.line_i[i], self.line_j[i], self.line_ckt[i], self.line_xsw[i], self.line_xsw_prior[i],
        #                 self.line_xsw_0[i], self.line_xsu[i], self.line_xsd[i], self.line_sw_qual[i], self.line_switching_allowed))

        self.summarize('line_switch_up_actual', self.line_xsu.sum())
        self.summarize('line_switch_down_actual', self.line_xsd.sum())

        #su_max = sw_qual * (1 - x_prior) * stat - this is not checking a constraint, just getting problem info
        np.subtract(1, self.line_xsw_prior, out=self.line_temp)
        np.multiply(self.line_temp, self.line_sw_qual, out=self.line_temp)
        np.multiply(self.line_temp, self.line_service_status, out=self.line_temp)
        self.summarize('line_switch_up_max', self.line_temp.sum())

        #sd_max = sw_qual * x_prior * stat - this is not checking a constraint, just getting problem info
        np.multiply(self.line_xsw_prior, self.line_sw_qual, out=self.line_temp)
        np.multiply(self.line_temp, self.line_service_status, out=self.line_temp)
        self.summarize('line_switch_down_max', self.line_temp.sum())

        # check sw qual
        np.add(self.line_xsu, self.line_xsd, out=self.line_temp) # absolute value
        np.subtract(self.line_temp, self.line_sw_qual, out=self.line_temp) # exceedance
        self.summarize('line_sw_qual', self.line_temp, self.line_key, self.epsilon)
    
    @timeit
    def eval_xfmr_xsw_qual(self):
        # C2 A1 S6 #86

        np.subtract(self.xfmr_xsw, self.xfmr_xsw_prior, out=self.xfmr_temp)
        np.multiply(self.xfmr_temp, self.xfmr_service_status, out=self.xfmr_temp) # 0 if out of service anyway
        np.clip(self.xfmr_temp, a_min=0.0, a_max=None, out=self.xfmr_xsu) # positive part
        np.negative(self.xfmr_temp, out=self.xfmr_temp)
        np.clip(self.xfmr_temp, a_min=0.0, a_max=None, out=self.xfmr_xsd) # negative part

        self.summarize('xfmr_switch_up_actual', self.xfmr_xsu.sum())
        self.summarize('xfmr_switch_down_actual', self.xfmr_xsd.sum())

        #su_max = sw_qual * (1 - x_prior) * stat - this is not checking a constraint, just getting problem info
        np.subtract(1, self.xfmr_xsw_prior, out=self.xfmr_temp)
        np.multiply(self.xfmr_temp, self.xfmr_sw_qual, out=self.xfmr_temp)
        np.multiply(self.xfmr_temp, self.xfmr_service_status, out=self.xfmr_temp)
        self.summarize('xfmr_switch_up_max', self.xfmr_temp.sum())

        #sd_max = sw_qual * x_prior * stat - this is not checking a constraint, just getting problem info
        np.multiply(self.xfmr_xsw_prior, self.xfmr_sw_qual, out=self.xfmr_temp)
        np.multiply(self.xfmr_temp, self.xfmr_service_status, out=self.xfmr_temp)
        self.summarize('xfmr_switch_down_max', self.xfmr_temp.sum())

        # check sw qual
        np.add(self.xfmr_xsu, self.xfmr_xsd, out=self.xfmr_temp) # absolute value
        np.subtract(self.xfmr_temp, self.xfmr_sw_qual, out=self.xfmr_temp) # exceedance
        self.summarize('xfmr_sw_qual', self.xfmr_temp, self.xfmr_key, self.epsilon)

    @timeit
    def eval_gen_xon_bounds(self):
        # C2 A1 S4 #77
        
        np.subtract(self.gen_xon, 1.0, out=self.gen_temp)
        self.summarize('gen_xon_max', self.gen_temp, self.gen_key, self.epsilon)
        np.negative(self.gen_xon, out=self.gen_temp)
        self.summarize('gen_xon_min', self.gen_temp, self.gen_key, self.epsilon)

    @timeit
    def eval_line_xsw_bounds(self):
        # C2 A1 S #44
        
        np.subtract(self.line_xsw, 1.0, out=self.line_temp)
        self.summarize('line_xsw_max', self.line_temp, self.line_key, self.epsilon)
        np.negative(self.line_xsw, out=self.line_temp)
        self.summarize('line_xsw_min', self.line_temp, self.line_key, self.epsilon)

    @timeit
    def eval_xfmr_xsw_bounds(self):
        # C2 A1 S #55
        
        np.subtract(self.xfmr_xsw, 1.0, out=self.xfmr_temp)
        self.summarize('xfmr_xsw_max', self.xfmr_temp, self.xfmr_key, self.epsilon)
        np.negative(self.xfmr_xsw, out=self.xfmr_temp)
        self.summarize('xfmr_xsw_min', self.xfmr_temp, self.xfmr_key, self.epsilon)
            
    @timeit
    def eval_xfmr_xst_bounds(self):
        # C2 A1 S #57
        
        np.absolute(self.xfmr_xst, out=self.xfmr_temp)
        np.subtract(self.xfmr_temp, self.xfmr_xst_max, out=self.xfmr_temp)
        self.summarize('xfmr_xst_bounds', self.xfmr_temp, self.xfmr_key, self.epsilon)
            
    @timeit
    def eval_swsh_xst_bounds(self):
        # C2 A1 S #42
        
        np.subtract(self.swsh_block_xst, self.swsh_block_num_steps, out=self.swsh_block_temp)
        np.amax(self.swsh_block_temp, axis=1, out=self.swsh_temp)
        self.summarize('swsh_xst_max', self.swsh_temp, self.swsh_key, self.epsilon)
        np.negative(self.swsh_block_xst, out=self.swsh_block_temp)
        np.amax(self.swsh_block_temp, axis=1, out=self.swsh_temp)
        self.summarize('swsh_xst_min', self.swsh_temp, self.swsh_key, self.epsilon)

    @timeit
    def eval_xfmr_tap(self):
        # C2 A1 S #58-61

        np.multiply(self.xfmr_tap_mag_step_size, self.xfmr_xst, out=self.xfmr_temp)
        np.add(self.xfmr_tap_mag_mid, self.xfmr_temp, out=self.xfmr_tap_mag)
        np.multiply(self.xfmr_tap_ang_step_size, self.xfmr_xst, out=self.xfmr_temp)
        np.add(self.xfmr_tap_ang_mid, self.xfmr_temp, out=self.xfmr_tap_ang)
        #self.xfmr_tap_mag = self.xfmr_tap_mag_0

    def eval_xfmr_imp_corr_single_xfmr(self, t_val, t_points, f_points):
        # C2 A1 S13 #66 #67

        out = 1.0
        n_points = min(len(t_points), len(f_points))
        n_points = len([i for i in range(n_points) if t_points[i] is not None and f_points[i] is not None])
        # t may fall outside [tmin, tmax] due to floating point errors
        if t_val <= t_points[0]:
            out = f_points[0]
        elif t_points[n_points - 1] <= t_val:
            out = f_points[n_points - 1]
        else:
            for i in range(n_points - 1):
                if t_val < t_points[i + 1]:
                    out = f_points[i] + (t_val - t_points[i]) * (f_points[i + 1] - f_points[i]) / (t_points[i + 1] - t_points[i])
                    break
        return out

    @timeit
    def eval_xfmr_imp_corr(self):
        # C2 A1 S13 #66 #67

        # todo
        # implement - done
        # correct - done
        # refactor - done
        # performance
        #   probably not necessary, most instances have very few transformers with impedance correction

        #self.xfmr_imp_corr[:] = 1.0

        for ind in self.xfmr_index_imp_corr_var_tap_ratio:
            key = self.xfmr_key[ind]
            tab = self.data.raw.transformers[key].tab1
            t = self.data.raw.transformer_impedance_correction_tables[tab].t
            f = self.data.raw.transformer_impedance_correction_tables[tab].f
            self.xfmr_imp_corr[ind] = self.eval_xfmr_imp_corr_single_xfmr(
                self.xfmr_tap_mag[ind], t, f)
        for ind in self.xfmr_index_imp_corr_var_phase_shift:
            key = self.xfmr_key[ind]
            tab = self.data.raw.transformers[key].tab1
            t = self.data.raw.transformer_impedance_correction_tables[tab].t
            f = self.data.raw.transformer_impedance_correction_tables[tab].f
            t = [ti * math.pi / 180.0 if ti is not None else None for ti in t]
            self.xfmr_imp_corr[ind] = self.eval_xfmr_imp_corr_single_xfmr(
                self.xfmr_tap_ang[ind], t, f)

    @timeit
    def eval_xfmr_adm(self):
        # C2 A1 S #62-65

        self.xfmr_adm_real = self.xfmr_adm_real_0 / self.xfmr_imp_corr
        self.xfmr_adm_imag = self.xfmr_adm_imag_0 / self.xfmr_imp_corr
            
    @timeit
    def eval_load_t_viol(self):
        # C2 A1 S #37

        # t max
        np.subtract(self.load_t, self.load_t_max, out=self.load_temp)
        np.clip(self.load_temp, a_min=0.0, a_max=None, out=self.load_temp)
        self.summarize('load_t_max_viol', self.load_temp, self.load_key, self.epsilon)

        # t min
        np.subtract(self.load_t_min, self.load_t, out=self.load_temp)
        np.clip(self.load_temp, a_min=0.0, a_max=None, out=self.load_temp)
        self.summarize('load_t_min_viol', self.load_temp, self.load_key, self.epsilon)

    @timeit
    def proj_load_t(self):

        if self.cfg.proj_load_t:
            np.clip(self.load_t, a_min=self.load_t_min, a_max=self.load_t_max, out=self.load_t)
        if self.cfg.proj_load_p_ramp:
            #np.multiply(self.load_ramp_up_max, self.delta_r, out=self.load_temp)
            #np.add(self.load_temp, self.load_pow_real_prior, out=self.load_temp)
            # CAUTION: need to project t based on ramp, then evaluate p and q
            # todo
            pass

    @timeit
    def eval_load_ramp_viol(self):
        # C2 A1 S #40,41

        # p - p0 <= ru * delta_r
        # max(0.0, p - p0 - ru * delta_r) <= 0.0
        np.multiply(self.load_ramp_up_max, self.delta_r, out=self.load_temp)
        np.add(self.load_temp, self.load_pow_real_prior, out=self.load_temp)
        np.subtract(self.load_pow_real, self.load_temp, out=self.load_temp)
        np.clip(self.load_temp, a_min=0.0, a_max=None, out=self.load_temp)
        self.summarize('load_ramp_up_max_viol', self.load_temp, self.load_key, self.epsilon)
        
        # p0 - p <= rd * delta_r
        # max(0.0, p0 - p - rd * delta_r) <= 0.0
        np.multiply(self.load_ramp_down_max, self.delta_r, out=self.load_temp)
        np.add(self.load_temp, self.load_pow_real, out=self.load_temp)
        np.subtract(self.load_pow_real_prior, self.load_temp, out=self.load_temp)
        np.clip(self.load_temp, a_min=0.0, a_max=None, out=self.load_temp)
        self.summarize('load_ramp_down_max_viol', self.load_temp, self.load_key, self.epsilon)

    @timeit
    def eval_load_benefit(self):
        # C2 A1 S18 #13-16

        #print_info('checking load benefit')
        #print_info(self.load_pow_real)
        #print_info(self.load_benefit)
        # evaluate benefit rate
        self.load_cost_evaluator.eval_benefit(self.load_pow_real, self.load_benefit)
        #print_info(self.load_pow_real)
        #print_info(self.load_benefit)
        total_load_benefit = np.sum(self.load_benefit) * self.delta
        self.summarize('total_load_benefit', total_load_benefit)

        # scale by time interval
        np.multiply(self.load_benefit, self.delta, out=self.load_benefit)

        self.summarize('load_benefit', self.load_benefit, self.load_key)
        self.summarize('min_total_load_benefit', self.min_total_load_benefit * self.delta) # apply delta here
        self.summarize('max_total_load_benefit', self.max_total_load_benefit * self.delta) # apply delta here

    @timeit
    def eval_min_max_total_load_benefit(self):
        # call this only in the base case
        # report it in every case
        # apply delta in each case not here
        # note, this does not account for ramping constraints

        np.multiply(self.load_pow_real_0, self.load_t_min, out=self.load_temp)
        self.load_cost_evaluator.eval_benefit(self.load_temp, self.load_benefit)
        self.min_total_load_benefit = float(np.sum(self.load_benefit))

        # 2021-06-23 adding max total load benefit
        np.multiply(self.load_pow_real_0, self.load_t_max, out=self.load_temp)
        self.load_cost_evaluator.eval_benefit(self.load_temp, self.load_benefit)
        self.max_total_load_benefit = float(np.sum(self.load_benefit))

        # todo 2021-06-23 account for ramping

    @timeit
    def eval_gen_ramp_viol(self):
        # C2 A1 S #83,84

        # p - p0 <= ru * delta_r
        # p - p0 - ru * delta_r <= 0.0
        #np.multiply(self.gen_ramp_up_max, self.delta_r, out=self.gen_temp)
        #np.add(self.gen_temp, self.gen_pow_real_prior, out=self.gen_temp)
        #np.subtract(self.gen_pow_real, self.gen_temp, out=self.gen_temp)
        #np.clip(self.gen_temp, a_min=0.0, a_max=None, out=self.gen_temp)
        #np.multiply(self.gen_temp, self.gen_service_status, out=self.gen_temp)
        #self.summarize('gen_ramp_up_max_viol', self.gen_temp, self.gen_key, self.epsilon)

        # more accurate version of ramp up constraint
        # p - (p0 + ru * delta_r) * (xon - xsu) - (pmin + ru * delta_r) * xsu <= 0.0
        # p - [(p0 + ru * delta_r) * xon + (pmin - p0) * xsu] <= 0.0
        # p - [(p0 + ru * delta_r) * xon + pmin * xsu] <= 0.0
        # needs two temp vectors
        np.multiply(self.gen_ramp_up_max, self.delta_r, out=self.gen_temp)
        np.add(self.gen_temp, self.gen_pow_real_prior, out=self.gen_temp)
        np.multiply(self.gen_temp, self.gen_xon, out=self.gen_temp)
        np.multiply(self.gen_pow_real_min, self.gen_xsu, out=self.gen_temp2)
        np.add(self.gen_temp, self.gen_temp2, out=self.gen_temp)
        np.subtract(self.gen_pow_real, self.gen_temp, out=self.gen_temp)
        np.clip(self.gen_temp, a_min=0.0, a_max=None, out=self.gen_temp)
        np.multiply(self.gen_temp, self.gen_service_status, out=self.gen_temp)
        self.summarize('gen_ramp_up_max_viol', self.gen_temp, self.gen_key, self.epsilon)
        
        # p0 - p <= rd * delta_r
        # p0 - p - rd * delta_r <= 0.0
        #np.multiply(self.gen_ramp_down_max, self.delta_r, out=self.gen_temp)
        #np.add(self.gen_temp, self.gen_pow_real, out=self.gen_temp)
        #np.subtract(self.gen_pow_real_prior, self.gen_temp, out=self.gen_temp)
        #np.clip(self.gen_temp, a_min=0.0, a_max=None, out=self.gen_temp)
        #np.multiply(self.gen_temp, self.gen_service_status, out=self.gen_temp)
        #self.summarize('gen_ramp_down_max_viol', self.gen_temp, self.gen_key, self.epsilon)

        # more accurate version of ramp down constraint
        # (p0 - rd * delta_r) * (xon - xsu) - p <= 0.0
        # needs two temp vectors
        np.multiply(self.gen_ramp_down_max, self.delta_r, out=self.gen_temp)
        np.subtract(self.gen_pow_real_prior, self.gen_temp, out=self.gen_temp)
        np.subtract(self.gen_xon, self.gen_xsu, out=self.gen_temp2)
        np.multiply(self.gen_temp, self.gen_temp2, out=self.gen_temp)
        np.subtract(self.gen_temp, self.gen_pow_real, out=self.gen_temp)
        np.clip(self.gen_temp, a_min=0.0, a_max=None, out=self.gen_temp)
        np.multiply(self.gen_temp, self.gen_service_status, out=self.gen_temp)
        self.summarize('gen_ramp_down_max_viol', self.gen_temp, self.gen_key, self.epsilon)

    @timeit
    def eval_gen_cost(self):
        # C2 A1 S21 #25-28

        # start with energy cost
        self.gen_cost_evaluator.eval_cost(self.gen_pow_real, self.gen_cost)
        total_gen_energy_cost = np.sum(self.gen_cost) * self.delta
        self.summarize('total_gen_energy_cost', total_gen_energy_cost)

        # add on (no load) cost
        np.multiply(self.gen_on_cost, self.gen_xon, out=self.gen_temp)
        np.add(self.gen_cost, self.gen_temp, out=self.gen_cost)
        total_gen_on_cost = np.sum(self.gen_temp) * self.delta
        self.summarize('total_gen_on_cost', total_gen_on_cost)

        # scale by time interval
        np.multiply(self.gen_cost, self.delta, out=self.gen_cost)

        # add su cost
        np.multiply(self.gen_su_cost, self.gen_xsu, out=self.gen_temp)
        np.add(self.gen_cost, self.gen_temp, out=self.gen_cost)
        total_gen_su_cost = np.sum(self.gen_temp)
        self.summarize('total_gen_su_cost', total_gen_su_cost)

        # add sd cost
        np.multiply(self.gen_sd_cost, self.gen_xsd, out=self.gen_temp)
        np.add(self.gen_cost, self.gen_temp, out=self.gen_cost)
        total_gen_sd_cost = np.sum(self.gen_temp)
        self.summarize('total_gen_sd_cost', total_gen_sd_cost)

        self.summarize('gen_cost', self.gen_cost, self.gen_key)
        total_gen_cost = np.sum(self.gen_cost)
        self.summarize('total_gen_cost', total_gen_cost)

    @timeit
    def eval_line_cost(self):
        # C2 A1 S #17-20

        # start with limit exceedance
        self.line_cost_evaluator.eval_cost(self.line_pow_mag_max_viol, self.line_cost)
        total_line_limit_cost = np.sum(self.line_cost) * self.delta
        self.summarize('total_line_limit_cost', total_line_limit_cost)

        # scale by time interval
        np.multiply(self.line_cost, self.delta, out=self.line_cost)

        # add switching
        np.subtract(self.line_xsw, self.line_xsw_prior, out=self.line_temp)
        np.multiply(self.line_service_status, self.line_temp, out=self.line_temp) # fixed a bug here - switching cost was being counted even on branches outaged by a contingency
        np.absolute(self.line_temp, out=self.line_temp)
        np.multiply(self.line_sw_cost, self.line_temp, out=self.line_temp)
        np.add(self.line_cost, self.line_temp, out=self.line_cost) # fixed a bug here - we were not adding switching cost in to total cost
        total_line_switch_cost = np.sum(self.line_temp)
        self.summarize('total_line_switch_cost', total_line_switch_cost)

        self.summarize('line_cost', self.line_cost, self.line_key)
        total_line_cost = np.sum(self.line_cost)
        self.summarize('total_line_cost', total_line_cost)

    @timeit
    def eval_xfmr_cost(self):
        # C2 A1 S #25-28

        # start with limit exceedance
        self.xfmr_cost_evaluator.eval_cost(self.xfmr_pow_mag_max_viol, self.xfmr_cost)
        total_xfmr_limit_cost = np.sum(self.xfmr_cost) * self.delta
        self.summarize('total_xfmr_limit_cost', total_xfmr_limit_cost)

        # scale by time interval
        np.multiply(self.xfmr_cost, self.delta, out=self.xfmr_cost)

        # add switching
        np.subtract(self.xfmr_xsw, self.xfmr_xsw_prior, out=self.xfmr_temp)
        np.multiply(self.xfmr_service_status, self.xfmr_temp, out=self.xfmr_temp) # fixed a bug here - switching cost was being counted even on branches outaged by a contingency
        np.absolute(self.xfmr_temp, out=self.xfmr_temp)
        np.multiply(self.xfmr_sw_cost, self.xfmr_temp, out=self.xfmr_temp)
        np.add(self.xfmr_cost, self.xfmr_temp, out=self.xfmr_cost) # fixed a bug here - we were not adding switching cost in to total cost
        total_xfmr_switch_cost = np.sum(self.xfmr_temp)
        self.summarize('total_xfmr_switch_cost', total_xfmr_switch_cost)

        self.summarize('xfmr_cost', self.xfmr_cost, self.xfmr_key)
        total_xfmr_cost = np.sum(self.xfmr_cost)
        self.summarize('total_xfmr_cost', total_xfmr_cost)

    @timeit
    def eval_bus_cost(self):
        # C2 A1 S #3-6
        
        #self.bus_cost[:] = 0.0
        #self.bus_real_cost_evaluator.eval_cost(self.bus_pow_real_imbalance, self.bus_temp)
        #np.add(self.bus_cost, self.bus_temp, out=self.bus_cost)

        # start with real power imbalance
        self.bus_real_cost_evaluator.eval_cost(self.bus_pow_real_imbalance, self.bus_cost)
        total_bus_real_cost = np.sum(self.bus_cost) * self.delta
        self.summarize('total_bus_real_cost', total_bus_real_cost)

        # add imaginary power imbalance
        self.bus_imag_cost_evaluator.eval_cost(self.bus_pow_imag_imbalance, self.bus_temp)
        np.add(self.bus_cost, self.bus_temp, out=self.bus_cost)
        total_bus_imag_cost = np.sum(self.bus_temp) * self.delta
        self.summarize('total_bus_imag_cost', total_bus_imag_cost)

        # scale by time interval
        np.multiply(self.bus_cost, self.delta, out=self.bus_cost)

        self.summarize('bus_cost', self.bus_cost, self.bus_key)
        total_bus_cost = np.sum(self.bus_cost)
        self.summarize('total_bus_cost', total_bus_cost)

    @timeit
    def eval_obj(self):
        '''evaluate the total objective on the current case'''
        #C2 A1 S23 #2

        bus_cost = np.sum(self.bus_cost)
        load_benefit = np.sum(self.load_benefit)
        gen_cost = np.sum(self.gen_cost)
        line_cost = np.sum(self.line_cost)
        xfmr_cost = np.sum(self.xfmr_cost)
        self.obj = - bus_cost + load_benefit - gen_cost - line_cost - xfmr_cost
        #self.summarize('total_bus_cost', bus_cost)
        #self.summarize('total_load_benefit', load_benefit)
        #self.summarize('total_gen_cost', gen_cost)
        #self.summarize('total_line_cost', line_cost)
        #self.summarize('total_xfmr_cost', xfmr_cost)
        self.summarize('obj', self.obj)

    @timeit
    def eval_prior_bus_pow(self):
        '''evaluate the prior point power imbalance, add to summary2'''

        self.set_sol_from_data()
        self.eval_load_pow()
        self.eval_fxsh_pow()
        self.eval_line_pow()
        self.eval_xfmr_tap()
        self.eval_xfmr_imp_corr()
        self.eval_xfmr_adm()
        self.eval_xfmr_pow()
        self.eval_swsh_adm()
        self.eval_swsh_pow()
        self.eval_bus_pow()
        
        self.summary2['prior_max_bus_pow_real_over'] = self.summary['max_bus_pow_real_over']['val']
        self.summary2['prior_sum_bus_pow_real_over'] = self.summary['sum_bus_pow_real_over']['val']
        self.summary2['prior_max_bus_pow_real_under'] = self.summary['max_bus_pow_real_under']['val']
        self.summary2['prior_sum_bus_pow_real_under'] = self.summary['sum_bus_pow_real_under']['val']
        self.summary2['prior_bus_pow_real_imbalance'] = self.summary['bus_pow_real_imbalance']['val']
        
        self.summary2['prior_max_bus_pow_imag_over'] = self.summary['max_bus_pow_imag_over']['val']
        self.summary2['prior_sum_bus_pow_imag_over'] = self.summary['sum_bus_pow_imag_over']['val']
        self.summary2['prior_max_bus_pow_imag_under'] = self.summary['max_bus_pow_imag_under']['val']
        self.summary2['prior_sum_bus_pow_imag_under'] = self.summary['sum_bus_pow_imag_under']['val']
        self.summary2['prior_bus_pow_imag_imbalance'] = self.summary['bus_pow_imag_imbalance']['val']

    @timeit
    def eval_case(self):
        """evaluate a case solution"""

        # buses
        self.proj_bus_volt_mag()
        self.eval_bus_volt_mag_viol()

        # loads
        self.proj_load_t()
        self.eval_load_t_viol()
        self.eval_load_pow()
        self.eval_load_ramp_viol()
        self.eval_load_benefit()

        # fixed shunts
        self.eval_fxsh_pow()

        # generators
        self.eval_gen_xon_bounds()
        self.eval_gen_xsusd()
        self.eval_gen_xsusd_qual()
        self.eval_gen_xsusd_not_both()
        self.proj_gen_pow()
        self.eval_gen_pow_viol()
        self.eval_gen_ramp_viol()
        self.eval_gen_cost()

        # lines
        self.eval_line_xsw_bounds()
        self.eval_line_xsw_qual()
        self.eval_line_pow()
        self.eval_line_pow_mag_max_viol()
        self.eval_line_cost()

        # transformers
        self.eval_xfmr_xsw_bounds()
        self.eval_xfmr_xst_bounds()
        self.eval_xfmr_xsw_qual()
        self.eval_xfmr_tap()
        self.eval_xfmr_imp_corr()
        self.eval_xfmr_adm()
        self.eval_xfmr_pow()
        self.eval_xfmr_pow_mag_max_viol()
        self.eval_xfmr_cost()

        # switched shunts
        self.eval_swsh_xst_bounds()
        self.eval_swsh_adm()
        self.eval_swsh_pow()

        # bus power balance
        self.eval_bus_pow()
        self.eval_bus_cost()

        # total case objective
        self.eval_obj()

        # case feasibility
        self.eval_infeas()
        #print_info(self.summary)

        # delta to prior - call at the end. it uses computed data
        self.eval_delta_to_prior()

    @timeit
    def eval_delta_to_prior(self):
        '''
        Compute deltas, as max abs over individual elements, called delta_to_prior
        Most important is deltas in bus theta, bus v, load t, gen p, gen q
        Gen x, line x, xfmr x are already considered in switching
        todo: evaluate this only on components that are not outaged by the current case
        but do count changes due to discretionary switching actions
        '''

        np.subtract(self.bus_volt_mag, self.bus_volt_mag_prior, out=self.bus_temp)
        np.absolute(self.bus_temp, out=self.bus_temp)
        self.summarize('bus_volt_mag_delta_to_prior', self.bus_temp, self.bus_key)

        np.subtract(self.bus_volt_ang, self.bus_volt_ang_prior, out=self.bus_temp)
        np.absolute(self.bus_temp, out=self.bus_temp)
        self.summarize('bus_volt_ang_delta_to_prior', self.bus_temp, self.bus_key)

        np.subtract(self.load_pow_real, self.load_pow_real_prior, out=self.load_temp)
        np.absolute(self.load_temp, out=self.load_temp)
        self.summarize('load_pow_real_delta_to_prior', self.load_temp, self.load_key)

        np.subtract(self.gen_pow_real, self.gen_pow_real_prior, out=self.gen_temp)
        np.absolute(self.gen_temp, out=self.gen_temp)
        np.multiply(self.gen_temp, self.gen_service_status, out=self.gen_temp)
        self.summarize('gen_pow_real_delta_to_prior', self.gen_temp, self.gen_key)

        np.subtract(self.gen_pow_imag, self.gen_pow_imag_prior, out=self.gen_temp)
        np.absolute(self.gen_temp, out=self.gen_temp)
        np.multiply(self.gen_temp, self.gen_service_status, out=self.gen_temp)
        self.summarize('gen_pow_imag_delta_to_prior', self.gen_temp, self.gen_key)

        np.subtract(self.xfmr_tap_mag, self.xfmr_tap_mag_prior, out=self.xfmr_temp)
        np.absolute(self.xfmr_temp, out=self.xfmr_temp)
        np.multiply(self.xfmr_temp, self.xfmr_service_status, out=self.xfmr_temp)
        self.summarize('xfmr_tau_delta_to_prior', self.xfmr_temp, self.xfmr_key)

        np.subtract(self.xfmr_tap_ang, self.xfmr_tap_ang_prior, out=self.xfmr_temp)
        np.absolute(self.xfmr_temp, out=self.xfmr_temp)
        np.multiply(self.xfmr_temp, self.xfmr_service_status, out=self.xfmr_temp)
        self.summarize('xfmr_phi_delta_to_prior', self.xfmr_temp, self.xfmr_key)

        np.subtract(self.swsh_adm_imag, self.swsh_adm_imag_prior, out=self.swsh_temp)
        np.absolute(self.swsh_temp, out=self.swsh_temp)
        self.summarize('swsh_b_delta_to_prior', self.swsh_temp, self.swsh_key)

    @timeit
    def eval_infeas(self):

        self.infeas = any([v['infeas'] for v in self.summary.values()])
        #if self.infeas:
        #    
        self.summarize('infeas', self.infeas, tol=0.0)

    @timeit
    def eval_bus_volt_mag_viol(self):
        # C2 A1 S #

        # v min
        np.subtract(self.bus_volt_mag_min, self.bus_volt_mag, out=self.bus_temp)
        np.clip(self.bus_temp, a_min=0.0, a_max=None, out=self.bus_temp)
        self.summarize('bus_volt_mag_min_viol', self.bus_temp, self.bus_key, self.epsilon)

        # v max
        np.subtract(self.bus_volt_mag, self.bus_volt_mag_max, out=self.bus_temp)
        np.clip(self.bus_temp, a_min=0.0, a_max=None, out=self.bus_temp)
        self.summarize('bus_volt_mag_max_viol', self.bus_temp, self.bus_key, self.epsilon)

        #self.bus_volt_mag_min_viol = np.maximum(0.0, self.bus_volt_mag_min - self.bus_volt_mag)
        #self.bus_volt_mag_max_viol = np.maximum(0.0, self.bus_volt_mag - self.bus_volt_mag_max)

    @timeit
    def proj_bus_volt_mag(self):

        if self.cfg.proj_bus_v:
            np.clip(self.bus_volt_mag, a_min=self.bus_volt_mag_min, a_max=self.bus_volt_mag_max, out=self.bus_volt_mag)
        
    # CHALLENGE2
    # Compute load real and reactive power consumption variables pjk, qjk
    # C2 A1 S8 #38 #39
    @timeit
    def eval_load_pow(self):

        np.multiply(self.load_pow_real_0, self.load_t, out=self.load_pow_real)
        np.multiply(self.load_pow_imag_0, self.load_t, out=self.load_pow_imag)

    @timeit
    def eval_fxsh_pow(self):
        # C2 A1 S #

        self.fxsh_pow_real = self.fxsh_adm_real * (self.bus_volt_mag[self.fxsh_bus] ** 2.0)
        self.fxsh_pow_imag = - self.fxsh_adm_imag * (self.bus_volt_mag[self.fxsh_bus] ** 2.0)

    @timeit
    def eval_swsh_adm(self):
        # CHALLENGE2 SUSCEPTANCE sum product
        # The resulting susceptance of a switched shunt is the total susceptance over all blocks and all activated steps:
        # C2 A1 S10 #43     
        
        # todo something more efficient with einsum
        #self.swsh_adm_imag.shape = (self.num_swsh, 1)
        #print('shsh b: {}'.format(self.swsh_adm_imag.shape))
        #print('shsh block b: {}'.format(self.swsh_block_adm_imag.shape))
        #print('shsh block xst: {}'.format(self.swsh_block_xst.shape))
        np.multiply(self.swsh_block_adm_imag, self.swsh_block_xst, out=self.swsh_block_temp)
        np.sum(self.swsh_block_temp, axis=1, out=self.swsh_adm_imag)
        #np.dot(self.swsh_block_adm_imag, self.swsh_block_xst.transpose(), out=self.swsh_adm_imag)
        #self.swsh_adm_imag.shape = (self.num_swsh,)
        #self.swsh_adm_imag = self.swsh_block_adm_imag.dot(self.swsh_block_xst.transpose())

    @timeit
    def eval_swsh_pow(self):
        # C2 A1 S #

        self.swsh_pow_imag = - self.swsh_adm_imag * (self.bus_volt_mag[self.swsh_bus] ** 2.0)

    @timeit
    def eval_gen_pow_viol(self):
        # C2 A1 S #

        # p min
        np.multiply(self.gen_pow_real_min, self.gen_xon, out=self.gen_temp)
        np.subtract(self.gen_temp, self.gen_pow_real, out=self.gen_temp)
        np.clip(self.gen_temp, a_min=0.0, a_max=None, out=self.gen_temp)
        self.summarize('gen_pow_real_min_viol', self.gen_temp, self.gen_key, self.epsilon)

        # p max
        np.multiply(self.gen_pow_real_max, self.gen_xon, out=self.gen_temp)
        np.subtract(self.gen_pow_real, self.gen_temp, out=self.gen_temp)
        np.clip(self.gen_temp, a_min=0.0, a_max=None, out=self.gen_temp)
        self.summarize('gen_pow_real_max_viol', self.gen_temp, self.gen_key, self.epsilon)

        # info - p = 0 if x = 0
        np.subtract(1.0, self.gen_xon, out=self.gen_temp)
        np.multiply(self.gen_pow_real, self.gen_temp, out=self.gen_temp)
        np.absolute(self.gen_temp, out=self.gen_temp)
        p_0_if_x_0_viol = np.amax(self.gen_temp)
        self.summarize('gen_pow_real_0_if_x_0_viol', p_0_if_x_0_viol)

        # q min
        np.multiply(self.gen_pow_imag_min, self.gen_xon, out=self.gen_temp)
        np.subtract(self.gen_temp, self.gen_pow_imag, out=self.gen_temp)
        np.clip(self.gen_temp, a_min=0.0, a_max=None, out=self.gen_temp)
        self.summarize('gen_pow_imag_min_viol', self.gen_temp, self.gen_key, self.epsilon)

        # q max
        np.multiply(self.gen_pow_imag_max, self.gen_xon, out=self.gen_temp)
        np.subtract(self.gen_pow_imag, self.gen_temp, out=self.gen_temp)
        np.clip(self.gen_temp, a_min=0.0, a_max=None, out=self.gen_temp)
        self.summarize('gen_pow_imag_max_viol', self.gen_temp, self.gen_key, self.epsilon)

        # info - q = 0 if x = 0
        np.subtract(1.0, self.gen_xon, out=self.gen_temp)
        np.multiply(self.gen_pow_imag, self.gen_temp, out=self.gen_temp)
        np.absolute(self.gen_temp, out=self.gen_temp)
        q_0_if_x_0_viol = np.amax(self.gen_temp)
        self.summarize('gen_pow_imag_0_if_x_0_viol', q_0_if_x_0_viol)

    @timeit
    def proj_gen_pow(self):

        if self.cfg.proj_gen_p:
            np.clip(self.gen_pow_real, a_min=self.gen_pow_real_min, a_max=self.gen_pow_real_max, out=self.gen_pow_real)
            np.multiply(self.gen_pow_real, self.gen_xon, out=self.gen_pow_real)
        if self.cfg.proj_gen_p_ramp:
            # max
            np.multiply(self.gen_ramp_up_max, self.delta_r, out=self.gen_temp)
            np.add(self.gen_temp, self.gen_pow_real_prior, out=self.gen_temp)
            np.multiply(self.gen_temp, self.gen_xon, out=self.gen_temp)
            np.multiply(self.gen_pow_real_min, self.gen_xsu, out=self.gen_temp2)
            np.add(self.gen_temp, self.gen_temp2, out=self.gen_temp)
            np.multiply(self.gen_temp, self.gen_service_status, out=self.gen_temp)
            np.minimum(self.gen_pow_real, self.gen_temp, out=self.gen_pow_real)
            # min
            np.multiply(self.gen_ramp_down_max, self.delta_r, out=self.gen_temp)
            np.subtract(self.gen_pow_real_prior, self.gen_temp, out=self.gen_temp)
            np.subtract(self.gen_xon, self.gen_xsu, out=self.gen_temp2)
            np.multiply(self.gen_temp, self.gen_temp2, out=self.gen_temp)
            np.multiply(self.gen_temp, self.gen_service_status, out=self.gen_temp)
            np.maximum(self.gen_pow_real, self.gen_temp, out=self.gen_pow_real)
        if self.cfg.proj_gen_q:
            np.clip(self.gen_pow_imag, a_min=self.gen_pow_imag_min, a_max=self.gen_pow_imag_max, out=self.gen_pow_imag)
            np.multiply(self.gen_pow_imag, self.gen_xon, out=self.gen_pow_imag)

    # Real and reactive power ﬂows into a line e at the origin and destination buses in a case k 
    # Compute line and transformer real and reactive power ﬂow variables poek, pd ek, qo ek, qd ek, pofk, pdfk, qofk, qd fk
    @timeit
    def eval_line_pow(self):

        self.line_orig_volt_mag = self.bus_volt_mag[self.line_orig_bus]
        self.line_dest_volt_mag = self.bus_volt_mag[self.line_dest_bus]
        self.line_volt_ang_diff = self.bus_volt_ang[self.line_orig_bus] - self.bus_volt_ang[self.line_dest_bus]
        self.line_cos_volt_ang_diff = np.cos(self.line_volt_ang_diff)
        self.line_sin_volt_ang_diff = np.sin(self.line_volt_ang_diff)
        self.line_orig_dest_volt_mag_prod = self.line_orig_volt_mag * self.line_dest_volt_mag
        self.line_orig_volt_mag_sq = self.line_orig_volt_mag ** 2.0
        self.line_dest_volt_mag_sq = self.line_dest_volt_mag ** 2.0
        # TODO some further factorization can be done, potentially giving more speedup
        
        # C2 A1 S14 #46
        self.line_orig_pow_real = self.line_xsw *      ( # CHALLENGE1 COMMENT - line_status not needed as we have already done it on the parameter level
            self.line_adm_real * self.line_orig_volt_mag_sq + # ** 2.0 +
            ( - self.line_adm_real * self.line_cos_volt_ang_diff
              - self.line_adm_imag * self.line_sin_volt_ang_diff) *
            self.line_orig_dest_volt_mag_prod)

        # C2 A1 S14 #47
        self.line_orig_pow_imag = self.line_xsw * (
            - self.line_adm_total_imag * self.line_orig_volt_mag_sq + # ** 2.0 +
            (   self.line_adm_imag * self.line_cos_volt_ang_diff
              - self.line_adm_real * self.line_sin_volt_ang_diff) *
            self.line_orig_dest_volt_mag_prod)


        # C2 A1 S14 #48
        self.line_dest_pow_real = self.line_xsw * (
            self.line_adm_real * self.line_dest_volt_mag_sq + # ** 2.0 +
            ( - self.line_adm_real * self.line_cos_volt_ang_diff
              + self.line_adm_imag * self.line_sin_volt_ang_diff) *
            self.line_orig_dest_volt_mag_prod)

        # C2 A1 S14 #49
        self.line_dest_pow_imag = self.line_xsw * (
            - self.line_adm_total_imag * self.line_dest_volt_mag_sq + # ** 2.0 +
            (   self.line_adm_imag * self.line_cos_volt_ang_diff
              + self.line_adm_real * self.line_sin_volt_ang_diff) *
            self.line_orig_dest_volt_mag_prod)

        # print('eval line orig pow real: {}'.format(self.line_orig_pow_real))
        # print('eval line orig pow imag: {}'.format(self.line_orig_pow_imag))
        # print('eval line dest pow real: {}'.format(self.line_dest_pow_real))
        # print('eval line dest pow imag: {}'.format(self.line_dest_pow_imag))

    # real and reactive power ﬂows into a line e at the origin and destination buses in a case k are subject to apparent current rating constraints. Any exceedance of these current rating constraints is expressed as a quantity s+ ek of apparent power
    # Current exceedance appears in the objective with a cost coeﬃcient
    # Compute minimal line rating exceedance variables
    # C2 A1 S16 #51 - #52
    # C2 A1 S16 #53 - #54
    @timeit
    def eval_line_pow_mag_max_viol(self):

        np.maximum(
            (self.line_orig_pow_real**2.0 + self.line_orig_pow_imag**2.0)**0.5 -
            self.line_curr_mag_max * self.line_orig_volt_mag,
            (self.line_dest_pow_real**2.0 + self.line_dest_pow_imag**2.0)**0.5 -
            self.line_curr_mag_max * self.line_dest_volt_mag,
            out=self.line_temp)
        np.clip(self.line_temp, a_min=0.0, a_max=None, out=self.line_pow_mag_max_viol)
        max_line_viol = np.amax(self.line_pow_mag_max_viol, initial=0.0)
        self.summarize('max_line_viol', max_line_viol)

    # todo speed up with inplace computation
    # Real and reactive power ﬂows into a transformer f at the origin and destination buses in a case k 
    # C2 A1 S14 #68 - #71
    @timeit
    def eval_xfmr_pow(self):

        self.xfmr_orig_volt_mag = self.bus_volt_mag[self.xfmr_orig_bus]
        self.xfmr_dest_volt_mag = self.bus_volt_mag[self.xfmr_dest_bus]
        self.xfmr_volt_ang_diff = self.bus_volt_ang[self.xfmr_orig_bus] - self.bus_volt_ang[self.xfmr_dest_bus] - self.xfmr_tap_ang
        self.xfmr_cos_volt_ang_diff = np.cos(self.xfmr_volt_ang_diff)
        self.xfmr_sin_volt_ang_diff = np.sin(self.xfmr_volt_ang_diff)
        self.xfmr_orig_dest_volt_mag_prod = self.xfmr_orig_volt_mag * self.xfmr_dest_volt_mag
        self.xfmr_orig_volt_mag_sq = self.xfmr_orig_volt_mag ** 2.0
        self.xfmr_dest_volt_mag_sq = self.xfmr_dest_volt_mag ** 2.0

        # C2 A1 S14 #68
        self.xfmr_orig_pow_real = self.xfmr_xsw * (
            (self.xfmr_adm_real / self.xfmr_tap_mag**2.0 + self.xfmr_adm_mag_real) * self.xfmr_orig_volt_mag_sq +
            ( - self.xfmr_adm_real / self.xfmr_tap_mag * self.xfmr_cos_volt_ang_diff
              - self.xfmr_adm_imag / self.xfmr_tap_mag * self.xfmr_sin_volt_ang_diff) *
                self.xfmr_orig_volt_mag * self.xfmr_dest_volt_mag)

        # C2 A1 S14 #69
        self.xfmr_orig_pow_imag = self.xfmr_xsw * (
            - (self.xfmr_adm_imag / self.xfmr_tap_mag**2.0 + self.xfmr_adm_mag_imag) * self.xfmr_orig_volt_mag_sq +
            (   self.xfmr_adm_imag / self.xfmr_tap_mag * self.xfmr_cos_volt_ang_diff
              - self.xfmr_adm_real / self.xfmr_tap_mag * self.xfmr_sin_volt_ang_diff) *
                self.xfmr_orig_volt_mag * self.xfmr_dest_volt_mag)

        # C2 A1 S14 #70
        self.xfmr_dest_pow_real = self.xfmr_xsw * (
            self.xfmr_adm_real * self.xfmr_dest_volt_mag_sq +
            ( - self.xfmr_adm_real / self.xfmr_tap_mag * self.xfmr_cos_volt_ang_diff
              + self.xfmr_adm_imag / self.xfmr_tap_mag * self.xfmr_sin_volt_ang_diff) *
                self.xfmr_orig_volt_mag * self.xfmr_dest_volt_mag)

        # C2 A1 S14 #70
        self.xfmr_dest_pow_imag = self.xfmr_xsw * (
            - self.xfmr_adm_imag * self.xfmr_dest_volt_mag_sq +
            (   self.xfmr_adm_imag / self.xfmr_tap_mag * self.xfmr_cos_volt_ang_diff
              + self.xfmr_adm_real / self.xfmr_tap_mag * self.xfmr_sin_volt_ang_diff) *
                self.xfmr_orig_volt_mag * self.xfmr_dest_volt_mag)

        # print('eval xfmr orig pow real: {}'.format(self.xfmr_orig_pow_real))
        # print('eval xfmr orig pow imag: {}'.format(self.xfmr_orig_pow_imag))
        # print('eval xfmr dest pow real: {}'.format(self.xfmr_dest_pow_real))
        # print('eval xfmr dest pow imag: {}'.format(self.xfmr_dest_pow_imag))
        # print('eval xfmr tap mag: {}'.format(self.xfmr_tap_mag))
        # print('eval xfmr tap ang: {}'.format(self.xfmr_tap_ang))

    # Real and reactive power flows into a transformer e at the origin and destination buses in a case k are subject to apparent current rating constraints. Any exceedance of these current rating constraints is expressed as a quantity s+ ek of apparent power
    # Current exceedance appears in the objective with a cost coeﬃcient
    # C2 A1 S16 #73 - #74
    @timeit
    def eval_xfmr_pow_mag_max_viol(self):

        # C2 A1 S16 #73
        # C2 A1 S16 #74
        np.maximum(
            (self.xfmr_orig_pow_real**2.0 + self.xfmr_orig_pow_imag**2.0)**0.5 -
            self.xfmr_pow_mag_max,
            (self.xfmr_dest_pow_real**2.0 + self.xfmr_dest_pow_imag**2.0)**0.5 -
            self.xfmr_pow_mag_max,
            out=self.xfmr_temp)
        np.clip(self.xfmr_temp, a_min=0.0, a_max=None, out=self.xfmr_pow_mag_max_viol)
        max_xfmr_viol = np.amax(self.xfmr_pow_mag_max_viol, initial=0.0)
        self.summarize('max_xfmr_viol', max_xfmr_viol)

    # todo - not sure if it is possible to do true in-place computation
    # of matrix-vector product with sparse matrix.
    # Bus real power balance constraints require that the sum of real power output from all generators at a given bus i in a case k is equal to the sum of all real power ﬂows into other grid components at the bus. Any power imbalance is assessed a cost in the objective. Variables p+ ik and p− ik are introduced to represent the positive and negative parts of the net imbalance
    # Compute minimal bus real and reactive power imbalance variables p+ ik, p− ik, q+ ik, q− ik
    # These power imbalance variables then appear in the objective with cost coeﬃcients. The real power balance constraints are then formulated 
    # C2 A1 S15 #31 - #36
    @timeit
    def eval_bus_pow(self):

        # #real pow imbalance abs(p_ik)
        # self.bus_pow_real_imbalance = (
        #     self.bus_gen_matrix.dot(self.gen_pow_real) -
        #     self.bus_load_matrix.dot(self.load_pow_real) -
        #     self.bus_fxsh_matrix.dot(self.fxsh_pow_real) -
        #     self.bus_line_orig_matrix.dot(self.line_orig_pow_real) -
        #     self.bus_line_dest_matrix.dot(self.line_dest_pow_real) -
        #     self.bus_xfmr_orig_matrix.dot(self.xfmr_orig_pow_real) -
        #     self.bus_xfmr_dest_matrix.dot(self.xfmr_dest_pow_real))
        # np.absolute(self.bus_pow_real_imbalance, out=self.bus_pow_real_imbalance)
        # self.summarize('bus_pow_real_imbalance', self.bus_pow_real_imbalance, self.bus_key)        

        # faster way?
        self.bus_pow_real_imbalance[:] = 0.0
        self.bus_pow_real_imbalance[:] += self.bus_gen_matrix.dot(self.gen_pow_real)
        self.bus_pow_real_imbalance[:] -= self.bus_load_matrix.dot(self.load_pow_real)
        self.bus_pow_real_imbalance[:] -= self.bus_fxsh_matrix.dot(self.fxsh_pow_real)
        self.bus_pow_real_imbalance[:] -= self.bus_line_orig_matrix.dot(self.line_orig_pow_real)
        self.bus_pow_real_imbalance[:] -= self.bus_line_dest_matrix.dot(self.line_dest_pow_real)
        self.bus_pow_real_imbalance[:] -= self.bus_xfmr_orig_matrix.dot(self.xfmr_orig_pow_real)
        self.bus_pow_real_imbalance[:] -= self.bus_xfmr_dest_matrix.dot(self.xfmr_dest_pow_real)

        np.maximum(self.bus_pow_real_imbalance, 0.0, out=self.bus_temp)
        self.summarize('max_bus_pow_real_over', self.bus_temp, self.bus_key)
        sum_bus_pow_real_over = np.sum(self.bus_temp)
        self.summarize('sum_bus_pow_real_over', sum_bus_pow_real_over)
        np.negative(self.bus_pow_real_imbalance, out=self.bus_temp)
        np.maximum(self.bus_temp, 0.0, out=self.bus_temp)
        self.summarize('max_bus_pow_real_under', self.bus_temp, self.bus_key)
        sum_bus_pow_real_under = np.sum(self.bus_temp)
        self.summarize('sum_bus_pow_real_under', sum_bus_pow_real_under)
        np.absolute(self.bus_pow_real_imbalance, out=self.bus_pow_real_imbalance)
        self.summarize('bus_pow_real_imbalance', self.bus_pow_real_imbalance, self.bus_key)        

        # self.bus_pow_imag_imbalance = (
        #     self.bus_gen_matrix.dot(self.gen_pow_imag) -
        #     self.bus_load_matrix.dot(self.load_pow_imag) -
        #     self.bus_fxsh_matrix.dot(self.fxsh_pow_imag) -
        #     self.bus_line_orig_matrix.dot(self.line_orig_pow_imag) -
        #     self.bus_line_dest_matrix.dot(self.line_dest_pow_imag) -
        #     self.bus_xfmr_orig_matrix.dot(self.xfmr_orig_pow_imag) -
        #     self.bus_xfmr_dest_matrix.dot(self.xfmr_dest_pow_imag) -
        #     self.bus_swsh_matrix.dot(self.swsh_pow_imag))
        # np.absolute(self.bus_pow_imag_imbalance, out=self.bus_pow_imag_imbalance)
        # self.summarize('bus_pow_imag_imbalance', self.bus_pow_imag_imbalance, self.bus_key)        

        # faster way?
        self.bus_pow_imag_imbalance[:] = 0.0
        self.bus_pow_imag_imbalance[:] += self.bus_gen_matrix.dot(self.gen_pow_imag)
        self.bus_pow_imag_imbalance[:] -= self.bus_load_matrix.dot(self.load_pow_imag)
        self.bus_pow_imag_imbalance[:] -= self.bus_fxsh_matrix.dot(self.fxsh_pow_imag)
        self.bus_pow_imag_imbalance[:] -= self.bus_line_orig_matrix.dot(self.line_orig_pow_imag)
        self.bus_pow_imag_imbalance[:] -= self.bus_line_dest_matrix.dot(self.line_dest_pow_imag)
        self.bus_pow_imag_imbalance[:] -= self.bus_xfmr_orig_matrix.dot(self.xfmr_orig_pow_imag)
        self.bus_pow_imag_imbalance[:] -= self.bus_xfmr_dest_matrix.dot(self.xfmr_dest_pow_imag)
        self.bus_pow_imag_imbalance[:] -= self.bus_swsh_matrix.dot(self.swsh_pow_imag)

        np.maximum(self.bus_pow_imag_imbalance, 0.0, out=self.bus_temp)
        self.summarize('max_bus_pow_imag_over', self.bus_temp, self.bus_key)
        sum_bus_pow_imag_over = np.sum(self.bus_temp)
        self.summarize('sum_bus_pow_imag_over', sum_bus_pow_imag_over)
        np.negative(self.bus_pow_imag_imbalance, out=self.bus_temp)
        np.maximum(self.bus_temp, 0.0, out=self.bus_temp)
        self.summarize('max_bus_pow_imag_under', self.bus_temp, self.bus_key)
        sum_bus_pow_imag_under = np.sum(self.bus_temp)
        self.summarize('sum_bus_pow_imag_under', sum_bus_pow_imag_under)
        np.absolute(self.bus_pow_imag_imbalance, out=self.bus_pow_imag_imbalance)
        self.summarize('bus_pow_imag_imbalance', self.bus_pow_imag_imbalance, self.bus_key)        

        # for debugging
        # print('bus volt mag: {}'.format(self.bus_volt_mag))
        # print('bus volt ang: {}'.format(self.bus_volt_ang))
        # print('gen pow real: {}'.format(self.gen_pow_real))
        # print('load pow real: {}'.format(self.load_pow_real))
        # print('fxsh pow real: {}'.format(self.fxsh_pow_real))
        # print('line orig pow real: {}'.format(self.line_orig_pow_real))
        # print('line dest pow real: {}'.format(self.line_dest_pow_real))
        # print('xfmr orig pow real: {}'.format(self.xfmr_orig_pow_real))
        # print('xfmr dest pow real: {}'.format(self.xfmr_dest_pow_real))
        # print('bus pow real imbalance: {}'.format(self.bus_pow_real_imbalance))
        # print('gen pow imag: {}'.format(self.gen_pow_imag))
        # print('load pow imag: {}'.format(self.load_pow_imag))
        # print('fxsh pow imag: {}'.format(self.fxsh_pow_imag))
        # print('line orig pow imag: {}'.format(self.line_orig_pow_imag))
        # print('line dest pow imag: {}'.format(self.line_dest_pow_imag))
        # print('xfmr orig pow imag: {}'.format(self.xfmr_orig_pow_imag))
        # print('xfmr dest pow imag: {}'.format(self.xfmr_dest_pow_imag))
        # print('swsh pow imag: {}'.format(self.swsh_pow_imag))
        # print('bus pow imag imbalance: {}'.format(self.bus_pow_imag_imbalance))

class CaseSolution:
    '''In model units, i.e. not the physical units of the data convention (different from C1)'''

    def __init__(self):
        '''items to be read from solution_ctg_label.txt'''

        '''Out dimensions are the number of components going out of service
        due to a given contingency'''
        self.num_bus_out = 0
        self.num_load_out = 0
        self.num_gen_out = 0
        self.num_line_out = 0
        self.num_xfmr_out = 0
        self.num_swsh_out = 0

    def set_array_dims(self, evaluation):
        '''Array dimensions are from the number of components (buses, lines, etc.)
        that are part of the problem.'''

        self.num_bus = evaluation.num_bus
        self.num_load = evaluation.num_load
        self.num_gen = evaluation.num_gen
        self.num_line = evaluation.num_line
        self.num_xfmr = evaluation.num_xfmr
        self.num_swsh = evaluation.num_swsh

    def set_ctg(self, evaluation):
        
        if evaluation.ctg_current is not None:
            self.num_gen_out = evaluation.ctg_num_gens_out[evaluation.ctg_current]
            self.num_line_out = evaluation.ctg_num_lines_out[evaluation.ctg_current]
            self.num_xfmr_out = evaluation.ctg_num_xfmrs_out[evaluation.ctg_current]
        #print(self.num_gen_out, self.num_line_out, self.num_xfmr_out)
        
    def set_read_dims(self):
        '''Read dimensions are the number of components to be read from the data file'''

        self.num_bus_read = self.num_bus - self.num_bus_out
        self.num_load_read = self.num_load - self.num_load_out
        self.num_gen_read = self.num_gen - self.num_gen_out
        self.num_line_read = self.num_line - self.num_line_out
        self.num_xfmr_read = self.num_xfmr - self.num_xfmr_out
        self.num_swsh_read = self.num_swsh - self.num_swsh_out

    def set_maps(self, evaluation):

        self.bus_map = evaluation.bus_map
        self.load_map = evaluation.load_map
        self.gen_map = evaluation.gen_map
        self.line_map = evaluation.line_map
        self.xfmr_map = evaluation.xfmr_map
        self.swsh_map = evaluation.swsh_map

    def init_arrays(self):

        self.bus_volt_mag = np.zeros(shape=self.num_bus)
        self.bus_volt_ang = np.zeros(shape=self.num_bus)
        self.load_t = np.zeros(shape=self.num_load)
        self.gen_pow_real = np.zeros(shape=self.num_gen)
        self.gen_pow_imag = np.zeros(shape=self.num_gen)
        self.gen_xon = np.zeros(shape=self.num_gen)
        self.line_xsw = np.zeros(shape=self.num_line)
        self.xfmr_xsw = np.zeros(shape=self.num_xfmr)
        self.xfmr_xst = np.zeros(shape=self.num_xfmr)
        self.swsh_xst = np.zeros(shape=(self.num_swsh, 8))

    # todo
    # this seems to be 20% of the run time of evaluating a single case
    # propably most of it comes from the python list operations
    # is there a faster numpy equivalent?
    # tightening up the solution format requirements would be very helpful here
    # i.e., buses all in the same order in each case, gens, lines, xfmrs, loads, swshs, as well
    # then we could skip the keys - bus numbers, id strings
    # definitely need to look for a numpy equivalent for the list operations
    # use numpy strip to remove leading and trailing whitespace from string fields, or just put it in the documentation - no quotes, no whitespace
    # read the bus num fields as str, then concatenate with id, then look up the str
    # or use numpy.ndarray.astype(str, copy=False)
    # numpy.char.add(i, id) to join, maybe with a comma in between
    # what about sorting to find the permuation in numpy? maybe it is not too expensive?
    # could use pandas update (DF1 -> DF2 overwrites the values with matching key) like join
    @timeit
    def set_arrays_from_dfs(self):

        self.set_bus_arrays_from_df()
        self.set_load_arrays_from_df()
        self.set_gen_arrays_from_df()
        self.set_line_arrays_from_df()
        self.set_xfmr_arrays_from_df()
        self.set_swsh_arrays_from_df()

    @timeit
    def set_bus_arrays_from_df(self):

        bus_key = list(self.bus_df.i.values)
        # check no repeated keys, all keys are in the set of bus keys
        bus_index = [self.bus_map[k] for k in bus_key]
        self.bus_volt_mag[:] = 1.0
        self.bus_volt_mag[bus_index] = self.bus_df.vm.values
        self.bus_volt_ang[:] = 0.0
        self.bus_volt_ang[bus_index] = self.bus_df.va.values

    @timeit
    def set_load_arrays_from_df(self):

        load_i = list(self.load_df.i.values)
        #load_id = map(clean_string, list(self.load_df.id.values))
        load_id = list(self.load_df.id.values)
        load_key = zip(load_i, load_id)
        load_index = [self.load_map[k] for k in load_key]
        self.load_t[:] = 0.0
        self.load_t[load_index] = self.load_df.t.values
 
    @timeit
    def set_gen_arrays_from_df(self):

        gen_i = list(self.gen_df.i.values)
        #gen_id = map(clean_string, list(self.gen_df.id.values))
        gen_id = list(self.gen_df.id.values)
        gen_key = zip(gen_i, gen_id)
        gen_index = [self.gen_map[k] for k in gen_key]
        self.gen_pow_real[:] = 0.0
        self.gen_pow_real[gen_index] = self.gen_df.p.values
        self.gen_pow_imag[:] = 0.0
        self.gen_pow_imag[gen_index] = self.gen_df.q.values
        self.gen_xon[:] = 0.0
        self.gen_xon[gen_index] = self.gen_df.x.values

    @timeit
    def set_line_arrays_from_df(self):

        line_iorig = list(self.line_df.iorig.values)
        line_idest = list(self.line_df.idest.values)
        #line_id = map(clean_string, list(self.line_df.id.values))
        line_id = list(self.line_df.id.values)
        line_key = zip(line_iorig, line_idest, line_id)
        line_index = [self.line_map[k] for k in line_key]
        self.line_xsw[:] = 0.0
        self.line_xsw[line_index] = self.line_df.x.values

    @timeit
    def set_xfmr_arrays_from_df(self):

        xfmr_iorig = list(self.xfmr_df.iorig.values)
        xfmr_idest = list(self.xfmr_df.idest.values)
        #xfmr_id = map(clean_string, list(self.xfmr_df.id.values))
        xfmr_id = list(self.xfmr_df.id.values)
        xfmr_key = zip(xfmr_iorig, xfmr_idest, xfmr_id)
        xfmr_index = [self.xfmr_map[k] for k in xfmr_key]
        self.xfmr_xsw[:] = 0.0
        self.xfmr_xsw[xfmr_index] = self.xfmr_df.x.values
        self.xfmr_xst[:] = 0.0
        self.xfmr_xst[xfmr_index] = self.xfmr_df.xst.values

    @timeit
    def set_swsh_arrays_from_df(self):

        swsh_key = list(self.swsh_df.i.values)
        swsh_index = [self.swsh_map[k] for k in swsh_key]
        # todo maybe can do this in a single assignment
        self.swsh_xst[swsh_index,0] = self.swsh_df.xst1.values
        self.swsh_xst[swsh_index,1] = self.swsh_df.xst2.values
        self.swsh_xst[swsh_index,2] = self.swsh_df.xst3.values
        self.swsh_xst[swsh_index,3] = self.swsh_df.xst4.values
        self.swsh_xst[swsh_index,4] = self.swsh_df.xst5.values
        self.swsh_xst[swsh_index,5] = self.swsh_df.xst6.values
        self.swsh_xst[swsh_index,6] = self.swsh_df.xst7.values
        self.swsh_xst[swsh_index,7] = self.swsh_df.xst8.values

        # check no repeats

        # check all keys exist in the data

    def set_row_start(self):

        self.row_start = { 
            'bus': 2,
            'load': (2 + self.num_bus_read + 2),
            'gen':  (2 + self.num_bus_read + 2 + self.num_load_read + 2),
            'line': (2 + self.num_bus_read + 2 + self.num_load_read + 2 + self.num_gen_read + 2),
            'xfmr': (2 + self.num_bus_read + 2 + self.num_load_read + 2 + self.num_gen_read + 2 + self.num_line_read + 2),
            'swsh': (2 + self.num_bus_read + 2 + self.num_load_read + 2 + self.num_gen_read + 2 + self.num_line_read + 2 + self.num_xfmr_read + 2)
        }
        #print_alert('CaseSolution file skipping rows#: {}'.format(self.row_start), raise_exception = False)
        #print_info('CaseSolution file skipping rows#: {}'.format(self.row_start))

    def check_for_sections(self, file_name):
        # todo include check for section order

        with open(file_name, "r") as file:
            sections = {'--bus section':0, '--load section':0, '--generator section':0, '--line section':0, '--transformer section':0, '--switched shunt section':0}
            for line in file:
                if line.startswith('--'):
                    if line.strip() in sections.keys():
                        sections[line.strip()] = 1
            for k,v in sections.items():
                if v == 0:
                    print_alert('{} missing in {}'.format(k,file_name))

    @timeit
    def round(self):
        #The commitment variables xon gk, the start up indicators xsu gk, and the shut down indicators xsd gk, are binary variables

        np.around(self.gen_xon, out=self.gen_xon)
        np.around(self.line_xsw, out=self.line_xsw)
        np.around(self.xfmr_xsw, out=self.xfmr_xsw)
        np.around(self.xfmr_xst, out=self.xfmr_xst)
        np.around(self.swsh_xst, out=self.swsh_xst)

    @timeit
    # todo
    # seems to be 80% of the run time of a single case eval
    # need to check the cost of constructing a new numpy array of comparable size
    # is it possible that the array construction is the main cost?
    # in which case writing a c reader to put the data into an already existing array
    # could help
    # if array construction is cheap, we can assume that the reading is the main cost
    # and a c reader probably could not improve that - though, who knows? it could be narrowly tailored to our format
    # we would be better off tightening the format
    # on 9462 case,
    #function: read_bus, time: 0.007072925567626953
    # whereas
    # >>> m = 100; n = 100; results = timeit.repeat('a = np.zeros(shape=(10000,5))', setup='import numpy as np', number=n, repeat=m); results = [r/n for r in results]; print(min(results), max(results))
    # 1.9866712391376496e-05 5.4778344929218295e-05
    # so numpy array creation is much less than read_bus time
    # I think it really is just the data reading
    # what if the data were only numeric? a little faster?
    # could we save just by reading fewer columns?
    # OK, what about HDF?
    def read(self, file_name):
        #C2 A1 S2
        #Read solution input variables xon gk, xsw ek , xsw fk, xst hak, xst fk, vik, θik, pgk, qgk, tjk from solution ﬁles. 

        self.set_row_start()
        self.check_for_sections(file_name)
        self.read_bus(file_name)
        self.read_load(file_name)
        self.read_gen(file_name)
        self.read_line(file_name)
        self.read_xfmr(file_name)
        self.read_swsh(file_name)

    @timeit
    def read_bus(self, file_name):

        #print_alert('expecting {} buses'.format(self.num_bus_read) , raise_exception = False)
        print_info('expecting {} buses'.format(self.num_bus_read))
        self.bus_df = pd.read_csv(
            file_name,
            sep=',',
            header=None,
            names=['i', 'vm', 'va'],
            dtype={'i':np.int_, 'vm':np.float_, 'va':np.float_},
            nrows=self.num_bus_read,
            engine='c',
            skiprows=self.row_start['bus'],
            skipinitialspace=True,
            float_precision=pandas_float_precision)
        #print_alert('\tloaded {} buses'.format(self.bus_df.shape[0]), raise_exception = False)
        print_info('\tloaded {} buses'.format(self.bus_df.shape[0]))
        #missing_buses = set(data.raw.buses.keys()).difference(self.bus_df.i.values)
        #if len(missing_buses) > 0:
        #    print_alert('Missing buses in {}: {}'.format(file_name, missing_buses))

    @timeit
    def read_load(self, file_name):

        #print_alert('expecting {} loads'.format(self.num_load_read), raise_exception = False)
        print_info('expecting {} loads'.format(self.num_load_read))
        self.load_df = pd.read_csv(
            file_name,
            sep=',',
            header=None,
            names=['i', 'id', 't'],
            dtype={'i':np.int_, 'id':np.str_, 't':np.float_},
            nrows=self.num_load_read,     #CHALLENGE2 - #RAW LOADS - SELF.LOAD_STATUS=0
            engine='c',
            na_values=None,
            keep_default_na=False,
            skiprows=self.row_start['load'],
            skipinitialspace=True,
            quotechar="'",
            float_precision=pandas_float_precision)
        self.load_df.id = [txt.strip() for txt in self.load_df.id] # AT-0724
        #sol_loads = list(zip(self.load_df.i, self.load_df.id))
        #print_alert('\tloaded {} loads'.format(self.load_df.shape[0]), raise_exception = False)
        print_info('\tloaded {} loads'.format(self.load_df.shape[0]))
        # excess_loads = set(sol_loads).difference( set( data.raw.active_loads.keys()  ))
        # if len(excess_loads) > 0:
        #     print_alert('\nLoads with inactive status found in {}: {}\n'.format(file_name, excess_loads))
        # else:
        #     missing_loads = set( data.raw.active_loads.keys()  ).difference(sol_loads)
        #     if len(missing_loads) > 0:
        #         print_alert('\nMissing loads in {}: {}\n'.format(file_name, missing_loads))

    @timeit
    def read_gen(self, file_name):
        
        #print_alert('expecting {} gens, total in the problem = {}'.format(self.num_gen_read, self.num_gen), raise_exception = False)
        print_info('expecting {} gens, total in the problem = {}'.format(self.num_gen_read, self.num_gen))
        self.gen_df = pd.read_csv(
            file_name,
            sep=',',
            header=None,
            names=['i', 'id', 'p', 'q', 'x'],
            dtype={'i':np.int_, 'id':np.str_, 'p':np.float_, 'q':np.float_, 'x':np.float_}, # read as int? or float?
            nrows=self.num_gen_read,
            engine='c',
            na_values=None,
            keep_default_na=False,
            #quoting=csv.QUOTE_NONE,
            quotechar="'",
            skiprows=self.row_start['gen'],
            skipinitialspace=True,
            float_precision=pandas_float_precision)
        self.gen_df.id = [txt.strip() for txt in self.gen_df.id] #AT-0724
        #sol_gens = list(zip(self.gen_df.i, self.gen_df.id))
        #print_alert('\tloaded {} gens'.format(self.gen_df.shape[0]), raise_exception = False)
        print_info('\tloaded {} gens'.format(self.gen_df.shape[0]))
        # if case == "BASECASE":
        #     excess_gen = set(sol_gens).difference( e.gen_key   )
        #     if len(excess_gen) > 0:
        #         print_alert('\tExtra generators found in {}: {}\n'.format(file_name, excess_gen))
        # else:
        #     excess_gen = set(sol_gens).difference( e.active_gens   )
        #     if len(excess_gen) > 0:
        #         print_alert('\tGenerators with inactive status found in {}: {}\n'.format(file_name, excess_gen))
        # if len(excess_gen) == 0:
        #     if case == "BASECASE":
        #         missing_gen = set( data.raw.generators.keys()  ).difference(sol_gens)
        #         if len(missing_gen) > 0:
        #             print_alert('\tMissing generators in {}: {}\n'.format(file_name, missing_gen))
        #     else:
        #         missing_gen = set( e.active_gens  ).difference(sol_gens)
        #         if len(missing_gen) > 0:
        #             print_alert('\tMissing generators in {}: {}\n'.format(file_name, missing_gen))
        # if  case == "CONTINGENCY":
        #     for inactive_gen_key in e.inactive_gens:
        #         gen = e.data.raw.generators[inactive_gen_key]
        #         k = e.gen_map[inactive_gen_key]
        #         inactive_gen_df = { 'i':  gen.i, 'id':gen.id, 'p':0, 'q':0, 'x':gen.stat }
        #         self.gen_df = Insert_row(k, self.gen_df, inactive_gen_df)

    @timeit
    def read_line(self, file_name):

        #print_alert('expecting {} lines, total in problem = {}'.format(self.num_line_read, self.num_line), raise_exception = False)
        print_info('expecting {} lines, total in problem = {}'.format(self.num_line_read, self.num_line))
        self.line_df = pd.read_csv(
            file_name,
            sep=',',
            header=None,
            names=['iorig', 'idest', 'id', 'x'],
            dtype={'iorig':np.int_, 'idest':np.int_, 'id':np.str_, 'x':np.float_}, # todo read this as int? or float?
            nrows=self.num_line_read,
            engine='c',
            na_values=None,
            keep_default_na=False,
            #quoting=csv.QUOTE_NONE,
            quotechar="'",
            skiprows=self.row_start['line'],
            skipinitialspace=True,
            float_precision=pandas_float_precision)
        self.line_df.id = [txt.strip() for txt in self.line_df.id] #AT-0724
        #sol_lines = list(zip(self.line_df.iorig, self.line_df.idest, self.line_df.id))
        #print_alert('\tloaded {} lines'.format(self.line_df.shape[0]), raise_exception = False)
        print_info('\tloaded {} lines'.format(self.line_df.shape[0]))
        # if case == "BASECASE":
        #     excess_line = set(sol_lines).difference( data.raw.nontransformer_branches.keys()   )
        #     if len(excess_line) > 0:
        #         print_alert('\tExtra lines found in {}: {}\n'.format(file_name, excess_line))
        # else:
        #     excess_line = set(sol_lines).difference( e.active_lines   )
        #     if len(excess_line) > 0:
        #         print_alert('\tlines with inactive status found in {}: {}\n'.format(file_name, excess_line))
        # if len(excess_line) == 0:
        #     if case == "BASECASE":
        #         missing_line = set( data.raw.nontransformer_branches.keys()  ).difference(sol_lines)
        #         if len(missing_line) > 0:
        #             print_alert('\tMissing lines in {}: {}\n'.format(file_name, missing_line))
        #     else:
        #         missing_line = set( e.active_lines  ).difference(sol_lines)
        #         if len(missing_line) > 0:
        #             print_alert('\tMissing lines in {}: {}\n'.format(file_name, missing_line))
        # if  case == "CONTINGENCY":
        #     for inactive_line_key in e.inactive_lines:
        #         line = e.data.raw.nontransformer_branches[inactive_line_key]
        #         k = e.line_map[inactive_line_key]
        #         inactive_line_df = { 'iorig':line.i, 'idest':line.j, 'id':line.ckt, 'x':line.x}
        #         self.line_df = Insert_row(k, self.line_df, inactive_line_df)
      
    @timeit
    def read_xfmr(self, file_name):

        #print_alert('expecting {} xfmrs, total in problem = {}'.format(self.num_xfmr_read, self.num_xfmr), raise_exception = False)
        print_info('expecting {} xfmrs, total in problem = {}'.format(self.num_xfmr_read, self.num_xfmr))
        self.xfmr_df = pd.read_csv(
            file_name,
            sep=',',
            header=None,
            names=['iorig', 'idest', 'id', 'x', 'xst'],
            dtype={'iorig':np.int_, 'idest':np.int_, 'id':np.str_, 'x':np.float_, 'xst':np.float_},
            nrows=self.num_xfmr_read,
            engine='c',
            na_values=None,
            keep_default_na=False,
            #quoting=csv.QUOTE_NONE,
            quotechar="'",
            skiprows=self.row_start['xfmr'],
            skipinitialspace=True,
            float_precision=pandas_float_precision)
        self.xfmr_df.id = [txt.strip() for txt in self.xfmr_df.id] #AT-0724
        #sol_xfmrs = list(zip(self.xfmr_df.iorig, self.xfmr_df.idest,[0]*len(self.xfmr_df.iorig), self.xfmr_df.id))
        #print_alert('\tloaded {} xfmrs'.format(self.xfmr_df.shape[0]), raise_exception = False)
        print_info('\tloaded {} xfmrs'.format(self.xfmr_df.shape[0]))
        # if case == "BASECASE":
        #     excess_xfmr = set(sol_xfmrs).difference( data.raw.transformers.keys()   )
        #     if len(excess_xfmr) > 0:
        #         print_alert('\tExtra xfmrs found in {}: {}\n'.format(file_name, excess_xfmr))
        # else:
        #     excess_xfmr = set(sol_xfmrs).difference( e.active_xfmrs   )
        #     if len(excess_xfmr) > 0:
        #         print_alert('\txfmrs with inactive status found in {}: {}\n'.format(file_name, excess_xfmr))
        # if len(excess_xfmr) == 0:
        #     if case == "BASECASE":
        #         missing_xfmr = set( data.raw.transformers.keys()  ).difference(sol_xfmrs)
        #         if len(missing_xfmr) > 0:
        #             print_alert('\tMissing xfmrs in {}: {}\n'.format(file_name, missing_xfmr))
        #     else:
        #         missing_xfmr = set( e.active_xfmrs  ).difference(sol_xfmrs)
        #         if len(missing_xfmr) > 0:
        #             print_alert('\tMissing xfmrs in {}: {}\n'.format(file_name, missing_xfmr))
        # if  case == "CONTINGENCY":
        #     for inactive_xfmr_key in e.inactive_xfmrs:
        #         xfmr = e.data.raw.transformers[inactive_xfmr_key]
        #         k = e.xfmr_map[inactive_xfmr_key]
        #         inactive_xfmr_df = { 'iorig':xfmr.i, 'idest':xfmr.j, 'id':xfmr.ckt, 'x':xfmr.stat, 'xst':0 }    #CHALLENGE2 REVIEW
        #         self.xfmr_df = Insert_row(k, self.xfmr_df, inactive_xfmr_df)
      
    @timeit
    def read_swsh(self, file_name):

        #print_alert('expecting {} swsh'.format(self.num_swsh_read), raise_exception = False)
        print_info('expecting {} swsh'.format(self.num_swsh_read))
        names=['i', 'xst1', 'xst2', 'xst3', 'xst4', 'xst5', 'xst6', 'xst7', 'xst8']
        self.swsh_df = pd.read_csv(
            file_name,
            sep=',',
            header=None,
            names=names,
            #dtype={'i':np.int_, 'idest':np.int_, 'id':np.str_, 'x':np.int_, 'xst':np.int_},
            #dtype={'i':np.int_, 'idest':np.int_, 'id':np.str_, 'x':np.int_, 'xst':np.int_},
            dtype={'i':np.int_, 'xst1':np.float_, 'xst2':np.float_, 'xst3':np.float_, 'xst4':np.float_, 'xst5':np.float_, 'xst6':np.float_, 'xst7':np.float_, 'xst8':np.float_},
            #dtype={  field: np.float_ for field in names },
            #dtype={  field: np.int_ for field in names },
            nrows=self.num_swsh_read,     #CHALLENGE2: #RAW SWSH - SWSH STATUS == 0
            engine='c',
            #na_values=None,
            #keep_default_na=False,
            #quoting=csv.QUOTE_NONE,
            quotechar="'",
            skiprows=self.row_start['swsh'],
            skipinitialspace=True,
            float_precision=pandas_float_precision)
        
        # todo
        # seems somewhat expensive
        # now that we only do fillna, and do not do downcasting, is it faster?
        # try np.nan_to_num(x, copy=False)
        #CHALLENGE2 - SET MISSING XSTi TO ZERO
        # may be possible to do fillna with inplace=True, downcast
        # need to test on some sol files with missing x
        #self.swsh_df[names] = self.swsh_df[names].fillna(0)
        self.swsh_df[names] = self.swsh_df[names].fillna(0.0)
        #self.swsh_df[names] = self.swsh_df[names].apply(pd.to_numeric, downcast='integer')        

        #sol_swshs = list(zip(self.swsh_df.i))
        #print_alert('\tloaded {} swshs'.format(self.swsh_df.shape[0]), raise_exception = False)
        print_info('\tloaded {} swshs'.format(self.swsh_df.shape[0]))
        # excess_swshs = set(sol_swshs).difference( set( data.raw.active_swsh.keys() ))
        # if len(excess_swshs) > 0:
        #     print_alert('\tswshs with inactive status found in {}: {}\n'.format(file_name, excess_swshs))
        # else:
        #     missing_swshs = set( data.raw.active_swsh.keys()  ).difference(sol_swshs)
        #     if len(missing_swshs) > 0:
        #         print_alert('\tMissing swshs in {}: {}\n'.format(file_name, missing_swshs))

        #CHALLENGE2 - ENSURE XSTi ARE INTS

    # todo remove but noteformat checks and coding
    #CHALLENGE2 - 
    # #rows in gen section - skip out of service gens
    # -do- for lines and xfmrs
    # self.gen_status - commitment status - do not use
    # use self.ctg_gen_keys_out
    #only 1 gen can go out of service per contingency
    # C2 A1 S1
    # Check solution ﬁle format against the speciﬁcation in Appendix C. If a solution ﬁle is formatted incorrectly, then the solution is deemed infeasible. 
    def read_old(self, case, file_name,e):
        
        data = e.data
        
        if case == "BASECASE":
            num_bus = e.num_bus
            num_load = e.data.raw.num_loads_active
            num_gen = e.num_gen
            num_line = e.num_line
            num_xfmr = e.num_xfmr
            num_swsh = e.data.raw.num_swsh_active
        else:
            num_bus = e.num_bus
            num_load = e.data.raw.num_loads_active
            num_gen = e.num_gens_active
            num_line = e.num_lines_active
            num_xfmr =  e.num_xfmrs_active
            num_swsh = e.data.raw.num_swsh_active

        row_start = { 
            'bus': 2,
            'load': (2 + num_bus + 2),
            'gen':  (2 + num_bus + 2 + num_load + 2),
            'line': (2 + num_bus + 2 + num_load + 2 + num_gen + 2),
            'xfmr': (2 + num_bus + 2 + num_load + 2 + num_gen + 2 +  num_line + 2),
            'swsh': (2 + num_bus + 2 + num_load + 2 + num_gen + 2 +  num_line + 2 + num_xfmr + 2)
        }

        print_alert('CaseSolution file skipping rows#: {}'.format(row_start), raise_exception = False)

        file = open(file_name, "r")
        sections = {'--bus section':0, '--load section':0, '--generator section':0, '--line section':0, '--transformer section':0, '--switched shunt section':0}
        for line in file:
            if line.startswith('--'):
                if line.strip() in sections.keys():
                    sections[line.strip()] = 1
        for k,v in sections.items():
            if v == 0:
                print_alert('{} missing in {}'.format(k,file_name), check_passed = (v == 1) )
                    

        print_alert('expecting {} buses'.format(num_bus) , raise_exception = False)

        self.bus_df = pd.read_csv(
            file_name,
            sep=',',
            header=None,
            names=['i', 'vm', 'va'],
            dtype={'i':np.int_, 'vm':np.float_, 'va':np.float_},
            nrows=num_bus,
            engine='c',
            skiprows=row_start['bus'],
            skipinitialspace=True,
            float_precision=pandas_float_precision)

        print_alert('\tloaded {} buses'.format(len(self.bus_df.i.values)), raise_exception = False)


        missing_buses = set(data.raw.buses.keys()).difference(self.bus_df.i.values)
        if len(missing_buses) > 0:
            print_alert('Missing buses in {}: {}'.format(file_name, missing_buses))

        print_alert('total {} loads, active  {} loads'.format(e.num_load, num_load), raise_exception = False)

        self.load_df = pd.read_csv(
            file_name,
            sep=',',
            header=None,
            names=['i', 'id', 't'],
            dtype={'i':np.int_, 'id':np.str_, 't':np.float_},
            nrows=num_load,     #CHALLENGE2 - #RAW LOADS - SELF.LOAD_STATUS=0
            engine='c',
            na_values=None,
            keep_default_na=False,
            skiprows=row_start['load'],
            skipinitialspace=True,
            quotechar="'",
            float_precision=pandas_float_precision)

        self.load_df.id = [txt.strip() for txt in self.load_df.id] # AT-0724
        sol_loads = list(zip(self.load_df.i, self.load_df.id))

        
        print_alert('\tloaded {} loads'.format(len(set(sol_loads))), raise_exception = False)


        excess_loads = set(sol_loads).difference( set( data.raw.active_loads.keys()  ))
        if len(excess_loads) > 0:
            print_alert('\nLoads with inactive status found in {}: {}\n'.format(file_name, excess_loads))
        else:
            missing_loads = set( data.raw.active_loads.keys()  ).difference(sol_loads)
            if len(missing_loads) > 0:
                print_alert('\nMissing loads in {}: {}\n'.format(file_name, missing_loads))

        
        print_alert('expecting {} gens'.format(num_gen), raise_exception = False)

        self.gen_df = pd.read_csv(
            file_name,
            sep=',',
            header=None,
            names=['i', 'id', 'p', 'q', 'x'],
            dtype={'i':np.int_, 'id':np.str_, 'p':np.float_, 'q':np.float_, 'x':np.float_},
            nrows=num_gen,
            engine='c',
            na_values=None,
            keep_default_na=False,
            #quoting=csv.QUOTE_NONE,
            quotechar="'",
            skiprows=row_start['gen'],
            skipinitialspace=True,
            float_precision=pandas_float_precision)

        self.gen_df.id = [txt.strip() for txt in self.gen_df.id] #AT-0724
        sol_gens = list(zip(self.gen_df.i, self.gen_df.id))

        print_alert('\tloaded {} gens'.format(len(set(sol_gens))), raise_exception = False)

        if case == "BASECASE":
            excess_gen = set(sol_gens).difference( e.gen_key   )
            if len(excess_gen) > 0:
                print_alert('\tExtra generators found in {}: {}\n'.format(file_name, excess_gen))
        else:
            excess_gen = set(sol_gens).difference( e.active_gens   )
            if len(excess_gen) > 0:
                print_alert('\tGenerators with inactive status found in {}: {}\n'.format(file_name, excess_gen))

        if len(excess_gen) == 0:
            if case == "BASECASE":
                missing_gen = set( data.raw.generators.keys()  ).difference(sol_gens)
                if len(missing_gen) > 0:
                    print_alert('\tMissing generators in {}: {}\n'.format(file_name, missing_gen))
            else:
                missing_gen = set( e.active_gens  ).difference(sol_gens)
                if len(missing_gen) > 0:
                    print_alert('\tMissing generators in {}: {}\n'.format(file_name, missing_gen))

        if  case == "CONTINGENCY":
            for inactive_gen_key in e.inactive_gens:
                gen = e.data.raw.generators[inactive_gen_key]
                k = e.gen_map[inactive_gen_key]
                inactive_gen_df = { 'i':  gen.i, 'id':gen.id, 'p':0, 'q':0, 'x':gen.stat }
                self.gen_df = Insert_row(k, self.gen_df, inactive_gen_df)

        print_alert('expecting {} lines'.format(num_line), raise_exception = False)

        self.line_df = pd.read_csv(
            file_name,
            sep=',',
            header=None,
            names=['iorig', 'idest', 'id', 'x'],
            dtype={'iorig':np.int_, 'idest':np.int_, 'id':np.str_, 'x':np.int_},
            nrows=num_line,
            engine='c',
            na_values=None,
            keep_default_na=False,
            #quoting=csv.QUOTE_NONE,
            quotechar="'",
            skiprows=row_start['line'],
            skipinitialspace=True,
            float_precision=pandas_float_precision)

        self.line_df.id = [txt.strip() for txt in self.line_df.id] #AT-0724
        sol_lines = list(zip(self.line_df.iorig, self.line_df.idest, self.line_df.id))

        print_alert('\tloaded {} lines'.format(len(set(sol_lines))), raise_exception = False)

        if case == "BASECASE":
            excess_line = set(sol_lines).difference( data.raw.nontransformer_branches.keys()   )
            if len(excess_line) > 0:
                print_alert('\tExtra lines found in {}: {}\n'.format(file_name, excess_line))
        else:
            excess_line = set(sol_lines).difference( e.active_lines   )
            if len(excess_line) > 0:
                print_alert('\tlines with inactive status found in {}: {}\n'.format(file_name, excess_line))

        if len(excess_line) == 0:
            if case == "BASECASE":
                missing_line = set( data.raw.nontransformer_branches.keys()  ).difference(sol_lines)
                if len(missing_line) > 0:
                    print_alert('\tMissing lines in {}: {}\n'.format(file_name, missing_line))
            else:
                missing_line = set( e.active_lines  ).difference(sol_lines)
                if len(missing_line) > 0:
                    print_alert('\tMissing lines in {}: {}\n'.format(file_name, missing_line))
      
        if  case == "CONTINGENCY":
            for inactive_line_key in e.inactive_lines:
                line = e.data.raw.nontransformer_branches[inactive_line_key]
                k = e.line_map[inactive_line_key]
                inactive_line_df = { 'iorig':line.i, 'idest':line.j, 'id':line.ckt, 'x':line.x}
                self.line_df = Insert_row(k, self.line_df, inactive_line_df)
      

        print_alert('expecting {} xfmrs'.format(num_xfmr), raise_exception = False)


        self.xfmr_df = pd.read_csv(
            file_name,
            sep=',',
            header=None,
            names=['iorig', 'idest', 'id', 'x', 'xst'],
            dtype={'iorig':np.int_, 'idest':np.int_, 'id':np.str_, 'x':np.int_, 'xst':np.int_},
            nrows=num_xfmr,
            engine='c',
            na_values=None,
            keep_default_na=False,
            #quoting=csv.QUOTE_NONE,
            quotechar="'",
            skiprows=row_start['xfmr'],
            skipinitialspace=True,
            float_precision=pandas_float_precision)

        self.xfmr_df.id = [txt.strip() for txt in self.xfmr_df.id] #AT-0724
        sol_xfmrs = list(zip(self.xfmr_df.iorig, self.xfmr_df.idest,[0]*len(self.xfmr_df.iorig), self.xfmr_df.id))

        print_alert('\tloaded {} xfmrs'.format(len(set(sol_xfmrs))), raise_exception = False)

        if case == "BASECASE":
            excess_xfmr = set(sol_xfmrs).difference( data.raw.transformers.keys()   )
            if len(excess_xfmr) > 0:
                print_alert('\tExtra xfmrs found in {}: {}\n'.format(file_name, excess_xfmr))
        else:
            excess_xfmr = set(sol_xfmrs).difference( e.active_xfmrs   )
            if len(excess_xfmr) > 0:
                print_alert('\txfmrs with inactive status found in {}: {}\n'.format(file_name, excess_xfmr))

        if len(excess_xfmr) == 0:
            if case == "BASECASE":
                missing_xfmr = set( data.raw.transformers.keys()  ).difference(sol_xfmrs)
                if len(missing_xfmr) > 0:
                    print_alert('\tMissing xfmrs in {}: {}\n'.format(file_name, missing_xfmr))
            else:
                missing_xfmr = set( e.active_xfmrs  ).difference(sol_xfmrs)
                if len(missing_xfmr) > 0:
                    print_alert('\tMissing xfmrs in {}: {}\n'.format(file_name, missing_xfmr))

        if  case == "CONTINGENCY":
            for inactive_xfmr_key in e.inactive_xfmrs:
                xfmr = e.data.raw.transformers[inactive_xfmr_key]
                k = e.xfmr_map[inactive_xfmr_key]
                inactive_xfmr_df = { 'iorig':xfmr.i, 'idest':xfmr.j, 'id':xfmr.ckt, 'x':xfmr.stat, 'xst':0 }    #CHALLENGE2 REVIEW
                self.xfmr_df = Insert_row(k, self.xfmr_df, inactive_xfmr_df)
      


        print_alert('expecting {} swsh'.format(num_swsh), raise_exception = False)

        names=['i', 'xst1', 'xst2', 'xst3', 'xst4', 'xst5', 'xst6', 'xst7', 'xst8']
        self.swsh_df = pd.read_csv(
            file_name,
            sep=',',
            header=None,
            names=['i', 'xst1', 'xst2', 'xst3', 'xst4', 'xst5', 'xst6', 'xst7', 'xst8'],
            dtype={  field: np.float_ for field in names },
            nrows=num_swsh,     #CHALLENGE2: #RAW SWSH - SWSH STATUS == 0
            engine='c',
            #na_values=None,
            #keep_default_na=False,
            #quoting=csv.QUOTE_NONE,
            quotechar="'",
            skiprows=row_start['swsh'],
            skipinitialspace=True,
            float_precision=pandas_float_precision)
        #CHALLENGE2 - SET MISSING XSTi TO ZERO
        self.swsh_df[names] = self.swsh_df[names].fillna(0)
        self.swsh_df[names] = self.swsh_df[names].apply(pd.to_numeric, downcast='integer')        
        
        sol_swshs = list(zip(self.swsh_df.i))

        print_alert('\tloaded {} swshs'.format(len(set(sol_swshs))), raise_exception = False)

        excess_swshs = set(sol_swshs).difference( set( data.raw.active_swsh.keys() ))
        if len(excess_swshs) > 0:
            print_alert('\tswshs with inactive status found in {}: {}\n'.format(file_name, excess_swshs))
        else:
            missing_swshs = set( data.raw.active_swsh.keys()  ).difference(sol_swshs)
            if len(missing_swshs) > 0:
                print_alert('\tMissing swshs in {}: {}\n'.format(file_name, missing_swshs))

        #CHALLENGE2 - ENSURE XSTi ARE INTS

# todo extract this to another module
# e.g. evaluate_solution.py, evaluate_solution_serial.py, evaluate_solution_mpi.py, etc.
@timeit
def run(raw_name, con_name, sup_name, solution_path=None, ctg_name=None, summary_name=None, detail_name=None, line_switching_allowed=None, xfmr_switching_allowed=None, check_contingencies=None):

    # todo - remove
    if debug:
        ctgs_so_far = 0

    # start timer
    start_time_all = time.time()

    global active_case
    global active_solution_path
    global eval_out_path
    global log_fileobject

    active_case = 'BASECASE'
    active_solution_path = solution_path
    eval_out_path = solution_path

    if(process_rank == 0):
        log_fileobject = open(f'{active_solution_path}/{active_case}.eval.log', "w")
        log_fileobject.write(f"Initiating Evaluation for {active_solution_path}...\n")

    print_info('USE_MPI: {}'.format(USE_MPI))
    print_info('check_contingencies: {}'.format(check_contingencies))
    print_info('line_switching_allowed: {}'.format(line_switching_allowed))
    print_info('xfmr_switching_allowed: {}'.format(xfmr_switching_allowed))
    print_info('hard_constr_tol: {}'.format(hard_constr_tol))
    print_info('pandas_float_precision: {}'.format(pandas_float_precision))
    print_info('stop_on_errors: {}'.format(stop_on_errors))

    if not ( os.path.exists(raw_name) and os.path.exists(con_name) and os.path.exists(sup_name)):
        print_info('Could not find input data files')
        print_info((raw_name, con_name, sup_name))
        #return (None, 1, False, {}, {})
        rval = (None, 1, False, {})
        #print('returning obj: {}, infeas: {}, sol_exist: {}'.format(rval[0], rval[1], rval[2]))
        print('obj: {}'.format(rval[0]))
        print('infeasibility: {}'.format(rval[1]))
        print('solutions exist: {}'.format(rval[2]))
        return rval
        #sys.exit()

    # read the data files
    start_time = time.time()
    p = data.Data()
    p.raw.read(raw_name)

    # todo
    # why do we read the ctg data one ctg at a time rather than all at once?
    # what is the read time for all at once? ~0.2 s
    # what is the read time for each individual one? ~0.2s
    # tested on C2S1N11152/scenario_01
    # there seems to be no time advantage to reading one at a time
    # therefore, in the serial method and in the MPI method,
    # and in the main rank and the sub ranks for the MPI method,
    # we should read all ctgs at the top of the code,
    # then select the active one in the ctg loop,
    # saving 0.2s per ctg out of a total ctg eval cost of ~0.5s
    read_all_ctgs_start_time = time.time()
    p.con.read(con_name)        #CHALLENGE2 DISCARDED - read only one contingency on the compute node
    read_all_ctgs_end_time = time.time()
    print_info("read all ctgs time: {}".format(read_all_ctgs_end_time - read_all_ctgs_start_time))
    # set it back
    #p.con = data.Con()

    p.sup.read(sup_name)
    time_elapsed = time.time() - start_time
    print_info("read data time: %u" % time_elapsed)
    
    # show data stats
    print_info("buses: %u" % len(p.raw.buses))
    print_info("loads: %u" % len(p.raw.loads))
    print_info("fixed_shunts: %u" % len(p.raw.fixed_shunts))
    print_info("generators: %u" % len(p.raw.generators))
    print_info("nontransformer_branches: %u" % len(p.raw.nontransformer_branches))
    print_info("transformers: %u" % len(p.raw.transformers))
    print_info("transformer impedance correction tables: %u" % len(p.raw.transformer_impedance_correction_tables))
    print_info("switched_shunts: %u" % len(p.raw.switched_shunts))
    print_info("contingencies: %u" % len(p.con.contingencies))
    
    print_info('done reading data')

    if solution_path is None:
        print_info('solution_path is invalid')
        return True

    # set up evaluation
    e = Evaluation()

    e.summary2['solutions_exist'] = True
    if not os.path.exists(f'{solution_path}/solution_BASECASE.txt'):
        e.summary2['solutions_exist'] = False
        print_info(f'{solution_path}/solution_BASECASE.txt could not be found')
        #return (None,  1, e.summary2['solutions_exist'], e.summary_all_cases, e.summary2)
        rval = (None,  1, e.summary2['solutions_exist'], e.summary_all_cases)
        print('returning obj: {}, infeas: {}, sol_exist: {}'.format(rval[0], rval[1], rval[2]))
        return rval

    if check_contingencies is not None:
        e.check_contingencies = check_contingencies
    if line_switching_allowed is not None:
        e.line_switching_allowed = line_switching_allowed
    if xfmr_switching_allowed is not None:
        e.xfmr_switching_allowed = xfmr_switching_allowed
    e.set_data(p)
    e.set_sol_initialize()
    e.eval_min_max_total_load_benefit()

    #print_alert('done setting evaluation data',raise_exception=False)
    print_info('done setting evaluation data')

    # initialize the base case solution
    sb = CaseSolution()
    sb.set_array_dims(e)
    sb.set_maps(e)
    sb.init_arrays()

    # initialize the contingency solution
    sc = CaseSolution()
    sc.set_array_dims(e)
    sc.set_maps(e)
    sc.init_arrays()

    # set data for base case evaluation
    e.set_data_for_base()

    # set prior solution values for base case evaluation
    e.set_prior_from_data_for_base()

    # evaluate prior power imbalance
    e.summary_written = False
    e.summary = create_new_summary()
    e.eval_prior_bus_pow()
    
    # base case solution evaluation
    case_start_time = time.time()

    e.summary_written = False
    e.summary = create_new_summary()

    # read the base case solution
    sb.set_read_dims()
    sb.read(f'{solution_path}/solution_BASECASE.txt')
    sb.set_arrays_from_dfs()

    # copy solution into evaluation
    e.set_sol(sb)
    e.round_sol()

    # and evaluate...
    e.eval_case()

    e.summary2['obj_cumulative'] += e.obj
    e.summary2['infeas_cumulative'] += e.infeas
    e.summary2['obj_all_cases']['BASECASE'] = e.obj
    e.summary2['infeas_all_cases']['BASECASE'] = e.infeas
    #e.summary_all_cases['BASECASE'] = copy.deepcopy(e.summary)

    # write summary
    if not e.summary_written:
        e.write_detail(eval_out_path, active_case, detail_json=True)
        if USE_MPI:
            #e.write_detail(eval_out_path,
            # if using MPI, write out each case as a single row in its own file as eval_detail_<case_label>.csv
            # with header row in eval_detail.csv
            # then add the case rows to eval_detail.csv after evaluation is complete
            # with open(f'{eval_out_path}/eval_detail.csv', mode='w') as detail_csv_file:
            #     detail_csv_writer = csv.writer(detail_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #     detail_csv_writer.writerow(['case_label'] + flatten_summary(e.summary)['keys'])
            # with open(f'{eval_out_path}/eval_detail_{active_case}.csv', mode='w') as detail_csv_file: # write
            #     detail_csv_writer = csv.writer(detail_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #     detail_csv_writer.writerow([active_case] + flatten_summary(e.summary)['values'])
            # todo fix this
            pass
        else:
            #e.write_detail(eval_out_path, active_case, detail_csv=True)
            pass
        e.summary_written = True

    case_end_time = time.time()
    print_info(
        "done evaluating case. label: {}, time: {}".format(
            'BASECASE', case_end_time - case_start_time))

    #CHALLENGE2 - return here if solution1 validation is requested
    if not e.check_contingencies:
        rval = (e.summary2['obj_cumulative'],  1 if e.summary2['infeas_cumulative'] else 0, e.summary2['solutions_exist'], e.summary_all_cases)
        print('returning obj: {}, infeas: {}, sol_exist: {}'.format(rval[0], rval[1], rval[2]))
        #return (e.summary2['obj_cumulative'],  1 if e.summary2['infeas_cumulative'] else 0, e.summary2['solutions_exist'], e.summary_all_cases, e.summary2)
        #return (e.summary2['obj_cumulative'],  1 if e.summary2['infeas_cumulative'] else 0, e.summary2['solutions_exist'], e.summary_all_cases)
        return rval

    #if ctg_name is None:
    #    return True

    # set data for contingency evaluation
    e.set_data_for_ctgs()
 
    # before overwriting base case solution, save some part of it as prior solution
    e.set_prior_from_base_for_ctgs()


    # TODO - extract this to a function and add a verification that we have not just the right number of contingencies but also the right contingency label values
    #CHALLENGE2 - validate solution2 file count
    print_info('solutions path: {}'.format(solution_path))
    sol_ctg_path = f"{solution_path}/solution_*.txt"
    solution_files = glob.glob( str( sol_ctg_path)  )
    solution2_files = [ solution_file for solution_file in solution_files if "BASECASE" not in solution_file]

    # check base case solution file exists
    base_case_solution_files = [solution_file for solution_file in solution_files if "BASECASE" in solution_file]
    print_alert(
        'Expected 1 BASECASE solution file, Encountered {} BASECASE solution files, files found: {}'.format(
            len(base_case_solution_files), base_case_solution_files),
        check_passed=(len(base_case_solution_files) == 1))
    if len(base_case_solution_files) != 1:
        e.summary2['solutions_exist'] = False
    
    #self.ctg_label#?????
    #print(solution2_files)
    ctg_labels_in_sol = [
        Path(solution_file).resolve().stem.replace("solution_","")
        for solution_file in solution2_files]
    ctg_labels_in_con = e.ctg_label

    # check contingency solution files exist
    ctg_labels_in_con_not_in_sol = sorted(list(set(ctg_labels_in_con).difference(set(ctg_labels_in_sol))))
    print_alert(
        'Expected 0 contingencies in CON not in solution_*.txt, found {}: {}'.format(
            len(ctg_labels_in_con_not_in_sol), ctg_labels_in_con_not_in_sol),
        check_passed=(len(ctg_labels_in_con_not_in_sol) == 0))
    if len(ctg_labels_in_con_not_in_sol) > 0:
        e.summary2['solutions_exist'] = False

    # check no extra solution files
    # do not need to return infeasible if extra solution files. it is just strange
    ctg_labels_in_sol_not_in_con = sorted(list(set(ctg_labels_in_sol).difference(set(ctg_labels_in_con))))
    print_alert(
        'Expected 0 contingencies in solution_*.txt not in CON, found {}: {}'.format(
            len(ctg_labels_in_sol_not_in_con), ctg_labels_in_sol_not_in_con),
        check_passed=(len(ctg_labels_in_sol_not_in_con) == 0))
    #if len(ctg_labels_in_sol_not_in_con) > 0:
    #    e.summary2['solutions_exist'] = False

    #with open(str(con_name)) as f:
    #   expected = sum('CONTINGENCY' in line for line in f)

    found =  len(solution2_files) 
    try:
        if found == e.num_ctg:
            print_info(f'Expected #contingencies: {e.num_ctg}, Encountered #contingencies: {found}')
        else:
            #print_alert(f'Expected #contingencies: {expected}, Encountered #contingencies: {found}', check_passed=(expected==found))
            print_alert(f'Expected #contingencies: {e.num_ctg}, Encountered #contingencies: {found}', check_passed=(found == e.num_ctg))
    except:
        pass

    if e.summary2['solutions_exist'] == False:
        if process_rank == 0:
            print_info(f'Some solution files are missing. Exiting...')
        #return (None, 1, e.summary2['solutions_exist'], e.summary_all_cases, e.summary2)   
        rval = (None, 1, e.summary2['solutions_exist'], e.summary_all_cases)   
        print('returning obj: {}, infeas: {}, sol_exist: {}'.format(rval[0], rval[1], rval[2]))
        return rval

    if log_fileobject is not None:
       log_fileobject.close()

    # now we can assume we have a solution file for every contingency in the CON file
    # but there might be solution files for nonexistent contingencies
    # get the files with labels corresponding to real contingencies
    label_to_sol_file_dict = {
        Path(solution_file).resolve().stem.replace("solution_",""):solution_file
        for solution_file in solution2_files}
    sol_files_with_label_in_con = [
        label_to_sol_file_dict[label]
        for label in ctg_labels_in_con]

    #start_time = time.time()

    #CHALLENGE2 - Contingencies will be spread across provided processors

    # use only the solution files corresponding to real contingencies
    solution2_files = sol_files_with_label_in_con

    contingency_labels = [Path(solution_file).resolve().stem.replace("solution_","")  for solution_file in solution2_files if not 'BASECASE' in solution_file]   
    contingency_list = { p[0]: { 'label':p[0], 'path':p[1]} for p in zip(contingency_labels, solution2_files)  }
    contingency_results = { contingency:None for contingency in contingency_list.keys()   }

    #def check_time(last_time, counter):
    #    current_time = time.time()
    #    counter += 1
    #    #print_info('time check. counter: {}, time: {}'.format(counter, current_time - last_time))
    #    last_time = current_time

    if not USE_MPI:
        for active_case in contingency_labels:
            # todo
            # close the log file object right here
            # then open a new one for this active case
            # do not close or open it again until the next time through the loop

            case_start_time = time.time()

            # todo - remove
            if debug:
                ctgs_so_far += 1
                if ctgs_so_far > stop_after_ctgs:
                    break

            e.summary_written = False
            e.summary = create_new_summary()
            print_info('processing contingency {}'.format(active_case))
            try:
                
                print_info(f'writing {active_case} log to {active_solution_path}')
                log_fileobject = open( f'{active_solution_path}/{active_case}.eval.log', "w")
                log_fileobject.write(f"Initiating Evaluation for {active_solution_path}...")

                #read_ctg_start_time = time.time()
                #ctg_name = contingency_list[active_case]['path']
                #p.con = data.Con()
                #p.con.read(con_name, active_case) 
                #read_ctg_end_time = time.time()
                #print_info('read contingency time: {}'.format(read_ctg_end_time - read_ctg_start_time))
                #e.set_data_ctg_params(e.data)

                e.ctg_current = e.ctg_map[clean_string(active_case)]
                print_info(
                    'current contingency. active_case: {}, index: {}, label: {}, gens_out: {}, lines_out: {}, xfmrs_out: {}'.format(
                        active_case, e.ctg_current, e.ctg_label[e.ctg_current],
                        [e.gen_key[i] for i in e.ctg_gens_out[e.ctg_current]],
                        [e.line_key[i] for i in e.ctg_lines_out[e.ctg_current]],
                        [e.xfmr_key[i] for i in e.ctg_xfmrs_out[e.ctg_current]]))
                e.set_service_status_for_ctg()

                # read the contingency solution
                sc.set_ctg(e)
                sc.set_read_dims()
                sc.read(contingency_list[active_case]['path'])
                sc.set_arrays_from_dfs()

                # copy solution into evaluation
                e.set_sol(sc)
                e.round_sol()

                # and evaluate...
                e.eval_case()

                #if e.infeas == 1:
                #    raise Exception(f'Infeasibility detected in {active_case}')
                
                e.summary2['obj_cumulative'] += (e.obj / e.num_ctg)
                e.summary2['infeas_cumulative'] += e.infeas
                e.summary2['obj_all_cases'][clean_string(active_case)] = e.obj
                e.summary2['infeas_all_cases'][clean_string(active_case)] = e.infeas
                #e.summary_all_cases[clean_string(active_case)] = copy.deepcopy(e.summary)
            except:
                e.infeas = 1
                e.summary2['infeas_cumulative'] += e.infeas
                e.summary2['infeas_all_cases'][clean_string(active_case)] = e.infeas
                traceback.print_exc()

            # write summary
            if not e.summary_written:
                e.write_detail(eval_out_path, active_case, detail_json=True)
                #e.write_detail(eval_out_path, active_case, detail_csv=True)
                e.summary_written = True

            case_end_time = time.time()
            print_info(
                "done evaluating case. label: {}, time: {}".format(
                    clean_string(active_case), case_end_time - case_start_time))

                             
            log_fileobject.close()

    if USE_MPI:
        if process_rank == 0:
            
            process_list = { i:None  for i in range(1,process_count)   }

            while True:
                for contingency_label, contingency_result in contingency_results.items():
                    #print(f'checking {contingency_label}...')
                    if contingency_label not in process_list.values() and  contingency_result is None:
                        #print(f' {contingency_label} is pending...{process_list}')
                        for process_id,process_ctg_label in process_list.items():
                            if process_ctg_label is None:
                                print(f' {process_id} is available for {contingency_label}...')
                                process_list [ process_id ] = contingency_label


                                print('sending contingency {} to {}'.format(contingency_label, process_id))
                                comm.isend(contingency_list[contingency_label], dest = process_id)
                                break
        
                if comm.iprobe(source = MPI.ANY_SOURCE):
                    status = MPI.Status()
                    contingency_result = comm.recv( source = MPI.ANY_SOURCE, status = status)
                    source = status.Get_source()
                    #print(f'process list before receiving from {source}: {process_list}, {contingency_results}')
                    process_list[source] = None
                    #print(f'process list after receiving from {source}: {process_list}, {contingency_results}')
                    contingency_results[ contingency_result['contingency'] ] = contingency_result
                    e_ctg_obj = contingency_result['e_ctg_obj']

                    if(e_ctg_obj != None):
                        e.summary2['obj_cumulative'] += (e_ctg_obj / e.num_ctg )
                        e.summary2['obj_all_cases'][clean_string(active_case)] = e_ctg_obj
                        e.summary2['infeas_all_cases'][clean_string(active_case)] = 0
                        #e.summary_all_cases[clean_string(active_case)] = e_ctg_summary
                    else:
                        e.infeas = 1
                        print_info('Infeasibility encountered for {}, exiting...'.format(contingency_result['contingency'])) 

                    ncompleted = sum( contingency_result != None for contingency_result in contingency_results.values())
                    print('received result from process:{}-{}; contingencies expected: {}, contingencies complete: {}'.format( source,contingency_result['contingency'], e.num_ctg, ncompleted))
                    if  ncompleted >= e.num_ctg:
                        print(f'All contingencies completed, exiting...')
                        [ comm.send(None,node) for node in range(1,process_count)]
                        comm.Barrier()
                        break  
                    if  e.infeas == 1:
                        print('Infeasibility detected in {}, exiting...'.format(contingency_result['contingency']))
                        [ comm.send(None,node) for node in range(1,process_count)]
                        comm.Barrier()

                        break

        else:
            contingency = comm.recv(source=0)
            print('received')
            while (contingency != None):
                active_case = contingency['label']

                print('[{}] - received contingency {}'.format(process_rank, active_case))
                try:
                    ctg_name = contingency['path']

                    log_fileobject = open( f'{active_solution_path}/{active_case}.eval.log', "w")
                    log_fileobject.write("Initiating Evaluation...")

                    e.ctg_current = e.ctg_map[clean_string(active_case)]
                    e.set_service_status_for_ctg()

                    # read the contingency solution
                    sc.set_ctg(e)
                    sc.set_read_dims()
                    sc.read(contingency_list[active_case]['path'])
                    sc.set_arrays_from_dfs()

                    # copy solution into evaluation
                    e.set_sol(sc)
                    e.round_sol()

                    # and evaluate...
                    e.eval_case()

                    log_fileobject.close()

                    #if e.infeas == 1:
                    #    raise Exception(f'Infeasibility dectected in {active_case}')

                    # TODO this causes an error. not sure why
                    #print('trying to write summary files eval_out_path: {}, active_case: {}'.format(eval_out_path, active_case))
                    #with open(eval_out_path + '/' + active_case + '.tmp', mode='w') as tmp_out_file:
                    #    tmp_out_file.write('test')
                    e.write_detail(eval_out_path, active_case, detail_json=True)
                    #self.write_detail(eval_out_path, active_case, detail_csv=True)

                except:
                    e.infeas = 1
                    traceback.print_exc()

                contingency_result = { 'contingency': active_case, 'e_ctg_obj': e.summary['obj']['val'] if e.infeas != 1 else None    }
                print(f'process {process_rank} sending contingency result for {active_case}')
                comm.isend(contingency_result, dest=0)

                print(f'process {process_rank} waiting to receive contingency')
                contingency = comm.recv(source=0)
            comm.Barrier()

    if process_rank == 0:
        end_time_all = time.time()
        print_info('obj all cases:')
        print_info(e.summary2['obj_all_cases'])
        print_info('infeas all cases:')
        print_info(e.summary2['infeas_all_cases'])
        print_info('obj cumulative: {}'.format(e.summary2['obj_cumulative']))
        print_info('infeas cumulative: {}'.format(e.summary2['infeas_cumulative']))
        print_info('eval time: {}'.format(end_time_all - start_time_all))
    
        # todo : short circuit if infeas

        e.write_final_summary_and_detail(eval_out_path)
        
        print_info("obj: {}".format(e.summary2['obj_cumulative']))
        print_info("infeas: {}".format(e.summary2['infeas_cumulative']))

    rval = (e.summary2['obj_cumulative'],  1 if e.summary2['infeas_cumulative'] else 0, e.summary2['solutions_exist'], e.summary_all_cases)
    print('returning obj: {}, infeas: {}, sol_exist: {}'.format(rval[0], rval[1], rval[2]))
    #return (e.summary2['obj_cumulative'],  1 if e.summary2['infeas_cumulative'] else 0, e.summary2['solutions_exist'], e.summary_all_cases, e.summary2)
    #return (e.summary2['obj_cumulative'],  1 if e.summary2['infeas_cumulative'] else 0, e.summary2['solutions_exist'], e.summary_all_cases)
    return rval


def run_main(data_basepath, solution_basepath, line_switching_allowed=None, xfmr_switching_allowed=None, check_contingencies=None):

    global active_case
    global active_solution_path
    global eval_out_path
    global USE_MPI

    active_case = 'BASECASE'
    active_solution_path = solution_basepath
    eval_out_path = solution_basepath
    print_info('writing logs to {}' .format(active_solution_path))
    print_info('writing detailed and summary evaluation output to {}' .format(eval_out_path))

    raw_name = f'{data_basepath}/case.scrubbed.raw' 

    if not os.path.exists(raw_name):
        raw_name = f'{data_basepath}/case.raw' 

    con_name = f"{data_basepath}/case.scrubbed.con"

    if not os.path.exists(con_name):
        con_name = f'{data_basepath}/case.con' 

    sup_name = f"{data_basepath}/case.scrubbed.json"

    if not os.path.exists(sup_name):
        sup_name = f'{data_basepath}/case.json' 


    print_info(f'raw: {raw_name}')
    print_info(f'con: {con_name}')
    print_info(f'sup: {sup_name}')
    

    base_name = f"{solution_basepath}/solution_BASECASE.txt"
    ctg_name = ""
    summary_name = f"{solution_basepath}/summary.csv"
    detail_name = f"{solution_basepath}/detail.csv"

    print_info(f'Setting data path to {data_basepath}')
    print_info(f'Setting solution path to {solution_basepath}')

    try:
        return run(raw_name, con_name, sup_name,solution_basepath, ctg_name, summary_name, detail_name, line_switching_allowed, xfmr_switching_allowed, check_contingencies)
    except:
        var = traceback.format_exc()
        traceback.print_exc()
        print_alert(var, raise_exception=False)
        rval = (None, 1, False, {})
        print('returning obj: {}, infeas: {}, sol_exist: {}'.format(rval[0], rval[1], rval[2]))
        return rval

def main():
    #arg_basepath = Path( '/pic/projects/goc/loadbalancing/src/challenge2-eval-repo/data/Terrence/sandbox')
    #arg_basepath = Path( '/pic/projects/goc/loadbalancing/src/challenge2-eval-repo/data/goc-datasets-c2/GaTech/14_bus')
    #arg_basepath = Path( '../data/goc-datasets-c2/GaTech/Network_06O-124-tgo2000_20190828/scenario_101')
    #arg_basepath = Path( '/pic/dtn/go/UWMAD_GO2/Sandbox072020' )

    arg_division = '1'
    arg_basepath = Path('/pic/projects/goc/loadbalancing/src/challenge2-eval-repo/C2Eval/data/ieee14/scenario_1')
    arg_datapath = arg_basepath


    if len(sys.argv) > 3  and not sys.argv[-1].endswith('.py'):
        arg_division = sys.argv[-3]
        arg_datapath = Path(sys.argv[-2])
        arg_basepath = Path(sys.argv[-1])

    if (arg_division == '3' or arg_division == '4'):
        line_switching_allowed = True
        xfmr_switching_allowed = True
    elif (arg_division == '1' or arg_division == '2'):
        line_switching_allowed = False
        xfmr_switching_allowed = False
    else:
        line_switching_allowed = None
        xfmr_switching_allowed = None
    #line_switching_allowed = True if arg_division == '3' or arg_division == '4' else None
    #xfmr_switching_allowed = True if arg_division == '3' or arg_division == '4' else None
    check_contingencies = True # set to False to check just the base case

    run_main(arg_datapath, arg_basepath, line_switching_allowed, xfmr_switching_allowed, check_contingencies)

if __name__ == "__main__":
    main()


#C:/Arun/Grid/GOC/Challenge2/Evaluation/C2Eval/py> mpiexec /np 4 python ./evaluation.py
#~/gocomp/C2Eval/py> mpirun -np 2 python evaluation.py [<division>] [<datapath>] [<solutionpath>]
