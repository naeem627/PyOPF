'''
infeasibility_solution.py
main code to construct the infeasibility solution

TODO:
check stat - done
add sol2 constructor - done
use real values from prior operating point - done
performance
log output
project transformer settings onto integer feasible set
project switched shunt settings onto integer feasible set
'''

# built in imports
import csv
import io
import math
import pickle
import time

import numpy as np
import pandas as pd

# note: really just using python 3 at this point
try:
    from StringIO import StringIO ## for Python 2
except ImportError:
    from io import StringIO ## for Python 3

# GOComp modules - this should be visible on the GOComp evaluation system
try:
    import data_utilities.data as data
    from data_utilities.swsh_utils import solve_py as swsh_solve
    from data_utilities.xfmr_utils import compute_xfmr_position
except:
    import data
    from swsh_utils import solve_py as swsh_solve
    from xfmr_utils import compute_xfmr_position

swsh_binit_tol = 1e-4

# modules for this code
#sys.path.append(os.path.normpath('.')) # better way to make this visible?
#import something

# todo
# speed up
#   maybe just maintain the sol2 dict and add/remove items as needed
#   maintain sol1/sol2 in numpy arrays, update as needed, write to string, compile strings and write to file
# project tap onto bounds/steps - done, need to document this in the formulation
# project swsh b onto bounds/steps

# def compute_swsh_steps(r):
#     # todo
#     steps = r.swsh_susc_count * [0]
#     return steps

def compute_swsh_xst(h_b0, ha_n, ha_b):

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

def csv2string(data):
    si = StringIO.StringIO()
    cw = csv.writer(si)
    cw.writerows(data)
    return si.getvalue()

sol_prefix = 'solution_'
sol_suffix = '.txt'
base_case_label = 'BASECASE'
section_names = [
    'buses',
    'loads',
    'generators',
    'lines',
    'transformers',
    'switched shunts']
section_start_lines = {
    'buses': '--bus section',
    'loads': '--load section',
    'generators': '--generator section',
    'lines': '--line section',
    'transformers': '--transformer section',
    'switched shunts': '--switched shunt section'}
section_field_names = {
    'buses': ['i', 'v', 'theta'],
    'loads': ['i', 'id', 't'],
    'generators': ['i', 'id', 'p', 'q', 'x'],
    'lines': ['iorig', 'idest', 'id', 'x'],
    'transformers': ['iorig', 'idest', 'id', 'x', 'xst'],
    'switched shunts': ['i', 'xst1', 'xst2', 'xst3', 'xst4', 'xst5', 'xst6', 'xst7', 'xst8']}
int_type_str = 'int'
float_type_str = 'float'
str_type_str = '<U2'

class Solver():

    def __init__(self):

        self.data = data.Data()

    def construct_arrays(self):

        start_time = time.time()
        self.construct_bus_arrays()
        self.construct_load_arrays()
        self.construct_gen_arrays()
        self.construct_line_arrays()
        self.construct_xfmr_arrays()
        self.construct_swsh_arrays()
        end_time = time.time()
        print('construct_arrays time: {}'.format(end_time - start_time))

    def construct_bus_arrays(self):

        self.bus = [r for r in self.data.raw.buses.values()]
        self.num_bus = len(self.bus)
        self.bus_i = np.array([r.i for r in self.bus], dtype=int)
        self.bus_vm = np.array([r.vm for r in self.bus])
        self.bus_va = np.array([r.va for r in self.bus])
        self.bus_nvhi = np.array([r.nvhi for r in self.bus])
        self.bus_nvlo = np.array([r.nvlo for r in self.bus])
        self.bus_evhi = np.array([r.evhi for r in self.bus])
        self.bus_evlo = np.array([r.evlo for r in self.bus])
        #self.bus_sol1 = np.zeros(shape=(self.num_bus,), dtype=[('i', int_type_str), ('v', float_type_str), ('theta', float_type_str)])
        self.bus_sol1 = pd.DataFrame(columns=['i', 'v', 'theta'])
        self.bus_sol1['i'] = self.bus_i
        self.bus_sol1['v'] = np.minimum(np.maximum(self.bus_vm, self.bus_nvlo), self.bus_nvhi)
        self.bus_sol1['theta'] = self.bus_va * np.pi / 180.0
        #self.bus_sol2 = np.zeros(shape=(self.num_bus,), dtype=[('i', int_type_str), ('v', float_type_str), ('theta', float_type_str)])
        self.bus_sol2 = pd.DataFrame(columns=['i', 'v', 'theta'])
        self.bus_sol2['i'] = self.bus_i
        self.bus_sol2['v'] = np.minimum(np.maximum(self.bus_vm, self.bus_evlo), self.bus_evhi)
        self.bus_sol2['theta'] = self.bus_va * np.pi / 180.0

    def construct_load_arrays(self):

        self.load = [r for r in self.data.raw.loads.values() if r.status == 1]
        self.num_load = len(self.load)
        self.load_i = np.array([r.i for r in self.load], dtype=int)
        self.load_id = np.array([r.id for r in self.load], dtype=str)
        load_key = [(self.load_i[i], self.load_id[i]) for i in range(self.num_load)]
        load_map = {load_key[i]:i for i in range(self.num_load)}
        sup_load = self.data.sup.get_loads()
        sup_load_key_map = {(r['bus'], r['id']):r for r in sup_load}
        sup_load_key = sup_load_key_map.keys()
        sup_load_key = list(set(sup_load_key).intersection(set(load_key)))
        sup_load = [sup_load_key_map[k] for k in sup_load_key]
        sup_load_index = [load_map[r['bus'], r['id']] for r in sup_load]
        self.load_tmin = np.zeros(shape=(self.num_load,))
        self.load_tmin[sup_load_index] = np.array([r['tmin'] for r in sup_load])
        self.load_tmax = np.zeros(shape=(self.num_load,))
        self.load_tmax[sup_load_index] = np.array([r['tmax'] for r in sup_load])
        #self.load_sol = np.zeros(shape=(self.num_load,), dtype=[('i', int_type_str), ('id', str_type_str), ('t', float_type_str)])
        self.load_sol = pd.DataFrame(columns=['i', 'id', 't'])
        self.load_sol['i'] = self.load_i
        self.load_sol['id'] = self.load_id
        self.load_sol['t'] = np.minimum(np.maximum(1.0, self.load_tmin), self.load_tmax)

    def construct_gen_arrays(self):

        self.gen = [r for r in self.data.raw.generators.values()]
        self.num_gen = len(self.gen)
        self.gen_i = np.array([r.i for r in self.gen], dtype=int)
        self.gen_id = np.array([r.id for r in self.gen], dtype=str)
        self.gen_pg = np.array([r.pg for r in self.gen])
        self.gen_qg = np.array([r.qg for r in self.gen])
        self.gen_pt = np.array([r.pt for r in self.gen])
        self.gen_pb = np.array([r.pb for r in self.gen])
        self.gen_qt = np.array([r.qt for r in self.gen])
        self.gen_qb = np.array([r.qb for r in self.gen])
        self.gen_stat = np.array([r.stat for r in self.gen])
        #self.gen_sol = np.zeros(shape=(self.num_gen,), dtype=[('i', int_type_str), ('id', str_type_str), ('p', float_type_str), ('q', float_type_str), ('x', int_type_str)])
        self.gen_sol = pd.DataFrame(columns=['i', 'id', 'p', 'q', 'x'])
        self.gen_sol['i'] = self.gen_i
        self.gen_sol['id'] = self.gen_id
        self.gen_sol['p'] = self.gen_stat * np.minimum(np.maximum(self.gen_pg, self.gen_pb), self.gen_pt) / self.data.raw.case_identification.sbase
        self.gen_sol['q'] = self.gen_stat * np.minimum(np.maximum(self.gen_qg, self.gen_qb), self.gen_qt) / self.data.raw.case_identification.sbase
        self.gen_sol['x'] = self.gen_stat

    def construct_line_arrays(self):

        self.line = [r for r in self.data.raw.nontransformer_branches.values()]
        self.num_line = len(self.line)
        self.line_i = np.array([r.i for r in self.line], dtype=int)
        self.line_j = np.array([r.j for r in self.line], dtype=int)
        self.line_ckt = np.array([r.ckt for r in self.line], dtype=str)
        self.line_st = np.array([r.st for r in self.line], dtype=int)
        #self.line_sol = np.zeros(shape=(self.num_line,), dtype=[('iorig', int_type_str), ('idest', int_type_str), ('id', str_type_str), ('x', int_type_str)])
        self.line_sol = pd.DataFrame(columns=['iorig', 'idest', 'id', 'x'])
        self.line_sol['iorig'] = self.line_i
        self.line_sol['idest'] = self.line_j
        self.line_sol['id'] = self.line_ckt
        self.line_sol['x'] = self.line_st

    def construct_xfmr_arrays(self):

        self.xfmr = [r for r in self.data.raw.transformers.values()]
        self.num_xfmr = len(self.xfmr)
        self.xfmr_i = np.array([r.i for r in self.xfmr], dtype=int)
        self.xfmr_j = np.array([r.j for r in self.xfmr], dtype=int)
        self.xfmr_ckt = np.array([r.ckt for r in self.xfmr], dtype=str)
        self.xfmr_stat = np.array([r.stat for r in self.xfmr], dtype=int)
        self.xfmr_cod1 = np.array([r.cod1 for r in self.xfmr], dtype=int)
        self.xfmr_rma1 = np.array([r.rma1 for r in self.xfmr])
        self.xfmr_rmi1 = np.array([r.rmi1 for r in self.xfmr])
        self.xfmr_ntp1 = np.array([r.ntp1 for r in self.xfmr], dtype=int)
        self.xfmr_windv1 = np.array([r.windv1 for r in self.xfmr])
        self.xfmr_windv2 = np.array([r.windv1 for r in self.xfmr])
        self.xfmr_ang1 = np.array([r.ang1 for r in self.xfmr])
        #self.xfmr_sol = np.zeros(shape=(self.num_xfmr,), dtype=[('iorig', int_type_str), ('idest', int_type_str), ('id', str_type_str), ('x', int_type_str), ('xst', int_type_str)])
        self.xfmr_sol = pd.DataFrame(columns=['iorig', 'idest', 'id', 'x', 'xst'])
        self.xfmr_sol['iorig'] = self.xfmr_i
        self.xfmr_sol['idest'] = self.xfmr_j
        self.xfmr_sol['id'] = self.xfmr_ckt
        self.xfmr_sol['x'] = self.xfmr_stat
        self.xfmr_sol['xst'] = np.array([compute_xfmr_position(r)[0] for r in self.xfmr], dtype=int) # todo - numpy version

    def construct_swsh_arrays(self):

        self.swsh = [r for r in self.data.raw.switched_shunts.values() if r.stat == 1]
        self.num_swsh = len(self.swsh)
        self.swsh_i = np.array([r.i for r in self.swsh], dtype=int)
        self.swsh_binit = np.array([r.binit for r in self.swsh])
        self.swsh_n = np.array([[r.n1, r.n2, r.n3, r.n4, r.n5, r.n6, r.n7, r.n8] for r in self.swsh], dtype=int)
        self.swsh_b = np.array([[r.b1, r.b2, r.b3, r.b4, r.b5, r.b6, r.b7, r.b8] for r in self.swsh])
        if self.num_swsh == 0:
            self.swsh_n.shape = (0, 8)
            self.swsh_b.shape = (0, 8)
        self.swsh_num_block = np.array([r.swsh_susc_count for r in self.swsh], dtype=int)
        #print('debugging')
        #print('num_swsh: {}'.format(self.num_swsh))
        #print('swsh_i: {}, {}'.format(self.swsh_i, self.swsh_i.shape))
        #print('swsh_binit: {}, {}'.format(self.swsh_binit, self.swsh_binit.shape))
        #print('swsh_n: {}, {}'.format(self.swsh_n, self.swsh_n.shape))
        #print('swsh_b: {}, {}'.format(self.swsh_b, self.swsh_b.shape))
        #print('swsh_num_block: {}, {}'.format(self.swsh_num_block, self.swsh_num_block.shape))
        self.swsh_xst = compute_swsh_xst(self.swsh_binit, self.swsh_n, self.swsh_b)
        #self.swsh_sol = np.zeros(shape=(self.num_swsh,), dtype=[('i', int_type_str), ('xst1', int_type_str), ('xst2', int_type_str), ('xst3', int_type_str), ('xst4', int_type_str), ('xst5', int_type_str), ('xst6', int_type_str), ('xst7', int_type_str), ('xst8', int_type_str)])
        self.swsh_sol = pd.DataFrame(columns=['i', 'xst1', 'xst2', 'xst3', 'xst4', 'xst5', 'xst6', 'xst7', 'xst8'])
        self.swsh_sol['i'] = self.swsh_i
        self.swsh_sol['xst1'] = self.swsh_xst[:,0]
        self.swsh_sol['xst2'] = self.swsh_xst[:,1]
        self.swsh_sol['xst3'] = self.swsh_xst[:,2]
        self.swsh_sol['xst4'] = self.swsh_xst[:,3]
        self.swsh_sol['xst5'] = self.swsh_xst[:,4]
        self.swsh_sol['xst6'] = self.swsh_xst[:,5]
        self.swsh_sol['xst7'] = self.swsh_xst[:,6]
        self.swsh_sol['xst8'] = self.swsh_xst[:,7]
        # todo what to do about ragged edge? mask?

    def construct_sol1_bus(self):

        start_time = time.time()
        self.sol1_bus = {
            r.i: [
                r.i, # i
                min(max(r.vm, r.nvlo), r.nvhi), # v
                r.va * math.pi / 180.0] # theta
            for r in self.data.raw.buses.values()}
        end_time = time.time()
        print('construct_sol1_bus time: %f' % (end_time - start_time))

    def construct_sol2_bus(self):

        start_time = time.time()
        self.sol2_bus = {
            r.i: [
                r.i, # i
                min(max(r.vm, r.evlo), r.evhi), # v
                r.va * math.pi / 180.0] # theta
            for r in self.data.raw.buses.values()}
        end_time = time.time()
        print('construct_sol2_bus time: %f' % (end_time - start_time))

    def construct_sol1(self):

        start_time = time.time()
        self.sol1 = {}
        self.sol1['buses'] = self.sol1_bus
        self.sol1['loads'] = {
            (r['bus'], r['id']): [
                r['bus'], # i
                r['id'], # id
                min(max(1.0, r['tmin']), r['tmax'])] # t
            for r in self.data.sup.get_loads()
            if self.data.raw.loads[(r['bus'], r['id'])].status == 1}
        self.sol1['generators'] = {
            (r.i, r.id): [
                r.i, # i
                r.id, # id
                (min(max(r.pg, r.pb), r.pt) / self.data.raw.case_identification.sbase)
                if r.stat == 1 else 0.0, # p
                (min(max(r.qg, r.qb), r.qt) / self.data.raw.case_identification.sbase)
                if r.stat == 1 else 0.0, # q
                r.stat] # x
            for r in self.data.raw.generators.values()}
        self.sol1['lines'] = {
            (r.i, r.j, r.ckt): [
                r.i, # iorig
                r.j, # idest
                r.ckt, # id
                r.st] # x
            for r in self.data.raw.nontransformer_branches.values()}
        self.sol1['transformers'] = {
            (r.i, r.j, r.ckt): [
                r.i, # iorig
                r.j, # idest
                r.ckt, # id
                r.stat, # x
                #0] # xst
                compute_xfmr_position(r)[0]] # xst
            for r in self.data.raw.transformers.values()}
        self.sol1['switched shunts'] = {
            self.swsh_i[i]: (
                [self.swsh_i[i]] + #i
                self.swsh_xst[i, 0:self.swsh[i].swsh_susc_count].flatten().tolist()) # xst1, xst2 ... swsh_susc_count
            for i in range(self.num_swsh)}
        # self.sol1['switched shunts'] = {
        #     r.i: (
        #         [r.i] + # i
        #         compute_swsh_steps(r)) # xst1, xst2 ... swsh_susc_count
        #     for r in self.data.raw.switched_shunts.values()
        #     if r.stat == 1}
        end_time = time.time()
        print('construct_sol1 time: %f' % (end_time - start_time))

    def construct_sol2(self, contingency):

        start_time = time.time()
        generators_out = set([(e.i, e.id) for e in contingency.generator_out_events])
        generators_in = list(set(
            self.data.raw.generators.keys()).difference(generators_out))
        branches_out = set([(e.i, e.j, e.ckt) for e in contingency.branch_out_events])
        lines_out = branches_out
        transformers_out = set([(b[0], b[1], b[2]) for b in branches_out]) # todo do we need this line now?
        lines_in = list(set(
            self.data.raw.nontransformer_branches.keys()).difference(lines_out))
        transformers_in = list(set(
            self.data.raw.transformers.keys()).difference(transformers_out))
        self.sol2 = {
            'buses': self.sol2_bus,
            'loads': self.sol1['loads'],
            'generators': {
                r: self.sol1['generators'][r]
                for r in generators_in},
            'lines': {
                r: self.sol1['lines'][r]
                for r in lines_in},
            'transformers': {
                r: self.sol1['transformers'][r]
                for r in transformers_in},
            'switched shunts': self.sol1['switched shunts']}
        end_time = time.time()
        print('construct_sol2 time: %f' % (end_time - start_time))

    def write_sol_bin(self, sol, filename):
        
        with open(filename, 'wb') as f:
            pickle.dump(sol, f)

    def read_sol_bin(self, filename):

        with open(filename, 'rb') as f:
            sol = pickle.load(f)
        return sol

    def write_sol(self, sol_dir):

        start_time = time.time()
        #print(self.data)
        self.construct_arrays()
        self.construct_sol1_bus()
        self.construct_sol2_bus()
        self.construct_sol1()
        filename = sol_dir + '/' + sol_prefix + base_case_label + sol_suffix
        self.write_sol_case(self.sol1, filename)
        for k in self.data.con.contingencies.values():
            self.construct_sol2(k)
            filename = sol_dir + '/' + sol_prefix + k.label + sol_suffix
            self.write_sol_case(self.sol2, filename)
        end_time = time.time()
        print('write_sol time: %f' % (end_time - start_time))
        
    def write_sol1(self, sol_dir, saved_sol_file_name=None):

        start_time = time.time()
        self.construct_arrays()
        self.construct_sol1_bus()
        #self.construct_sol2_bus()
        self.construct_sol1()
        filename = sol_dir + '/' + sol_prefix + base_case_label + sol_suffix
        self.write_sol_case(self.sol1, filename)
        if saved_sol_file_name is not None:
            self.write_sol_bin(self.sol1, saved_sol_file_name)
        end_time = time.time()
        print('write_sol1 time: %f' % (end_time - start_time))
        
    def write_sol2(self, sol_dir, saved_sol_file_name=None):

        start_time = time.time()
        #self.construct_arrays()
        self.construct_sol2_bus()
        if saved_sol_file_name is not None:
            self.sol1 = self.read_sol_bin(saved_sol_file_name)
        else:
            self.construct_arrays()
            self.construct_sol1()
        for k in self.data.con.contingencies.values():
            self.construct_sol2(k)
            filename = sol_dir + '/' + sol_prefix + k.label + sol_suffix
            self.write_sol_case(self.sol2, filename)
        end_time = time.time()
        print('write_sol2 time: %f' % (end_time - start_time))
        
    def write_sol_case(self, sol, filename):

        start_time = time.time()

        # sio = io.StringIO()
        # self.bus_sol1.to_csv(path_or_buf=sio, index=False) # pd
        # self.load_sol.to_csv(path_or_buf=sio, index=False) # pd
        # self.gen_sol.to_csv(path_or_buf=sio, index=False) # pd
        # self.line_sol.to_csv(path_or_buf=sio, index=False) # pd
        # self.xfmr_sol.to_csv(path_or_buf=sio, index=False) # pd
        # self.swsh_sol.to_csv(path_or_buf=sio, index=False) # pd
        # with open(filename, 'w') as sol_file:
        #     sol_file.write(sio.getvalue())
        #     #sio.seek(0)
        #     #shutil.copyfileobj(sio, sol_file)
        # sio.close()
        
        # with open(filename, 'w') as sol_file:
        #     w = csv.writer(
        #         sol_file, delimiter=",", quotechar="'", quoting=csv.QUOTE_MINIMAL)
        #     for n in section_names:
        #         self.write_sol_case_section(
        #             sol_file, w, sol[n].values(), section_start_lines[n], section_field_names[n])

        sio = io.StringIO()
        w = csv.writer(
            sio, delimiter=",", quotechar="'", quoting=csv.QUOTE_MINIMAL)
        for n in section_names:
            self.write_sol_case_section(
                sio, w, sol[n].values(), section_start_lines[n], section_field_names[n])
        with open(filename, 'w') as sol_file:
            sol_file.write(sio.getvalue())
        sio.close()

        # todo maybe use some kind of string formatting thing on np or pd
        
        # with open(filename, 'w') as sol_file:
        #     w = csv.writer(
        #         sol_file, delimiter=",", quotechar="'", quoting=csv.QUOTE_MINIMAL)
        #     for n in section_names:
        #         self.write_sol_case_section(
        #             sol_file, w, sol[n].values(), section_start_lines[n], section_field_names[n])

        end_time = time.time()
        print('write_sol_case time: %f' % (end_time - start_time))

    def write_sol_case_section(self, open_file, writer, records, start_line, field_names):

        start_time = time.time()
        writer.writerow([start_line])
        writer.writerow(field_names)
        writer.writerows(records)
        # can we write self.sol_bus?
        #self.bus_sol1.to_csv(index=False) # pd to str
        #np.savetxt(open_file, self.bus_sol1) # np
        #writer.writerows(
        end_time = time.time()
        #print('write_sol_case_section time: %f' % (end_time - start_time))

