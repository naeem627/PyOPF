"""
store problem data in numpy arrays
Author: Jesse Holzer, jesse.holzer@pnnl.gov
Date: 2020-09-18
"""

#import inspect
#import os
#import re
# import inspect
import math
import sys
# import os
# import re
import time
import traceback

try:
    import pyopf.preprocess.data_utilities.data as p_data
except:
    import data as p_data
import numpy as np
#import pandas as pd
#from cost_utils import CostEvaluator
from scipy import sparse as sp

# tolerance on hard constraints (in the units of the constraint, typically pu)
hard_constr_tol = 1e-4
num_swsh_block = 8
num_xfmr_imp_corr_point = 11

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

#sys.excepthook = uncaught_exception_handler

def timeit(function):
    def timed(*args, **kw):
        start_time = time.time()
        result = function(*args, **kw)
        end_time = time.time()
        print('function: {}, time: {}'.format(function.__name__, end_time - start_time))
        return result
    return timed

class Data:
    '''In per unit convention, i.e. same as the model'''

    def __init__(self):

        pass
    
    @timeit
    def set_data_scalars(self, data):

        self.base_mva = data.raw.case_identification.sbase
        self.delta_base = data.sup.sup_jsonobj['systemparameters']['delta']
        self.delta_r_base = data.sup.sup_jsonobj['systemparameters']['deltar']
        self.delta_ctg = data.sup.sup_jsonobj['systemparameters']['deltactg']
        self.delta_r_ctg = data.sup.sup_jsonobj['systemparameters']['deltarctg']
        self.epsilon = hard_constr_tol
        self.num_swsh_block = num_swsh_block
        self.num_xfmr_imp_corr_point = num_xfmr_imp_corr_point

    @timeit
    def set_data_bus_params(self, data):

        buses = list(data.raw.buses.values())
        self.num_bus = len(buses)
        self.bus_i = [r.i for r in buses]
        self.bus_key = self.bus_i
        self.bus_map = {self.bus_i[i]:i for i in range(len(self.bus_i))}
        self.bus_volt_mag_0 = np.array([r.vm for r in buses])
        self.bus_volt_ang_0 = np.array([r.va for r in buses]) * np.pi / 180.0
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
            [data.sup.loads[k]['tmin'] for k in self.load_key])
        self.load_t_max = np.array(
            [data.sup.loads[k]['tmax'] for k in self.load_key])
        self.load_ramp_up_max_base = np.array(
            [data.sup.loads[k]['prumax'] for k in self.load_key]) / self.base_mva
        self.load_ramp_down_max_base = np.array(
            [data.sup.loads[k]['prdmax'] for k in self.load_key]) / self.base_mva
        self.load_ramp_up_max_ctg = np.array(
            [data.sup.loads[k]['prumaxctg'] for k in self.load_key]) / self.base_mva
        self.load_ramp_down_max_ctg = np.array(
            [data.sup.loads[k]['prdmaxctg'] for k in self.load_key]) / self.base_mva

        self.bus_load_matrix = sp.csr_matrix(
            ([1.0 for i in range(self.num_load)],
             (self.load_bus, list(range(self.num_load)))),
            (self.num_bus, self.num_load))
        self.bus_loads = [[] for i in range(self.num_bus)]
        for j in range(self.num_load):
            self.bus_loads[self.load_bus[j]].append(j)
        self.bus_num_load = [len(self.bus_loads[i]) for i in range(self.num_bus)]

    @timeit
    def set_data_load_cost_params(self, data):
        
        self.load_num_cost_block = np.array(
            [len(data.sup.loads[k]['cblocks']) for k in self.load_key],
            dtype=int)
        self.num_load_cost_block = np.amax(self.load_num_cost_block)
        self.load_cost_block_max_quantity = np.zeros(
            shape=(self.num_load, self.num_load_cost_block))
        self.load_cost_block_marginal = np.zeros(
            shape=(self.num_load, self.num_load_cost_block))
        for i in range(self.num_load):
            k = self.load_key[i]
            self.load_cost_block_max_quantity[i, 0:self.load_num_cost_block[i]] = [
                b['pmax'] for b in data.sup.loads[k]['cblocks']]
            self.load_cost_block_marginal[i, 0:self.load_num_cost_block[i]] = [
                b['c'] for b in data.sup.loads[k]['cblocks']]
        self.load_cost_block_max_quantity /= self.base_mva
        self.load_cost_block_marginal *= self.base_mva

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
        self.bus_fxshs = [[] for i in range(self.num_bus)]
        for j in range(self.num_fxsh):
            self.bus_fxshs[self.fxsh_bus[j]].append(j)
        self.bus_num_fxsh = [len(self.bus_fxshs[i]) for i in range(self.num_bus)]

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
        self.bus_gens = [[] for i in range(self.num_bus)]
        for j in range(self.num_gen):
            self.bus_gens[self.gen_bus[j]].append(j)
        self.bus_num_gen = [len(self.bus_gens[i]) for i in range(self.num_bus)]

    @timeit
    def set_data_gen_cost_params(self, data):
        
        self.gen_num_cost_block = np.array(
            [len(data.sup.generators[k]['cblocks']) for k in self.gen_key],
            dtype=int)
        self.num_gen_cost_block = np.amax(self.gen_num_cost_block)
        self.gen_cost_block_max_quantity = np.zeros(
            shape=(self.num_gen, self.num_gen_cost_block))
        self.gen_cost_block_marginal = np.zeros(
            shape=(self.num_gen, self.num_gen_cost_block))
        for i in range(self.num_gen):
            k = self.gen_key[i]
            self.gen_cost_block_max_quantity[i, 0:self.gen_num_cost_block[i]] = [
                b['pmax'] for b in data.sup.generators[k]['cblocks']]
            self.gen_cost_block_marginal[i, 0:self.gen_num_cost_block[i]] = [
                b['c'] for b in data.sup.generators[k]['cblocks']]
        self.gen_cost_block_max_quantity /= self.base_mva
        self.gen_cost_block_marginal *= self.base_mva

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

        self.line_imp_real = np.array([r.r for r in lines])
        self.line_imp_imag = np.array([r.x for r in lines])
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
        # todo: careful with this
        self.line_sw_qual = np.array([data.sup.lines[k]['swqual'] for k in self.line_key])
        #if self.line_switching_allowed:
        #    self.line_sw_qual = np.array([data.sup.lines[k]['swqual'] for k in self.line_key])
        #else:
        #    self.line_sw_qual = np.zeros(shape=(self.num_line))
        self.line_service_status = np.ones(shape=(self.num_line,))
        self.bus_lines_orig = [[] for i in range(self.num_bus)]
        self.bus_lines_dest = [[] for i in range(self.num_bus)]
        for j in range(self.num_line):
            self.bus_lines_orig[self.line_orig_bus[j]].append(j)
            self.bus_lines_dest[self.line_dest_bus[j]].append(j)
        self.bus_num_line_orig = [len(self.bus_lines_orig[i]) for i in range(self.num_bus)]
        self.bus_num_line_dest = [len(self.bus_lines_dest[i]) for i in range(self.num_bus)]

    # @timeit
    # def set_data_line_cost_params(self, data):
        
    #     self.num_line_cost_block = len(data.sup.scblocks)
    #     self.line_num_cost_block = np.array(
    #         [self.num_line_cost_block for k in self.line_key],
    #         dtype=int)
    #     self.line_cost_block_max_quantity = np.array(
    #         shape=(self.num_line, self.num_line_cost_block))
    #     self.line_cost_block_marginal = np.zeros(
    #         shape=(self.num_line, self.num_line_cost_block))
    #     for i in range(self.num_line):
    #         k = self.line_key[i]
    #         self.line_cost_block_max_quantity[i, 0:self.line_num_cost_block[i]] = [
    #             b['pmax'] for b in data.sup.lines[k]['cblocks']]
    #         self.line_cost_block_marginal[i, 0:self.line_num_cost_block[i]] = [
    #             b['c'] for b in data.sup.lines[k]['cblocks']]
    #     self.line_cost_block_max_quantity /= self.base_mva
    #     self.line_cost_block_marginal *= self.base_mva

    @timeit
    def set_data_xfmr_params(self, data):

        xfmrs = list(data.raw.transformers.values()) # note here we take all xfmrs, regardless of status in RAW
        self.num_xfmr = len(xfmrs)
        self.xfmr_i = [r.i for r in xfmrs]
        self.xfmr_j = [r.j for r in xfmrs]
        self.xfmr_ckt = [r.ckt for r in xfmrs]
        self.xfmr_key = [(r.i, r.j, r.ckt) for r in xfmrs] # do we really need the '0'?
        self.xfmr_orig_bus = [self.bus_map[self.xfmr_i[i]] for i in range(self.num_xfmr)]
        self.xfmr_dest_bus = [self.bus_map[self.xfmr_j[i]] for i in range(self.num_xfmr)]
        self.xfmr_map = {(self.xfmr_i[i], self.xfmr_j[i], self.xfmr_ckt[i]):i for i in range(self.num_xfmr)}

        # closed-open status in operating point prior to base case
        self.xfmr_xsw_0 = np.array([r.stat for r in xfmrs])
        
        # series admittance (conductance and susceptance) from data
        # impedance correction divides these by impedance correction factor
        self.xfmr_imp_real_0 = np.array([r.r12 for r in xfmrs])
        self.xfmr_imp_imag_0 = np.array([r.x12 for r in xfmrs])
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

        self.bus_xfmr_orig_matrix = sp.csr_matrix(
            ([1.0 for i in range(self.num_xfmr)],
             (self.xfmr_orig_bus, list(range(self.num_xfmr)))),
            (self.num_bus, self.num_xfmr))
        self.bus_xfmr_dest_matrix = sp.csr_matrix(
            ([1.0 for i in range(self.num_xfmr)],
             (self.xfmr_dest_bus, list(range(self.num_xfmr)))),
            (self.num_bus, self.num_xfmr))
        self.xfmr_sw_cost = np.array([data.sup.transformers[k]['csw'] for k in self.xfmr_key])
        # todo: careful with this
        self.xfmr_sw_qual = np.array([data.sup.transformers[k]['swqual'] for k in self.xfmr_key])
        #if self.xfmr_switching_allowed:
        #    self.xfmr_sw_qual = np.array([data.sup.transformers[k]['swqual'] for k in self.xfmr_key])
        #else:
        #    self.xfmr_sw_qual = np.zeros(shape=(self.num_xfmr))
        self.xfmr_service_status = np.ones(shape=(self.num_xfmr,))

        # todo transformer impedance correction
        # todo which of these 2 options is best?
        #self.xfmr_index_imp_corr = [ind for ind in range(self.num_xfmr) if (xfmrs[ind].tab1 > 0 and xfmrs[ind].cod1 in [-3, -1, 1, 3])]
        self.xfmr_index_imp_corr = [ind for ind in range(self.num_xfmr) if xfmrs[ind].tab1 > 0]
        #self.xfmr_index_fixed_tap_ratio_and_phase_shift = [ind for ind in range(self.num_xfmr) if xfmrs[ind].cod1 == 0]
        self.xfmr_index_fixed_tap_ratio_and_phase_shift = [ind for ind in range(self.num_xfmr) if xfmrs[ind].cod1 not in [-3, -1, 1, 3]]
        self.xfmr_index_var_tap_ratio = [ind for ind in range(self.num_xfmr) if xfmrs[ind].cod1 in [-1, 1]]
        self.xfmr_index_var_phase_shift = [ind for ind in range(self.num_xfmr) if xfmrs[ind].cod1 in [-3, 3]]
        self.xfmr_index_imp_corr_var_tap_ratio = sorted(
            list(set(self.xfmr_index_imp_corr).intersection(
                    set(self.xfmr_index_var_tap_ratio))))
        self.xfmr_index_imp_corr_var_phase_shift = sorted(
            list(set(self.xfmr_index_imp_corr).intersection(
                    set(self.xfmr_index_var_phase_shift))))

        # some topology information
        self.bus_xfmrs_orig = [[] for i in range(self.num_bus)]
        self.bus_xfmrs_dest = [[] for i in range(self.num_bus)]
        for j in range(self.num_xfmr):
            self.bus_xfmrs_orig[self.xfmr_orig_bus[j]].append(j)
            self.bus_xfmrs_dest[self.xfmr_dest_bus[j]].append(j)
        self.bus_num_xfmr_orig = [len(self.bus_xfmrs_orig[i]) for i in range(self.num_bus)]
        self.bus_num_xfmr_dest = [len(self.bus_xfmrs_dest[i]) for i in range(self.num_bus)]

    @timeit
    def set_data_swsh_params(self, data):

        swshs = [r for r in data.raw.switched_shunts.values() if r.stat == 1]
        self.num_swsh = len(swshs)
        self.swsh_i = [r.i for r in swshs]
        self.swsh_key = self.swsh_i
        self.swsh_bus = [self.bus_map[self.swsh_i[i]] for i in range(self.num_swsh)]
        self.swsh_map = {self.swsh_i[i]:i for i in range(self.num_swsh)}
        self.swsh_adm_imag_init = np.array([r.binit for r in swshs]) / self.base_mva
        self.swsh_block_adm_imag = np.array(
            [[r.b1, r.b2, r.b3, r.b4, r.b5, r.b6, r.b7, r.b8]
             for r in swshs]) / self.base_mva
        self.swsh_block_num_steps = np.array(
            [[r.n1, r.n2, r.n3, r.n4, r.n5, r.n6, r.n7, r.n8]
             for r in swshs])
        if self.num_swsh == 0:
            self.swsh_block_num_steps.shape = (0, 8)
            self.swsh_block_adm_imag.shape = (0, 8)
        self.swsh_num_blocks = np.array(
            [r.swsh_susc_count for r in swshs])
        self.bus_swsh_matrix = sp.csr_matrix(
            ([1.0 for i in range(self.num_swsh)],
             (self.swsh_bus, list(range(self.num_swsh)))),
            (self.num_bus, self.num_swsh))
        self.bus_swshs = [[] for i in range(self.num_bus)]
        for j in range(self.num_swsh):
            self.bus_swshs[self.swsh_bus[j]].append(j)
        self.bus_num_swsh = [len(self.bus_swshs[i]) for i in range(self.num_bus)]

    # @timeit
    # def set_data_gen_cost_params(self, data):

    #     #print_info('pcblocks:')
    #     #print_info(data.sup.sup_jsonobj['pcblocks'])
    #     data.sup.convert_generator_cblock_units(self.base_mva)
    #     data.sup.convert_load_cblock_units(self.base_mva)
    #     data.sup.convert_pcblock_units(self.base_mva)
    #     data.sup.convert_qcblock_units(self.base_mva)
    #     data.sup.convert_scblock_units(self.base_mva)
    #     #print_info('pcblocks:')
    #     #print_info(data.sup.sup_jsonobj['pcblocks'])

    @timeit
    def set_data_ctg_params(self, data):
        # TODO 1/2 this is incomplete ? maybe? need to check
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

        ctg_line_keys_out = {k:(v & line_keys) for k,v in ctg_branch_keys_out.items()}
        ctg_xfmr_keys_out = {k:(v & xfmr_keys) for k,v in ctg_branch_keys_out.items()}

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

        self.ctg_index_gen_out = [i for i in range(self.num_ctg) if self.ctg_num_gens_out[i] > 0]
        self.ctg_index_line_out = [i for i in range(self.num_ctg) if self.ctg_num_lines_out[i] > 0]
        self.ctg_index_xfmr_out = [i for i in range(self.num_ctg) if self.ctg_num_xfmrs_out[i] > 0]

        self.ctg_gen_out = [self.ctg_gens_out[i][0] for i in self.ctg_index_gen_out]
        self.ctg_line_out = [self.ctg_lines_out[i][0] for i in self.ctg_index_line_out]
        self.ctg_xfmr_out = [self.ctg_xfmrs_out[i][0] for i in self.ctg_index_xfmr_out]

    @timeit
    def read(self, raw, sup, con):
        p = p_data.Data()
        p.read(raw, sup, con)
        self.set_data(p)

    @timeit
    def set_data(self, data):
        ''' set values from the data object
        convert to per unit (p.u.) convention'''

        self.set_data_scalars(data)
        self.set_data_bus_params(data)
        self.set_data_load_params(data)
        self.set_data_fxsh_params(data)
        self.set_data_gen_params(data)
        self.set_data_line_params(data)
        self.set_data_xfmr_params(data)
        self.set_data_swsh_params(data)
        self.set_data_load_cost_params(data)
        self.set_data_gen_cost_params(data)
        self.set_data_system_cost_params(data)
        #self.set_data_line_cost_params(data)
        #self.set_data_xfmr_cost_params(data)
        self.set_data_ctg_params(data)

    @timeit
    def set_data_system_cost_params(self, data):

        self.num_cost_block_p = len(data.sup.sup_jsonobj['pcblocks'])
        self.cost_block_p_max_quantity = np.array(
            [b['pmax'] for b in data.sup.sup_jsonobj['pcblocks']]) / self.base_mva
        self.cost_block_p_marginal = np.array(
            [b['c'] for b in data.sup.sup_jsonobj['pcblocks']]) * self.base_mva
        self.num_cost_block_q = len(data.sup.sup_jsonobj['qcblocks'])
        self.cost_block_q_max_quantity = np.array(
            [b['qmax'] for b in data.sup.sup_jsonobj['qcblocks']]) / self.base_mva
        self.cost_block_q_marginal = np.array(
            [b['c'] for b in data.sup.sup_jsonobj['qcblocks']]) * self.base_mva
        self.num_cost_block_s = len(data.sup.sup_jsonobj['scblocks'])
        self.cost_block_s_max_quantity = np.array(
            [b['tmax'] for b in data.sup.sup_jsonobj['scblocks']]) # note no need to deal with base_mva here
        self.cost_block_s_marginal = np.array(
            [b['c'] for b in data.sup.sup_jsonobj['scblocks']]) * self.base_mva
        
    @timeit
    def set_cost_evaluators(self, data):
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
        self.bus_real_cost_evaluator.setup(self.num_bus, [data.sup.sup_jsonobj['pcblocks']])
        self.bus_imag_cost_evaluator.setup(self.num_bus, [data.sup.sup_jsonobj['qcblocks']])
        self.load_cost_evaluator.setup(self.num_load, [data.sup.loads[k]['cblocks'] for k in self.load_key])
        self.gen_cost_evaluator.setup(self.num_gen, [data.sup.generators[k]['cblocks'] for k in self.gen_key])
        self.line_cost_evaluator_base.setup(
            self.num_line,
            [[{'smax': (b['tmax'] * self.line_curr_mag_max_base[i]),
               'c': b['c']}
              for b in data.sup.sup_jsonobj['scblocks']]
             for i in range(self.num_line)])
        self.line_cost_evaluator_ctg.setup(
            self.num_line,
            [[{'smax': (b['tmax'] * self.line_curr_mag_max_ctg[i]),
               'c': b['c']}
              for b in data.sup.sup_jsonobj['scblocks']]
             for i in range(self.num_line)])
        self.xfmr_cost_evaluator_base.setup(
            self.num_xfmr,
            [[{'smax': (b['tmax'] * self.xfmr_pow_mag_max_base[i]),
               'c': b['c']}
              for b in data.sup.sup_jsonobj['scblocks']]
             for i in range(self.num_xfmr)])
        self.xfmr_cost_evaluator_ctg.setup(
            self.num_xfmr,
            [[{'smax': (b['tmax'] * self.xfmr_pow_mag_max_ctg[i]),
               'c': b['c']}
              for b in data.sup.sup_jsonobj['scblocks']]
             for i in range(self.num_xfmr)])

        # extra stuff to translate between load benefit and cost
        self.load_cost_evaluator.compute_f_z_at_x_max()
