'''
syntax:

from a command prompt:
python check_data.py raw sup con

from a Python interpreter:
import sys
sys.argv = [raw, sup, con]
execfile("check_data.py")

sup is the JSON-formatted supplementary data file
'''

import argparse
import time

import numpy as np

# gocomp imports
#import data_utilities.data as data
#from data_utilities.evaluation import Evaluation, CaseSolution
import pyopf.preprocess.data_utilities.data as data
from pyopf.preprocess.data_utilities.evaluation import Evaluation, CaseSolution, create_new_summary

    
def main():

    parser = argparse.ArgumentParser(description='Check data files for a problem instance')
    
    parser.add_argument('raw_in', help='raw_in')
    parser.add_argument('sup_in', help='sup_in')
    parser.add_argument('con_in', help='con_in')
    
    args = parser.parse_args()

    start_time = time.time()
    p = data.Data()
    p.read(args.raw_in, args.sup_in, args.con_in)
    time_elapsed = time.time() - start_time
    print("read data time: %f" % time_elapsed)
    
    # show data stats
    p.print_summary()

    start_time = time.time()
    p.check()
    time_elapsed = time.time() - start_time
    print("check data time: %f" % time_elapsed)

    # check prior point power balance, etc.
    # some checks should not be done,
    # e.g. pmin, pmax, tmin, tmax, as thes change from prior to base
    # set up evaluation
    start_time = time.time()
    e = Evaluation()
    e.set_data(p)
    e.set_sol_initialize()
    e.eval_min_max_total_load_benefit()
    s = CaseSolution()
    s.set_array_dims(e)
    s.set_maps(e)
    s.init_arrays()
    e.set_data_for_base()
    e.set_prior_from_data_for_base()
    e.summary_written = False
    e.summary = create_new_summary()
    e.eval_prior_bus_pow()

    bus_max_bus_pow_real_imbalance = np.argmax(np.absolute(e.bus_pow_real_imbalance))
    bus_max_bus_pow_imag_imbalance = np.argmax(np.absolute(e.bus_pow_imag_imbalance))
    max_bus_pow_real_imbalance = e.base_mva * e.bus_pow_real_imbalance[bus_max_bus_pow_real_imbalance]
    max_bus_pow_imag_imbalance = e.base_mva * e.bus_pow_imag_imbalance[bus_max_bus_pow_imag_imbalance]
    bus_max_bus_pow_real_imbalance = e.bus_i[bus_max_bus_pow_real_imbalance]
    bus_max_bus_pow_imag_imbalance = e.bus_i[bus_max_bus_pow_imag_imbalance]
    power_balance_report = 'max bus power imbalance (MVA): [real: [bus: {}, val: {}], imag: [bus: {}, val: {}]]'.format(
        bus_max_bus_pow_real_imbalance,
        max_bus_pow_real_imbalance,
        bus_max_bus_pow_imag_imbalance,
        max_bus_pow_imag_imbalance)
    if max(abs(max_bus_pow_real_imbalance), abs(max_bus_pow_imag_imbalance)) > data.prior_point_pow_imbalance_tol:
        data.alert(
            {'data_type': 'Raw',
             'error_message': 'power imbalance in prior point exceeds tolerance',
             'diagnostics': (power_balance_report + ', tol: {}'.format(data.prior_point_pow_imbalance_tol))})
    time_elapsed = time.time() - start_time
    print("check data power balance time: %f" % time_elapsed)

if __name__ == '__main__':
    main()
