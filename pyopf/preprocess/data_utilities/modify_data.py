'''
syntax:

from a command prompt:
python modify_data.py raw_in sup_in con_in raw_out sup_out con_out

from a Python interpreter:
import sys
sys.argv = [raw_in, sup_in, con_in, raw_out, sup_out, con_out]
execfile("modify_data.py")

modifications supported:
1. Fix load tmin and tmax to a given value, then project to ensure feasibility
   4. = original tmin
   2. = original tmax
   5. = 1.0
   3. = values read in from a file (formatted as a case solution file?)

modifications proposed:
2. set all generators suqual and sdqual to 0
3. 
'''

import argparse
import time

# gocomp imports
try:
    import pyopf.preprocess.data_utilities.data as data
    from pyopf.preprocess.data_utilities.evaluation import Evaluation, CaseSolution
except:
    import data
    from evaluation import Evaluation, CaseSolution
    
def main():

    parser = argparse.ArgumentParser(description='Modify the data for a problem instance')
    
    parser.add_argument('raw_in', help='raw_in')
    parser.add_argument('sup_in', help='sup_in')
    parser.add_argument('con_in', help='con_in')
    parser.add_argument('raw_out', help='raw_out')
    parser.add_argument('sup_out', help='sup_out')
    parser.add_argument('con_out', help='con_out')
    parser.add_argument('load_mode', help='load_mode')
    parser.add_argument('case_sol', help='case_sol')
    
    args = parser.parse_args()

    start_time = time.time()
    p = data.Data()
    p.sup.do_force_defaults = True # TODO need to add this to scrubber code in other places?
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
    case_sol = None
    do_load_base_case_sol = False
    print('hello world 1')
    if args.load_mode in ['given']:
        do_load_base_case_sol = True
    if do_load_base_case_sol:
        e = Evaluation()
        e.set_data(p, convert_units=False)
        case_sol = CaseSolution()
        case_sol.set_array_dims(e)
        case_sol.set_maps(e)
        case_sol.init_arrays()
        case_sol.set_read_dims()
        case_sol.read(args.case_sol)
        case_sol.set_arrays_from_dfs()
        case_sol.round()
    end_time = time.time()
    print("load solution time: %f" % (end_time - start_time))

    start_time = time.time()
    print('modifying')
    print('load_mode: {}'.format(args.load_mode))
    print('case_sol: {}'.format(args.case_sol))
    p.modify(load_mode=args.load_mode, case_sol=case_sol) # max, min, 1, given. if using given, need to supply values also. todo later
    end_time = time.time()
    print("modify data time: %f" % (end_time - start_time))

    start_time = time.time()
    p.write(args.raw_out, args.sup_out, args.con_out)
    time_elapsed = time.time() - start_time
    print("write data time: %f" % time_elapsed)

if __name__ == '__main__':
    main()
