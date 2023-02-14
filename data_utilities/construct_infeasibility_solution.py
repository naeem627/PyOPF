'''
syntax:

from a command prompt:
python construct_infeasibility_solution.py raw sup con sol_dir

from a Python interpreter:
import sys
sys.argv = [raw, sup, con, sol_dir]
execfile("construct_infeasibility_solution.py")

sup is the JSON-formatted supplementary data file

Construct the infeasibility solution for a given problem instance
'''

import argparse
import time

# gocomp imports
try:
    from data_utilities.infeasibility_solution import Solver
except:
    from infeasibility_solution import Solver

import os

def main():

    parser = argparse.ArgumentParser(description='Construct the infeasibility solution for a problem instance')
    
    parser.add_argument('raw', help='raw')
    parser.add_argument('sup', help='sup')
    parser.add_argument('con', help='con')
    parser.add_argument('sol_dir', help='sol_dir')
    
    args = parser.parse_args()

    s = Solver()
    s.data.read(args.raw, args.sup, args.con)

    # write solution files
    start_time = time.time()
    s.write_sol(args.sol_dir)
    time_elapsed = time.time() - start_time
    print("write solution time: %f" % time_elapsed)

if __name__ == '__main__':

    #Use this block to test directly from here
    print(os.getcwd())
    case_dir='./data/ieee14/scenario_1/'
    sol_dir='./data/ieee14/scenario_1/'
    raw=f"../{case_dir}case.raw"
    sup=f"../{case_dir}case.json"
    con=f"../{case_dir}case.con"
    #sys.argv.extend([ raw, sup, con, f"../{sol_dir}" ])

    main()
