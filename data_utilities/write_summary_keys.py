'''
syntax:

from a command prompt:
python write_summary_keys <out_file_name>
'''

import argparse
try:
    from data_utilities.evaluation import write_summary_keys
except:
    from evaluation import write_summary_keys

def main(out_file_name):

    write_summary_keys(out_file_name)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='write evaluator summary output file keys into a given file')
    parser.add_argument('out_file_name', help='out_file_name')
    args = parser.parse_args()
    main(args.out_file_name)
