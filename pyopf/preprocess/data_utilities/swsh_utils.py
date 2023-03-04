#
# Grid Optimization Competion
# author: Jesse Holzer
# date: 2020-09-06
#

try:
    from pyopf.preprocess.data_utilities.swsh_utils_py import solve as solve_py
except:
    from swsh_utils_py import solve as solve_py



# try:
#     from data_utilities.swsh_utils_cy import solve as solve_cy
# except Exception as e:
#     print('could not import swsh_utils_cy. this will not be a problem if you do not use solve_cy. If you want to use solve_cy, do "python setup.py build_ext --inplace" to compile it to a Python extension module.')
#     #raise e
