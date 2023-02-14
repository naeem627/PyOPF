""" Functions for evaluatin piecewise linear convex functions,
manipulating their data representations, etc.

Author: Jesse Holzer, jesse.holzer@pnnl.gov
Author: Arun Veeramany, arun.veeramany@pnnl.gov

Date: 2020-07-23

"""
import numpy as np

DEBUG_MODE = False

def print_info(*args):

    if DEBUG_MODE:
        print(*args)

def make_pjn():

    pass


# general function to evaluate piecewise linear convex functions using numpy
# view such functions as giving the cost of production of a given quantity
#
# We are evaluating the cost c_j(p_j) for a set of products j
# let n_j be the number of products
# then p_j is the quantity of product j produced
# and c_j is the total cost of production of product j
#
# Assume the cost func function c_j is given by a set of offer blocks n
# characterized by m_{jn} and p_{jn}
# 
#
# input (all numpy arrays):
#
#   pj
#   quantity produced
#   num_j by 1
#   pj[:] >= 0
#
#   pjn
#   num_j by (num_n - 1)
#   pjn >= 0
#   pjn[:, n] <= pjn[:, (n + 1)] for n in range(num_n - 2)
#
#   mcj0
#   first marginal cost
#   num_j by 1
#
#   dmcjn
#   num_j by (num_n - 1)
#   dmcjn >= 0
#   
#def eval_cost(mcj0, dmcjn, pjn, pj):
#    
#    mcj0 * pj + dmcjn * max(0.0, pj - pjn)

def test():

    test1()
    test2()

def test1():

    num_j = 4
    num_f = 4
    num_b = 5
    functions = [
        [{'pmax': 5.0,
          'c': -1.0},
         {'pmax':10,
          'c': 2.0}],
        [{'pmax': 30.0,
          'c': 12.0},
         {'pmax':20,
          'c': 6.0}],
        [{'pmax': 30.0,
          'c': 12.0},
         {'pmax':20,
          'c': 16.0}],
        [{'pmax': 4.0,
          'c': 1.0},
         {'pmax':3,
          'c': 1.2},
         {'pmax':2,
          'c': 6.0},
         {'pmax':10,
          'c': 5.0},
         {'pmax':20,
          'c': 4.0}]]
    ce = CostEvaluator()
    ce.set_dims(num_j, num_f, num_b)
    ce.add_functions(functions)
    
    f_x = np.array([3.0, -1.0, 2.5, 6.4]).reshape(num_j, 1)
    f_z = np.zeros(shape=(num_j, 1))
    ce.eval_cost(f_x, f_z)

def test2():

    num_j = 4
    num_f = 1
    num_b = 5
    functions = [
        [{'pmax': 4.0,
          'c': 1.0},
         {'pmax':3,
          'c': 1.2},
         {'pmax':2,
          'c': 6.0},
         {'pmax':10,
          'c': 5.0},
         {'pmax':20,
          'c': 4.0}]]
    ce = CostEvaluator()
    ce.set_dims(num_j, num_f, num_b)
    ce.add_functions(functions)
    
    f_x = np.array([3.0, -1.0, 2.5, 6.4]).reshape(num_j, 1)
    f_z = np.zeros(shape=(num_j, 1))
    ce.eval_cost(f_x, f_z)

class CostEvaluator:

    def __init__(self, block_width_string=None, block_marginal_cost_string=None):

        if block_width_string is None:
            self.block_width_string = 'pmax'
        else:
            self.block_width_string = block_width_string
        if block_marginal_cost_string is None:
            self.block_marginal_cost_string = 'c'
        else:
            self.block_marginal_cost_string = block_marginal_cost_string
        self.num_j = 0 # number of products
        self.num_f = 0 # number of cost functions - equals either 1 or the number of products
        self.num_b = 0 # maximum number of blocks per cost function - shorter cost functions are padded with 0 blocks
        self.f_z_at_x_max_computed = False

    def setup(self, num_j, functions):

        max_num_cost_blocks = max([0] + [len(f) for f in functions])
        self.set_dims(num_j, len(functions), max_num_cost_blocks)
        self.add_functions(functions)

    def set_dims(self, num_j, num_f, num_b):

        self.num_j = num_j
        self.num_f = num_f
        self.num_b = num_b
        self.fb_c = np.zeros(shape=(self.num_f, self.num_b))
        self.fb_x = np.zeros(shape=(self.num_f, self.num_b))
        self.fb_y = np.zeros(shape=(self.num_j, self.num_b))
        self.f_x_max = np.zeros(shape=(self.num_f,))

    def add_functions(self, functions):

        print_info('functions:')
        for f in functions:
            print_info(f)

        num_f = len(functions)
        num_b = max([0] + [len(f) for f in functions])
        assert(num_f == self.num_f)
        assert(num_b == self.num_b)

        for f in range(self.num_f):
            self.fb_c[f, 0:len(functions[f])] = [b[self.block_marginal_cost_string] for b in functions[f]]
            self.fb_x[f, 0:len(functions[f])] = [b[self.block_width_string] for b in functions[f]]
        print_info('fb_c:')
        print_info(self.fb_c)
        print_info('fb_x:')
        print_info(self.fb_x)

        ordering = np.argsort(self.fb_c, axis=1)
        print_info('ordering:')
        print_info(ordering)

        self.fb_c = np.take_along_axis(self.fb_c, ordering, axis=1)
        self.fb_x = np.take_along_axis(self.fb_x, ordering, axis=1)
        print_info('fb_c:')
        print_info(self.fb_c)
        print_info('fb_x:')
        print_info(self.fb_x)        

        np.subtract(self.fb_c[:, 1:self.num_b], self.fb_c[:, 0:(self.num_b - 1)], out=self.fb_c[:, 1:self.num_b])
        np.cumsum(self.fb_x, axis=1, out=self.fb_x)
        self.f_x_max[:] = self.fb_x[:, self.num_b - 1] # need to copy values into fxmax because later we set this last column to 0
        #self.f_x_max.shape = (self.num_f,)
        self.fb_x[:, 1:self.num_b] = self.fb_x[:, 0:(self.num_b - 1)]
        self.fb_x[:, 0] = 0.0
        print_info('fb_c:')
        print_info(self.fb_c)
        print_info('fb_x:')
        print_info(self.fb_x)

    def compute_f_z_at_x_max(self):
        
        assert(not self.f_z_at_x_max_computed)
        self.f_z_at_x_max = np.zeros(shape=(self.num_f,))
        self.eval_cost(self.f_x_max, self.f_z_at_x_max)
        self.f_z_at_x_max_computed = True

    def eval_benefit(self, f_x, f_z):

        assert(self.f_z_at_x_max_computed)
        #print('here!')
        #print(self.f_x_max)
        #print(f_x)
        np.subtract(self.f_x_max, f_x, out=f_z)
        #print(f_z)
        #f_z_temp = np.array(f_z)
        #self.eval_cost(f_z_temp, f_z)
        self.eval_cost(f_z, f_z)
        #print(f_z)
        #print(self.f_z_at_x_max)
        np.subtract(self.f_z_at_x_max, f_z, out=f_z)
        #print(f_z)

    def eval_cost(self, f_x, f_z):
        '''
        f_x: vector of production quantities
        f_z: vector of cost values
        '''

        print_info('f_x:')
        print_info(f_x)
        
        assert(f_x.size == self.num_j)
        assert(f_z.size == self.num_j)

        # later restore the shapes
        f_x_shape = f_x.shape
        f_z_shape = f_z.shape

        # make column vectors for now
        f_x.shape = (self.num_j, 1)
        f_z.shape = (self.num_j, 1)
        
        np.subtract(f_x, self.fb_x, out=self.fb_y)
        print_info('fb_y:')
        print_info(self.fb_y)

        np.clip(self.fb_y, a_min=0.0, a_max=None, out=self.fb_y)
        print_info('fb_y:')
        print_info(self.fb_y)

        np.multiply(self.fb_c, self.fb_y, out=self.fb_y)
        print_info('fb_y:')
        print_info(self.fb_y)

        np.sum(self.fb_y, axis=1, keepdims=True, out=f_z)
        print_info('f_z:')
        print_info(f_z)

        # restore shapes
        f_x.shape = f_x_shape
        f_z.shape = f_z_shape

if __name__ == "__main__":
    test()
