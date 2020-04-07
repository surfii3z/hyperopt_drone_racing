# hyper_opt.py
'''Hyper-parameters optimizer'''
import random
import numpy as np
import copy

V_MIN = 5
V_MAX = 25
A_MIN = 50
A_MAX = 200
D_MIN = 2
D_MAX = 9

class Range(object):
    '''
        Range of each of the hyper-parameter to be optimized
    '''
    def __init__(self, min_range, max_range):
        assert (max_range >= min_range), "max should be greater than min"
        self.min = min_range
        self.max = max_range
        self.range = max_range - min_range

    def random_pick(self):
        '''
            uniformly draw a real number from Range [self.min, self.max)
        '''
        temp = random.random() * self.range + self.min
        assert (self.min <= temp <= self.max), "random_pick is out of range"
        return temp

class HyperParameter(object):
    def __init__(self, num):
        self.v = np.ones(num) * -1
        self.a = np.ones(num) * -1
        self.d = np.ones(num) * -1

    def __str__(self):
        return f"v = {self.v}\na = {self.a}\nd = {self.d}"

    def random_mutation(self, max_num_mutation, last_gate_idx_before_termination):
        num_mutation = random.randint(1, max_num_mutation)
        parameter_length = len(self.v)
        for i in range(num_mutation):

            parameter_type = random.randint(0, 2)
            parameter_idx = random.randint(0, last_gate_idx_before_termination)
            if parameter_type == 0:
                new_v = Range(5, 25).random_pick()
                self.v[parameter_idx] = new_v
            elif parameter_type == 1:
                new_a = Range(60, 150).random_pick()
                self.a[parameter_idx] = new_a
            elif parameter_type == 2:
                new_d = Range(4, 15).random_pick()
                self.d[parameter_idx] = new_d
    
    def random_mutation_at_idx(self, idx):
        '''
            given idx, modify v, a, d randomly
        '''
        c = np.random.randint(low=2, size=3)

        if (sum(c) == 0):   # there is no hyper-parameter update
            return

        if c[0] == 1:
            self.random_mutation_v_at_idx(idx)
        if c[1] == 1:
            self.random_mutation_a_at_idx(idx)
        if c[2] == 1:
            self.random_mutation_d_at_idx(idx)
            

    def random_mutation_v_at_idx(self, idx):
        self.v[idx] = Range(V_MIN, V_MAX).random_pick()
    
    def random_mutation_a_at_idx(self, idx):
        self.a[idx] = Range(A_MIN, A_MAX).random_pick()

    def random_mutation_d_at_idx(self, idx):
        self.d[idx] = Range(D_MIN, D_MAX).random_pick()



if __name__ == "__main__":
    t1 = HyperParameter()
    t1.v = 1
    t1.a = 2
    t1.d = 3
    print(t1)
    # t2 = HyperParameter()
    t2 = copy.copy(t1)
    print(t2)
    t2.v = 5
    print(t2.v)
    print(t1.v)

    print(Range(3, 20).random_pick())
    