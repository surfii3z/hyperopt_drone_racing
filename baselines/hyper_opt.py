# hyper_opt.py
'''Hyper-parameters optimizer'''
import random
import numpy as np
import copy

V_MIN = 8.5
V_MAX = 35
A_MIN = 20
A_MAX = 160
D_MIN = 3.5
D_MAX = 6.5

class Range(object):
    '''
        Range of each of the hyper-parameter to be optimized
    '''
    def __init__(self, min_range, max_range):
        assert (max_range >= min_range), "max should be greater than min"
        self.min = min_range
        self.max = max_range

    def random_pick(self):
        '''
            uniformly draw a real number from Range [self.min, self.max)
        '''
        temp = random.uniform(self.min, self.max)
        return temp

class HyperParameter(object):
    def __init__(self, num):
        self.v = np.ones(num) * -1
        self.a = np.ones(num) * -1
        self.d = np.ones(num) * -1
        self.num = num


    def random_initialization(self):
        self.v = np.round(np.random.uniform(V_MIN, V_MAX, self.num), 2)
        self.a = np.round(np.random.uniform(A_MIN, A_MAX, self.num), 2)
        self.d = np.round(np.random.uniform(D_MIN, D_MAX, self.num), 2)


    def __str__(self):
        return f"v = {self.v}\na = {self.a}\nd = {self.d}"


    def random_mutation(self, max_num_mutation, last_gate_idx_before_termination):
        num_mutation = random.randint(1, max_num_mutation)
        for i in range(num_mutation):

            parameter_type = random.randint(0, 2)
            parameter_idx = random.randint(0, last_gate_idx_before_termination)
            if parameter_type == 0:
                new_v = round(Range(5, 25).random_pick(), 2)
                self.v[parameter_idx] = new_v
            elif parameter_type == 1:
                new_a = round(Range(60, 150).random_pick(), 2)
                self.a[parameter_idx] = new_a
            elif parameter_type == 2:
                new_d = round(Range(4, 15).random_pick(), 2)
                self.d[parameter_idx] = new_d
    

    def random_mutation_at_idx(self, idx):
        '''
            given idx, modify v, a, d randomly
        '''
        c = np.random.randint(low=2, size=3)

        if (sum(c[:-1]) == 0):   # there is no hyper-parameter update
            return

        self.random_mutation_v_at_idx(idx)
        # if c[0] == 1:
        #     self.random_mutation_v_at_idx(idx)
        if c[1] == 1:
            self.random_mutation_a_at_idx(idx)
        if c[2] == 1:
            self.random_mutation_d_at_idx(idx)


    def random_ensure_mutation_at_idx(self, idx):
        '''
            given idx, modify v, a, d randomly
        '''
        c = np.random.randint(low=2, size=3)

        while (sum(c[:-1]) == 0):   # there is no hyper-parameter update
            c = np.random.randint(low=2, size=3)

        self.random_mutation_v_at_idx(idx)
        # if c[0] == 1:
        #     self.random_mutation_v_at_idx(idx)
        if c[1] == 1:
            self.random_mutation_a_at_idx(idx)
        if c[2] == 1:
            self.random_mutation_d_at_idx(idx)
            

    def random_mutation_v_at_idx(self, idx):
        self.v[idx] = round(Range(V_MIN, V_MAX).random_pick(), 2)
    

    def random_mutation_a_at_idx(self, idx):
        self.a[idx] = round(Range(A_MIN, A_MAX).random_pick(), 2)


    def random_mutation_d_at_idx(self, idx):
        self.d[idx] = round(Range(D_MIN, D_MAX).random_pick(), 2)



if __name__ == "__main__":
    pass
    