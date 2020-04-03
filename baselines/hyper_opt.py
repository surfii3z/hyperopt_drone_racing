# hyper_opt.py
'''Hyper-parameters optimizer'''
import random
import numpy as np
import copy

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
    def __init__(self):
        self.v = None
        self.a = None
        self.d = None

    def __str__(self):
        return f"v = {self.v}\na = {self.a}\nd = {self.d}"

class HyperOpt(object):
    def __init__(self):
        '''
            Do Something
        '''
        pass

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
    