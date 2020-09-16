# hyOpt.py
'''Hyper-parameters optimizer'''
import random
import numpy as np
import copy

class HyperParameter(object):
    def __init__(self, num):
        self.v = None
        self.a = None
        self.d = None
        self.v_range = None
        self.a_range = None
        self.d_range = None
        self.time = None
        self.num_hyper = num


    def __str__(self):
        return f"v = {self.v}\na = {self.a}\nd = {self.d}"


    def init_hypers(self, v_val, a_val, d_val):
        self.init_v(v_val)
        self.init_a(a_val)
        self.init_d(d_val)

    def random_init_hypers(self):
        assert not(self.v_range is None or self.a_range is None or self.d_range is None), "range is not initialized"
        self.v = np.round(np.random.uniform(self.v_range[0], self.v_range[1], self.num_hyper), 2)
        self.a = np.round(np.random.uniform(self.a_range[0], self.a_range[1], self.num_hyper), 2)
        self.d = np.round(np.random.uniform(self.d_range[0], self.d_range[1], self.num_hyper), 2)


    def init_v(self, v_val):
        self.v = np.ones(self.num_hyper) * v_val


    def init_a(self, a_val):
        self.a = np.ones(self.num_hyper) * a_val


    def init_d(self, d_val):
        self.d = np.ones(self.num_hyper) * d_val
        self.d[-1] = 2  # to ensure that it finishes the race


    def init_time(self):
        self.time = np.ones(self.num_hyper) * 1000.


    def set_v_range(self, v_range):
        assert(v_range[0] <= v_range[1]), "v_range[0] should be less than v_range[1]"
        self.v_range = (v_range[0], v_range[1])


    def set_a_range(self, a_range):
        assert(a_range[0] <= a_range[1]), "a_range[0] should be less than a_range[1]"
        self.a_range = (a_range[0], a_range[1])


    def set_d_range(self, d_range):
        assert(d_range[0] <= d_range[1]), "d_range[0] should be less than d_range[1]"
        self.d_range = (d_range[0], d_range[1])


    def random_mutation_v_at_idx(self, idx):
        assert(not self.v_range is None), "v_range is not initialized"
        self.v[idx] = random.uniform(self.v_range[0], self.v_range[1])
        self.v[idx] = round(self.v[idx], 2)


    def random_mutation_a_at_idx(self, idx):
        assert(not self.a_range is None), "a_range is not initialized"
        self.a[idx] = random.uniform(self.a_range[0], self.a_range[1])
        self.a[idx] = round(self.a[idx], 2)


    def random_mutation_d_at_idx(self, idx):
        assert(not self.d_range is None), "d_range is not initialized"
        self.d[idx] = random.uniform(self.d_range[0], self.d_range[1])
        self.d[idx] = round(self.d[idx], 2)


class hyOpt(HyperParameter):

    def __init__(self, num):
        self.num_hyper = num
        self.curr_hyper = HyperParameter(num)
        self.best_hyper = HyperParameter(num)
        

    def update_best_hyper(self):
        '''
            IF the current race wins
                - update the best hyperparameters with the current hyperparameters
                - update the best race time with the current race time
                - random mutation
            ELSE
                - update the best hyperparameters up until where the best hyperparameters
                  loses to the current hyperparameters (Start looking at the back)
                - update the best race time with the current race time
                  up until where the best race loses
                - random mutation (only random after the gate that the best time loses)
        '''
        idx = self.get_curr_losing_idx()
        self.copy_curr_to_best_hyper(idx)
        self.copy_curr_to_best_time(idx)

        return idx

    def save_curr_time(self, curr_time_list):
        self.curr_hyper.time = np.array(curr_time_list)


    def curr_win(self):
        assert (not self.best_hyper.time is None), "best time in not initialized"
        return self.curr_hyper.time[-1] < self.best_hyper.time[-1]


    def copy_curr_to_best_hyper(self, idx=-1):
        if idx == -1:   # the current race wins
            self.best_hyper = copy.deepcopy(self.curr_hyper) 
        else:
            self.best_hyper.v[:idx] = self.curr_hyper.v[:idx]
            self.best_hyper.a[:idx] = self.curr_hyper.a[:idx]
            self.best_hyper.d[:idx] = self.curr_hyper.d[:idx]
        
        if idx == len(self.best_hyper.time) - 1:
            self.best_hyper.v[-1] = self.curr_hyper.v[-1]
            self.best_hyper.a[-1] = self.curr_hyper.a[-1]
            self.best_hyper.d[-1] = self.curr_hyper.d[-1]


    def copy_curr_to_best_time(self, idx=-1):
        if idx == -1:   # the current race wins
            self.best_hyper.time = copy.copy(self.curr_hyper.time)
        else:
            # also copy the last element, if wins until the end 
            self.best_hyper.time[:idx] = self.curr_hyper.time[:idx]
        
        if idx == len(self.best_hyper.time) - 1:   
            # also copy the last element, if wins until the end 
            self.best_hyper.time[-1] = self.curr_hyper.time[-1]


    
    def random_mutation_from_best(self, num_mutation, start_idx=0):
        '''
            if start_idx is specified from update_best_hyper(), it will reduce the search space
            because mutation is only allowed at the point after the gate that 
            the best race time lost to the current race time 
        '''
        new_hyper = copy.deepcopy(self.best_hyper)
        # end_idx = self.num_hyper - 1
        for _ in range(num_mutation):
            idx_1 = random.randint(0, 2)    # choose between v, a, or d
            idx_2 = random.randint(0, start_idx)
            
            print(f"random_idx_1 = {idx_1}, random_idx_2 = {idx_2}")

            if idx_1 == 0:
                new_hyper.random_mutation_v_at_idx(idx_2)
            elif idx_1 == 1:
                new_hyper.random_mutation_a_at_idx(idx_2)
            elif idx_2 == 2:
                new_hyper.random_mutation_d_at_idx(idx_2)

        new_hyper.d[-1] = 2 # to ensure that the drone passes the last gate
        
        return new_hyper


    def get_curr_losing_idx(self):
        '''
            In the case that the overall current race time lose to the best race time
            Try to find if the best race time actually lose to the current race time at some gate

            Winning condition:
                - the current race time must be less than the best race time >= 0.5 seconds
                - if the drone current race is better than the best race at ith gate,
                  the drone should also pass the (i+1)th gate. 
                    - This is because the performance of ith gate affect the performance of (i+1)th gate
                    - If it passes the ith gate too fast, it might not be able to pass (i+1)th gate at all
        ''' 
        idx = None
        for idx in range(self.num_hyper):
            if self.curr_hyper.time[idx] > self.best_hyper.time[idx] and \
                abs(self.best_hyper.time[idx] - self.curr_hyper.time[idx]) > .5:
                    break
            # else:
            #     # if idx != (self.num_hyper - 1) \
            #     #         and self.curr_hyper.time[idx + 1] != 1000: # need to pass the next gate
            #     #         break
            #     break

        out_idx = idx
        # if idx > 0 and idx != len(self.best_hyper.time) - 1:
        #     out_idx = idx - 1 
                    
        print(f"Lose at = {out_idx}")
        return out_idx              

if __name__ == "__main__":
    print("test hyOpt Operation")
    opt = hyOpt(6)
    opt.curr_hyper.init_hypers(15, 21, 4)
    opt.best_hyper.init_hypers(12, 20, 3)
    opt.curr_hyper.init_time()
    opt.best_hyper.init_time()
    opt.best_hyper.time = [4, 3, 7, 8, 9, 10]
    opt.curr_hyper.time = [20, 2, 0, 0, 0, 20]
    idx = opt.update_best_hyper()
    print(opt.best_hyper)


