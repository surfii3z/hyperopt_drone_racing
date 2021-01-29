FILE_PATH = "/home/usrg/god_ws/hyperopt_ea_game_of_drones/baselines/logging_results/BO_no_slower_01230.txt"

''' 
Print the best time in the log file
'''

def retrieve_best(file_path):
    # Determine number of iteration
    last_iteration = 0
    with open(file_path) as f:
        for line in f:
            token = line.split()
            if len(token) > 0 and token[0] == 'iteration:':
                last_iteration = token[-1]

    f.closed
    if last_iteration==0:
        return last_iteration, None, None, None, None

    with open(file_path) as f:
        for line in f:
            token = line.split()
            if len(token) > 0 and token[0] == 'iteration:' and token[-1] == last_iteration:
                import json
                best_time = f.readline().strip()[6:]
                best_time = json.loads(best_time)

                f.readline().strip() # current_time
                f.readline().strip() # current_hyper_parameter
                f.readline().strip() # curr_v
                f.readline().strip() # curr_a
                f.readline().strip() # curr_d
                f.readline().strip() # BEST_hyper_parameter
                
                best_v = f.readline().strip()[3:]
                best_v = json.loads(best_v)
                best_a = f.readline().strip()[3:]
                best_a = json.loads(best_a)
                best_d = f.readline().strip()[3:]
                best_d = json.loads(best_d)
            else:
                continue
    f.closed

    return int(last_iteration), best_time, best_v, best_a, best_d

if __name__=='__main__':
    last_iter, best_time, best_v, best_a, best_d = retrieve_best(FILE_PATH)
    print(last_iter)
    print(best_time)
    print(best_v)
    print(best_a)
    print(best_d)