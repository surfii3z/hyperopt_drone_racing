FILE_PATH = "/home/usrg/god_ws/hyperopt_ea_game_of_drones/baselines/BO04.txt"

''' 
Print the best time in the log file
'''

with open(FILE_PATH) as f:
    for line in f:
        token = line.split()
        if len(token) > 0 and token[0] == 'best:':
            print(token[-1][:-1])

f.closed