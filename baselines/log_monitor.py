import glob
import os
import time

disqualified_racers = set()
finished_racers = set()
PATH_TO_LOG = '/home/jedsadakorn/git/AirSim_Training/AirSimExe/Saved/Logs/RaceLogs/'
class LogMonitor(object):
    def __init__(self, path_to_log=PATH_TO_LOG):
        self.path_to_log = path_to_log

    def open_file(self, path_to_log):
        list_of_files = glob.glob(path_to_log + '*.log')
        latest_file = max(list_of_files, key=os.path.getctime)
        # print("Opened file: " + latest_file)
        return open(latest_file, "r+")
    
    def skip_header(self, opened_file):
        opened_file.readline()
        opened_file.readline()
        opened_file.readline()
    
    def get_score_at_gate(self, gate_idx):
        '''
            get time used to passed gate_idx (time + penalty)
            NOTE: the gate idx in the log file starts from 1, but from 0 in the code
        '''
        counter = 0
        penalty = 0
        gate_passed_time = -1
        f = self.open_file(self.path_to_log)
        self.skip_header(f)
        for line in f:
            token = line.split()
            if token[-2] == "gates_passed" and token[0] == "drone_1" and token[-1] == gate_idx:
                gate_passed_time = int(token[2]) / 1000.
                counter = 1
            elif counter == 1 and token[-2] == "collision_count":   # there is collision
                counter = 2
            elif counter == 1 and not token[-2] == "collision_count":   # no collision
                f.close()
                return (gate_passed_time, penalty)
            elif counter == 2 and token[-2] == "penalty":
                penalty = int(token[-1]) / 1000.
                f.close()
                return (gate_passed_time, penalty)
            
            
        print ("    The drone did not pass the gate_idx " + gate_idx)
        return (1000, 0)


    def get_race_time(self, drone_name):
        f = self.open_file(self.path_to_log)
        self.skip_header(f)
        for line in f:
            token = line.split()
            
            if token[0] == drone_name and token[3] == "finished":
                # print("finish time", token[2])
                f.close()
                return token[2]
        
        assert(False), "Did not get the race time requested"
    

    def check_gate_missed(self):
        f = self.open_file(self.path_to_log)
        self.skip_header(f)
        for line in f:
            token = line.split()
            if token[-2] == "gates_missed":
                f.close()
                return True
        f.close()
        return False


    def check_collision(self):
        f = self.open_file(self.path_to_log)
        self.skip_header(f)
        for line in f:
            token = line.split()
            if token[-2] == "collision_count":
                f.close()
                return True
        f.close()
        return False

    def get_last_gate_idx_before_termination(self):
        gate_idx = 1
        f = self.open_file(self.path_to_log)
        self.skip_header(f)
        for line in f:
            token = line.split()
            if token[-2] == "gates_passed":
                gate_idx = token[-1]

        f.close()
        # print(f"gate_idx passed before termination {int(gate_idx) - 1}")

        return int(gate_idx) - 1

    def get_score(self, finish_time):
        '''
        score = (time, num_gates_passed, num_gates_missed, penalty)
        '''
        f = self.open_file(self.path_to_log)
        self.skip_header(f)
        num_gates_missed = 0
        for line in f:
            token = line.split()
            if token[2] == finish_time:
                
                if token[-2] == "gates_passed":
                    num_gates_passed = int(token[-1])
                elif token[-2] == "gates_missed":
                    num_gates_missed = int(token[-1])
                elif token[-2] == "penalty":
                    penalty = int(token[-1]) / 1000.
                elif token[-2] == "finished":
                    race_time = int(finish_time) / 1000.
                    # print(time, num_gates_passed, num_gates_missed, penalty)
                    # return the score
                    f.close()
                    return (race_time, num_gates_passed, num_gates_missed, penalty)
                else:
                    continue

        assert(False), "Did not get the score requested"
                
    def get_current_race_time(self):
        f = self.open_file(self.path_to_log)
        self.skip_header(f)
        for line in f:
            token = line.split()
            if not len(token) == 5:
                break
            current_race_time = token[2]

        f.close()
        return int(current_race_time) / 1000.

    def read_log(self):
        finish_time = self.get_race_time("drone_1")
        return self.get_score(finish_time)
        # for line in f:
        #     print(line.split())
        # for line in self.follow(f):
        #     print("process lines")
        #     self.process(line)

if __name__ == "__main__":
    log_monitor = LogMonitor()
    print(log_monitor.get_current_race_time())