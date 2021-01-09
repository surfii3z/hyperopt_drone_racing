import baseline_racer_BO as baseline_racer
import optuna
import hyOpt
import numpy as np
import copy
import joblib

BEST_TIME = 1000

ITERATION = 1
NUM_HYPER = 14

V_MIN = 8.5
V_MAX = 35
A_MIN = 20
A_MAX = 160
D_MIN = 3.5
D_MAX = 6.5

best_hyper_parameters = hyOpt.HyperParameter(NUM_HYPER)
best_hyper_parameters.init_hypers(12, 50, 3.5)
best_hyper_parameters.init_time()

hyper_parameters = hyOpt.HyperParameter(NUM_HYPER)
hyper_parameters.init_hypers(12, 50, 3.5)
hyper_parameters.init_time()
NAME = 'BO_no_slower_00'
save_to_file_name = "{}.txt".format(NAME)
data_logging = open(save_to_file_name, "a")

def objective(trial):
    global ITERATION, hyper_parameters, best_hyper_parameters, BEST_TIME
    print(f"================ iteration: {ITERATION} ================")

    
    # print(hyper_parameters)

    if ITERATION == 1:
        race_time_list = baseline_racer.main(best_time=BEST_TIME)
    else:
        for i in range(NUM_HYPER):
            hyper_parameters.v[i] = np.round(trial.suggest_uniform('v_{}'.format(i), V_MIN, V_MAX), 2)
            hyper_parameters.a[i] = np.round(trial.suggest_uniform('a_{}'.format(i), A_MIN, A_MAX), 2)
            hyper_parameters.d[i] = np.round(trial.suggest_uniform('d_{}'.format(i), D_MIN, D_MAX), 2)

        hyper_parameters.d[-1] = 2  # to ensure that it finishes the race
        race_time_list = baseline_racer.main(hyper_parameters, best_time=BEST_TIME)

    hyper_parameters.time = np.array(race_time_list)

    log_data()

    if race_time_list[-1] <= BEST_TIME:
        best_hyper_parameters = copy.deepcopy(hyper_parameters)
        BEST_TIME = race_time_list[-1]
    else:
        pass
    
    


    ITERATION += 1
    return race_time_list[-1]

def log_data():
    data_logging.write(f"\niteration: {ITERATION}\n")
    data_logging.flush()
    data_logging.write(f"best: {best_hyper_parameters.time.tolist()}\n")
    data_logging.flush()
    data_logging.write(f"time: {hyper_parameters.time.tolist()}\n")
    data_logging.flush()

    data_logging.write(f"current_hyper_parameter\n")
    data_logging.flush()
    data_logging.write(f"v: {hyper_parameters.v.tolist()}\n")
    data_logging.flush()
    data_logging.write(f"a: {hyper_parameters.a.tolist()}\n")
    data_logging.flush()
    data_logging.write(f"d: {hyper_parameters.d.tolist()}\n")
    data_logging.flush()

    data_logging.write(f"BEST_hyper_parameter\n")
    data_logging.flush()
    data_logging.write(f"v: {best_hyper_parameters.v.tolist()}\n")
    data_logging.flush()
    data_logging.write(f"a: {best_hyper_parameters.a.tolist()}\n")
    data_logging.flush()
    data_logging.write(f"d: {best_hyper_parameters.d.tolist()}\n")
    data_logging.flush()

def retrieve_best_param(study):
    for i in range(NUM_HYPER):
        best_hyper_parameters.v[i] = np.round(study.best_params['v_{}'.format(i)], 2)
        best_hyper_parameters.a[i] = np.round(study.best_params['a_{}'.format(i)], 2)
        best_hyper_parameters.d[i] = np.round(study.best_params['d_{}'.format(i)], 2)
    
    best_hyper_parameters.time[-1] = np.round(study.best_value, 2)
    return best_hyper_parameters


if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=NAME, storage='sqlite:///{}.db'.format(NAME), load_if_exists=True)
    if len(study.trials) > 0:
        ITERATION = study.trials[-1]._trial_id
        BEST_TIME = study.best_value
        best_hyper_parameters = retrieve_best_param(study)
        

    study.optimize(objective, n_trials=1000)

    print(study.best_params)  # E.g. {'x': 2.002108042}