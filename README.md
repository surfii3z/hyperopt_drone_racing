# Hyperparamters Optimization in game of drones
The approach is inspired by the winner (the report is available on the official website). It is the improve version of genetic algorithm (though)
# Prerequisite
1) Install game of drones binaries following the instructions from the official website

[airsim_neurips2019](https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing)

2) Install tensorflow object detection API (used for gate detection)

[tensorflow 1 object detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1.md)

3) python >= 3.6
# Run
Open two terminals
1) for running airsim binaries
``` bash
cd /path/to/AirSim_Qualification
./AirSimExe.sh -windowed -opengl4
```
2) for running hyperparameter optimization
``` bash
cd /path/to/hyperopt_ea_game_of_drones/baselines
python baseline_racer_baseline_GA.py
```
# Result
https://docs.google.com/presentation/d/15Ji3PlZ-SBxpDjpm1BdEuDm6ERNtocofC-S8EgZiPAM/edit?usp=sharing

# Example of hyperparameters found by each algorithm after 200 iterations
```
Map: ZhangJiajie_Medium
Number of gates: 14
```
Random search (remains the same after 400 iterations) @ 46.21 seconds
```
v: [14.69, 34.88, 26.15, 16.1, 21.91, 25.59, 29.13, 22.2, 33.3, 25.78, 13.66, 18.41, 16.63, 22.34]
a: [54.54, 34.82, 59.93, 30.88, 140.68, 146.64, 28.58, 23.0, 79.18, 53.38, 137.22, 116.31, 86.32, 103.68]
d: [4.3, 3.95, 4.31, 5.6, 4.43, 3.9, 6.2, 4.28, 4.66, 3.55, 5.36, 6.39, 5.49, 5.05]
```

EA @ 45.09 seconds
```
v: [12.0, 25.81, 24.49, 12.0, 29.9, 12.0, 32.04, 31.94, 12.0, 12.0, 12.0, 12.0, 31.17, 12.0]
a: [125.93, 110.36, 154.52, 61.49, 50.0, 124.79, 72.61, 93.77, 121.4, 117.18, 50.0, 102.14, 91.32, 50.0]
d: [3.5, 3.5, 5.08, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 2.0]
```

GD @ 41.60 seconds
```
v: [34.25, 30.08, 24.74, 19.58, 16.93, 26.46, 34.29, 24.49, 23.31, 13.39, 27.8, 12.0, 12.0, 12.0]
a: [156.32, 85.44, 130.84, 157.3, 109.55, 107.35, 107.77, 50.0, 116.54, 59.15, 77.03, 129.87, 55.24, 60.44]
d: [3.5, 3.5, 4.8, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 2.0]
```

PO @ 44.00 seconds
```
v: [18.69, 15.76, 24.38, 22.32, 31.52, 26.01, 34.3, 11.54, 20.36, 32.44, 12.0, 12.0, 12.0, 11.64]
a: [120.07, 101.66, 110.75, 152.02, 146.16, 154.36, 99.39, 133.41, 102.7, 99.54, 146.46, 50.0, 50.0, 133.33]
d: [3.5, 3.5, 5.64, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 2.0]
```

BO @ 40.04 seconds
```
v: [22.59, 32.36, 18.66, 16.79, 16.94, 27.48, 34.08, 17.88, 28.13, 14.13, 15.66, 32.25, 26.14, 23.5]
a: [151.75, 57.78, 91.54, 131.0, 58.94, 152.9, 132.63, 113.68, 91.82, 97.83, 137.93, 68.08, 153.08, 42.96]
d: [4.21, 5.61, 4.4, 5.85, 4.42, 5.25, 3.69, 6.3, 3.9, 5.64, 4.48, 4.69, 4.42, 2.0]
```



	
