from argparse import ArgumentParser
import airsimneurips as airsim
import cv2
import threading
import time
import utils
import numpy as np
import math
from baseline_racer_planner import BaselineRacer

baseline_racer = BaselineRacer(drone_name="drone_1", viz_image_cv2=True, viz_traj=True, viz_traj_color_rgba=[1.0, 1.0, 0.0, 1.0])
baseline_racer.load_level("Soccer_Field_Easy")
baseline_racer.start_image_callback_thread()
baseline_racer.get_ground_truth_gate_poses()
baseline_racer.initialize_drone()
baseline_racer.start_race(1)
baseline_racer.takeoff_with_moveOnSpline()
baseline_racer.start_odometry_callback_thread()

