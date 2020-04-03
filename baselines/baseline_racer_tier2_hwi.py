from argparse import ArgumentParser
import airsimneurips as airsim

import threading
import time
import datetime
import utils
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import signal

import sys
import cv2

## gate detection
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# # from log_monitor import next_gate_idx
# import log_monitor

MODEL_NAME = 'inference_graph'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

NUM_CLASSES = 1

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    #sess = tf.Session(graph=detection_graph,config=tf.ConfigProto(gpu_options=gpu_options))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=detection_graph,config=config)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# drone_name should match the name in ~/Document/AirSim/settings.json
class BaselineRacer(object):
    def __init__(self, drone_name = "drone_1", viz_traj=True, viz_traj_color_rgba=[1.0, 0.0, 0.0, 1.0], viz_image_cv2=True):
        self.drone_name = drone_name
        self.gate_poses_ground_truth = None
        self.viz_image_cv2 = viz_image_cv2
        self.viz_traj = viz_traj
        self.viz_traj_color_rgba = viz_traj_color_rgba

        self.airsim_client = airsim.MultirotorClient()
        self.airsim_client.confirmConnection()
        # we need two airsim MultirotorClient objects because the comm lib we use (rpclib) is not thread safe
        # so we poll images in a thread using one airsim MultirotorClient object
        # and use another airsim MultirotorClient for querying state commands
        self.airsim_client_images = airsim.MultirotorClient()
        self.airsim_client_images.confirmConnection()
        self.airsim_client_odom = airsim.MultirotorClient()
        self.airsim_client_odom.confirmConnection()
        self.level_name = None

        self.image_callback_thread = threading.Thread(target=self.repeat_timer_image_callback, args=(self.image_callback, 0.03))
        self.odometry_callback_thread = threading.Thread(target=self.repeat_timer_odometry_callback, args=(self.odometry_callback, 1))
        self.control_thread = threading.Thread(target=self.repeat_timer_control_callback, args=(self.control_callback, 0.02))
        self.is_image_thread_active = False
        self.is_odometry_thread_active = False
        self.is_control_thread_active = False
        self.is_log_thread_active = False

        self.MAX_NUMBER_OF_GETOBJECTPOSE_TRIALS = 10 # see https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing/issues/38

        self.img_mutex = threading.Lock()

        ###################gate detection result variables#################
        self.W = 0
        self.H = 0
        self.Mx = 0
        self.My = 0
        self.detect_flag = False
        self.detect_big_gate = False
        self.gate6_go_ahead = False
        self.gate10_go_ahead = False
        self.gate13_go_ahead = False
        self.gate15_go_ahead = False

        

        ################### isPassing function variable ####################
        self.next_gate_idx = 0
        self.gate_passed_thresh = 1
        self.pass_cnt = 0

        self.check_pass = 1

        ################### using moveToPosition function variable ###########
        self.chk_first_flag = True
        self.cannot_detect_count = 0
        self.using_moveToPosition_threshold = 10
        self.gate_back_dis = 5
        self.lam = 0.5 # 0 ~ 1
        self.lam_z = 0.95

        self.vision_lam = 0.3 # low value is big ocsilation

        self.pass_position_vec = None
        self.pass_position_ori = None

        self.estimate_depth = 8
        self.position_control_on = False

        self.prev_time = 0
        self.curr_time = 0
        self.distance_y_prev = 0
        self.distance_y_curr = 0
        self.distance_z_prev = 0
        self.distance_z_curr = 0
        self.desired_yaw_prev = 0
        self.desired_yaw_curr = 0

        self.prev_vel = 0
        ############################## TaeYeon method ###############################
        self.duration_move_ahead = 1



        ############################## Record Parameter Tuning for Generation #####################
        # self.f = open("./Curve_fit/tuning.txt", "w")

        ############################ Save img ########################
        self.count = 0
        self.day = datetime.datetime.today().day
        self.minute = datetime.datetime.today().minute
        self.hour = datetime.datetime.today().hour


    def fly_through_all_gates_at_once_with_moveOnSpline(self):
        # if self.level_name in ["Soccer_Field_Medium", "Soccer_Field_Easy", "ZhangJiaJie_Medium"] :
        #     vel_max = 30.0
        #     acc_max = 15.0

        if self.level_name == "Building99_Hard":
            vel_max = 4.0
            acc_max = 1.0
        else:
            vel_max = 30.0
            acc_max = 15.0

        return self.airsim_client.moveOnSplineAsync([gate_pose.position for gate_pose in self.gate_poses_ground_truth], vel_max=30.0, acc_max=15.0, 
            add_position_constraint=True, add_velocity_constraint=False, viz_traj=self.viz_traj, vehicle_name=self.drone_name)


    def get_ground_truth_gate_poses(self):
        gate_names_sorted_bad = sorted(self.airsim_client.simListSceneObjects("Gate.*"))
        # gate_names_sorted_bad is of the form `GateN_GARBAGE`. for example:
        # ['Gate0', 'Gate10_21', 'Gate11_23', 'Gate1_3', 'Gate2_5', 'Gate3_7', 'Gate4_9', 'Gate5_11', 'Gate6_13', 'Gate7_15', 'Gate8_17', 'Gate9_19']
        # we sort them by their ibdex of occurence along the race track(N), and ignore the unreal garbage number after the underscore(GARBAGE)
        gate_indices_bad = [int(gate_name.split('_')[0][4:]) for gate_name in gate_names_sorted_bad]
        gate_indices_correct = sorted(range(len(gate_indices_bad)), key=lambda k:gate_indices_bad[k])
        gate_names_sorted = [gate_names_sorted_bad[gate_idx] for gate_idx in gate_indices_correct]
        self.gate_poses_ground_truth = [self.airsim_client.simGetObjectPose(gate_name) for gate_name in gate_names_sorted]

    # loads desired level
    def load_level(self, level_name, sleep_sec = 2.0):
        self.level_name = level_name
        self.airsim_client.simLoadLevel(self.level_name)
        self.airsim_client.confirmConnection() # failsafe
        time.sleep(sleep_sec) # let the environment load completely

    # Starts an instance of a race in your given level, if valid
    def start_race(self, tier=3):
        self.airsim_client.simStartRace(tier)

    # Resets a current race: moves players to start positions, timer and penalties reset
    def reset_race(self):
        self.airsim_client.simResetRace()

    # arms drone, enable APIs, set default traj tracker gains
    def initialize_drone(self):
        self.airsim_client.enableApiControl(vehicle_name=self.drone_name)
        self.airsim_client.arm(vehicle_name=self.drone_name)

        # set default values for trajectory tracker gains
        traj_tracker_gains = airsim.TrajectoryTrackerGains(kp_cross_track = 5.0, kd_cross_track = 0.0,
                                                            kp_vel_cross_track = 3.0, kd_vel_cross_track = 0.0,
                                                            kp_along_track = 0.4, kd_along_track = 0.0,
                                                            kp_vel_along_track = 0.04, kd_vel_along_track = 0.0,
                                                            kp_z_track = 2.0, kd_z_track = 0.0,
                                                            kp_vel_z = 0.4, kd_vel_z = 0.0,
                                                            kp_yaw = 3.0, kd_yaw = 0.1)

        self.airsim_client.setTrajectoryTrackerGains(traj_tracker_gains, vehicle_name=self.drone_name)
        time.sleep(0.2)

    def takeoffAsync(self):
        self.airsim_client.takeoffAsync().join()

    # like takeoffAsync(), but with moveOnSpline()
    def takeoff_with_moveOnSpline(self, takeoff_height = 1.0):
        start_position = self.airsim_client.simGetVehiclePose(vehicle_name=self.drone_name).position
        takeoff_waypoint = airsim.Vector3r(start_position.x_val, start_position.y_val, start_position.z_val-takeoff_height)

        self.pass_position_vec = takeoff_waypoint

        self.airsim_client.moveOnSplineAsync([takeoff_waypoint], vel_max=15.0, acc_max=5.0, add_position_constraint=True, add_velocity_constraint=False,
            add_acceleration_constraint=False, viz_traj=self.viz_traj, viz_traj_color_rgba=self.viz_traj_color_rgba, vehicle_name=self.drone_name).join()

    # this is utility function to get a velocity constraint which can be passed to moveOnSplineVelConstraints()
    # the "scale" parameter scales the gate facing vector accordingly, thereby dictating the speed of the velocity constraint
    def get_gate_facing_vector_from_quaternion(self, airsim_quat, scale = 1.0):
        import numpy as np
        # convert gate quaternion to rotation matrix.
        # ref: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion; https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
        q = np.array([airsim_quat.w_val, airsim_quat.x_val, airsim_quat.y_val, airsim_quat.z_val], dtype=np.float64)
        n = np.dot(q, q)
        if n < np.finfo(float).eps:
            return airsim.Vector3r(0.0, 1.0, 0.0)
        q *= np.sqrt(2.0 / n)
        q = np.outer(q, q)
        rotation_matrix = np.array([[1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
                                    [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
                                    [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]])
        gate_facing_vector = rotation_matrix[:,1]
        return airsim.Vector3r(scale * gate_facing_vector[0], scale * gate_facing_vector[1], scale * gate_facing_vector[2])

    def get_world_frame_vel_from_drone_frame_vel(self, airsim_quat, velocity):
           import numpy as np
           # convert gate quaternion to rotation matrix.
           # ref: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion; https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
           q = np.array([airsim_quat.w_val, airsim_quat.x_val, airsim_quat.y_val, airsim_quat.z_val], dtype=np.float64)
           n = np.dot(q, q)
           if n < np.finfo(float).eps:
               return airsim.Vector3r(0.0, 1.0, 0.0)
           q *= np.sqrt(2.0 / n)
           q = np.outer(q, q)
           rotation_matrix = np.array([[1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
                                       [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
                                       [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]])
           drone_frame_vel_array = np.array([velocity.x_val, velocity.y_val, velocity.z_val])
           world_vel = np.matmul(rotation_matrix, drone_frame_vel_array)
           return airsim.Vector3r(world_vel[0], world_vel[1], world_vel[2])


    def get_drone_frame_vector_from_world_frame_vector(self, airsim_quat, vec):
       import numpy as np
       q = np.array([airsim_quat.w_val, airsim_quat.x_val, airsim_quat.y_val, airsim_quat.z_val], dtype=np.float64)
       n = np.dot(q, q)
       if n < np.finfo(float).eps:
           return airsim.Vector3r(0.0, 1.0, 0.0)
       q *= np.sqrt(2.0 / n)
       q = np.outer(q, q)
       rotation_matrix = np.array([[1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
                                   [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
                                   [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]])
       rotation_matrix = np.linalg.inv(rotation_matrix)
       world_vec = np.array([vec.x_val, vec.y_val, vec.z_val])
       drone_vec = np.matmul(rotation_matrix, world_vec)
       return airsim.Vector3r(drone_vec[0], drone_vec[1], drone_vec[2])


    def isPassGate(self):
        gate_passed_thresh = 3

        if(self.estimate_depth < gate_passed_thresh and self.next_gate_idx < len(self.gate_poses_ground_truth)-1):
            curr_position = self.airsim_client.simGetVehiclePose(vehicle_name = "drone_1").position

            dist_from_next_gate = math.sqrt( (curr_position.x_val - self.gate_poses_ground_truth[self.next_gate_idx].position.x_val)**2
                                            + (curr_position.y_val - self.gate_poses_ground_truth[self.next_gate_idx].position.y_val)**2
                                            + (curr_position.z_val- self.gate_poses_ground_truth[self.next_gate_idx].position.z_val)**2)
            
            if dist_from_next_gate < 18 :
                self.next_gate_idx += 1
                self.pass_position_vec = curr_position
                self.pass_position_ori = self.airsim_client.simGetVehiclePose("drone_1").orientation
                self.check_pass = 1
                return True

        return False


    def isPassGate_v2(self):
        gate_passed_thresh = 3

        if(self.estimate_depth < gate_passed_thresh and self.next_gate_idx < len(self.gate_poses_ground_truth)-1):
            curr_position = self.airsim_client.simGetVehiclePose(vehicle_name = "drone_1").position

            dist_from_next_gate = math.sqrt( (curr_position.x_val - self.gate_poses_ground_truth[self.next_gate_idx].position.x_val)**2
                                            + (curr_position.y_val - self.gate_poses_ground_truth[self.next_gate_idx].position.y_val)**2
                                            + (curr_position.z_val- self.gate_poses_ground_truth[self.next_gate_idx].position.z_val)**2)
            
            if dist_from_next_gate < 5 :
                self.next_gate_idx += 1
                self.pass_position_vec = curr_position
                self.pass_position_ori = self.airsim_client.simGetVehiclePose("drone_1").orientation
                self.check_pass = 1
                return True

        return False


    def get_lam_point(self, a, b, lam):
        lam_a = list(map(lambda x: x*lam, a))
        lam_b = list(map(lambda x: x*(1-lam), b))

        return [lam_a[0] + lam_b[0], lam_a[1] + lam_b[1], lam_a[2] + lam_b[2]]

    def gate_detection(self, img_rgb):
        with self.img_mutex:
            #### gate detection
            frame_expanded = np.expand_dims(img_rgb, axis=0)
            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})
            index = np.squeeze(scores >= 0.90)
            boxes_detected = np.squeeze(boxes)[index]   # only interested in the bounding boxes that show detection
            # Draw the results of the detection (aka 'visualize the results')
            N = len(boxes_detected)
            H_list = []
            W_list = []
            if N >= 1:  # in the case of more than one gates are detected, we want to select the nearest gate (biggest bounding box)
                for element in boxes_detected:
                    H_list.append(element[2] - element[0])
                    W_list.append(element[3] - element[1])
                if N > 1:
                    # print('boxes_detected', boxes_detected, boxes_detected.shape)
                    Area = np.array(H_list) * np.array(W_list)
                    max_Area = np.max(Area)
                    idx_max = np.where(Area == max_Area)[0][0]  # find where the maximum area is
                    # print(Area)
                else:
                    idx_max = 0
                box_of_interest = boxes_detected[idx_max]
                h_box = box_of_interest[2]-box_of_interest[0]
                w_box = box_of_interest[3]-box_of_interest[1]
                Area_box = h_box * w_box
                # if N > 1:
                #     print('box_of_interest', box_of_interest, box_of_interest.shape)
                #     print('----------------------------------')
                if Area_box <= 0.98 and Area_box >= 0.01:    # Feel free to change this number, set to 0 if don't want this effect
                    # If we detect the box but it's still to far keep the same control command
                    # This is to prevent the drone to track the next gate when it has not pass the current gate yet
                    self.detect_flag = True
                    self.H = box_of_interest[2]-box_of_interest[0]
                    self.W = box_of_interest[3]-box_of_interest[1]
                    self.My = (box_of_interest[2]+box_of_interest[0])/2
                    self.Mx = (box_of_interest[3]+box_of_interest[1])/2
                    #print("boxes_detected : ", boxes_detected, "W : ", self.W, "H", self.H, "M : ", self.Mx, " ", self.My)
                else:
                    self.detect_flag = False
                    if self.next_gate_idx == 13:
                        self.detect_big_gate = True
                    # print("Area_box", Area_box)
                #     print("=============== NOT DETECT ===============")
            else:
                # print('==================== set detect_flag to FALSE ====================')
                self.estimate_depth = 8
                self.detect_flag = False

            vis_util.visualize_boxes_and_labels_on_image_array(
                img_rgb,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.90)


    ###############################################
    ### detect gate and velocity code by localryu ###
    ###############################################

    def control_callback(self):
        Dist = 3.2
        if (self.next_gate_idx < 3):
            velocity_x_o = 6
            Dist = -3
        elif (self.next_gate_idx == 3):
            velocity_x_o = 6
            Dist = 1
        elif(self.next_gate_idx == 4): # the small gate
            velocity_x_o = 5.4
            Dist = 1
        elif(self.next_gate_idx == 5):
            velocity_x_o = 5.5
            Dist = -1
        elif(self.next_gate_idx == 6): # the wall
            velocity_x_o = 8   
            Dist = -2
        elif(self.next_gate_idx == 7 or self.next_gate_idx == 8): # behind the tree
            velocity_x_o = 7
            Dist = -3
        elif(self.next_gate_idx == 9): # before the long one
            velocity_x_o = 7
            Dist = 1
        # elif(self.next_gate_idx == 10):
        #     velocity_x_o = 7
        #     Dist = 1
        elif(self.next_gate_idx == 12): # the small gate before the big gate
            velocity_x_o = 6
            Dist = 1
        elif(self.next_gate_idx == 13): # the big gate
            velocity_x_o = 12
            Dist = -3
        elif(self.next_gate_idx == 18):
            velocity_x_o = 3.5
        elif (self.next_gate_idx == 19):
            Dist = 3.5
            velocity_x_o = 3
        elif(self.next_gate_idx == 20): 
            Dist = -2
            velocity_x_o = 6
        else:
            velocity_x_o = 6 # 3.8
        duration = 0.05
        
        
        last_gate_idx = self.airsim_client.simGetLastGatePassed("drone_1")
        if(last_gate_idx < 30):
            if(last_gate_idx == self.next_gate_idx):
                self.pass_position_vec = self.airsim_client.simGetVehiclePose(vehicle_name = "drone_1").position
                self.next_gate_idx = last_gate_idx + 1
                print(str(self.next_gate_idx) + " PASS")
            elif(self.next_gate_idx == 13 and self.isPassGate()):
                print("using isPassGate()")
                self.pass_position_vec = self.airsim_client.simGetVehiclePose(vehicle_name = "drone_1").position
                print(str(self.next_gate_idx) + " PASS")
                drone_heading_vec = self.get_world_frame_vel_from_drone_frame_vel(self.airsim_client.simGetVehiclePose("drone_1").orientation, airsim.Vector3r(13, 0, 0))
                # self.airsim_client.moveByVelocityAsync(drone_heading_vec.x_val, drone_heading_vec.y_val, drone_heading_vec.z_val - 3, duration = 1).join()
            elif(self.next_gate_idx == 17  and self.isPassGate()):
                print("using isPassGate() gate 17: Go ahead")
                self.pass_position_vec = self.airsim_client.simGetVehiclePose(vehicle_name = "drone_1").position
                print(str(self.next_gate_idx) + " PASS")
                drone_heading_vec = self.get_world_frame_vel_from_drone_frame_vel(self.airsim_client.simGetVehiclePose("drone_1").orientation, airsim.Vector3r(6, 0, 0))
                self.airsim_client.moveByVelocityAsync(drone_heading_vec.x_val, drone_heading_vec.y_val, drone_heading_vec.z_val, duration = 1).join()
            elif(self.next_gate_idx == 20 and self.isPassGate()):
                print("using isPassGate()")
                self.pass_position_vec = self.airsim_client.simGetVehiclePose(vehicle_name = "drone_1").position
                print(str(self.next_gate_idx) + " PASS")
                drone_heading_vec = self.get_world_frame_vel_from_drone_frame_vel(self.airsim_client.simGetVehiclePose("drone_1").orientation, airsim.Vector3r(2, 0, 0))
                self.airsim_client.moveByVelocityAsync(drone_heading_vec.x_val, drone_heading_vec.y_val, drone_heading_vec.z_val, duration = 2).join()
            elif(self.next_gate_idx == 19 and self.isPassGate()):
                print("using isPassGate()")
                self.pass_position_vec = self.airsim_client.simGetVehiclePose(vehicle_name = "drone_1").position
                print(str(self.next_gate_idx) + " PASS")
            elif(self.next_gate_idx == 5 and self.isPassGate()):
                print("using isPassGate()")
                self.pass_position_vec = self.airsim_client.simGetVehiclePose(vehicle_name = "drone_1").position
                print(str(self.next_gate_idx) + " PASS")
            elif(self.next_gate_idx == 7 and self.gate6_go_ahead == False):
                print("gate 6:  go ahead")
                self.pass_position_vec = self.airsim_client.simGetVehiclePose(vehicle_name = "drone_1").position
                drone_heading_vec = self.get_world_frame_vel_from_drone_frame_vel(self.airsim_client.simGetVehiclePose("drone_1").orientation, airsim.Vector3r(12, 0, 0))
                self.airsim_client.moveByVelocityAsync(drone_heading_vec.x_val, drone_heading_vec.y_val, drone_heading_vec.z_val + 1.2, duration = 2).join()
                self.gate6_go_ahead = True
                print(str(self.next_gate_idx) + " PASS")

            
        
        if self.detect_flag :
            with self.img_mutex:
                # velocity_x_o = ((1-self.vision_lam) * velocity_x_o + self.vision_lam * self.prev_vel)

                #print("detection control")
                ####################model######################
                ## box_width : real_gate_width = u : distance_y
                ## distance_y = 1.5 * u / box_width
                self.curr_time = time.time()
                param_A = 13.401
                param_B = -1.976
                self.estimate_depth = param_A * np.exp(param_B * self.W)
                #############Control Gain param################
                velocity_x = velocity_x_o + (self.estimate_depth - Dist)*0.187 - abs(self.distance_y_curr) * 0.2  #- abs(self.distance_z_curr) * 0.1
                
                Vely_P_Gain = velocity_x*0.62 # 0.47 is good when vel 3.8 #0.9 is good when vel = 3 #0.43
                Vely_D_Gain = velocity_x*0.17
                Velz_P_Gain = velocity_x*0.62 # 0.49 is good when vel 3.8 #0.9 is good when vel = 3 #0.43
                Velz_D_Gain = velocity_x*0.09
                Yaw_P_Gain = velocity_x*0.7 #1.5 is good when vel = 3
                Yaw_D_Gain = 0.07



                ##################Get Error####################
                Dead_zone = 0.025
                if(self.next_gate_idx == 20):      
                    self.distance_y_curr =  (1.5 * (self.Mx - 0.50) / self.W)
                    self.distance_z_curr =  (1.5 * (self.My - 0.40) / self.H)
                    if abs((self.Mx - 0.50)) < Dead_zone:
                        self.distance_y_curr = 0
                    if abs((self.My - 0.42) ) < Dead_zone:
                        self.distance_z_curr = 0

                elif(self.next_gate_idx == 19):
                    self.distance_y_curr =  (1.5 * (self.Mx - 0.80) / self.W)
                    self.distance_z_curr =  (1.5 * (self.My - 0.42) / self.H)
                    if abs((self.Mx - 0.80)) < Dead_zone:
                        self.distance_y_curr = 0
                    if abs((self.My - 0.42) ) < Dead_zone:
                        self.distance_z_curr = 0
                
                elif(self.next_gate_idx == 4): # the small gate
                    self.distance_y_curr =  (1.5 * (self.Mx - 0.55) / self.W)
                    self.distance_z_curr =  (1.5 * (self.My - 0.50) / self.H)
                    if abs((self.Mx - 0.55)) < Dead_zone:
                        self.distance_y_curr = 0
                    if abs((self.My - 0.50) ) < Dead_zone:
                        self.distance_z_curr = 0
                
                elif(self.next_gate_idx == 13): # the big gate
                    self.distance_y_curr =  (1.5 * (self.Mx - 0.50) / self.W)
                    self.distance_z_curr =  (1.5 * (self.My - 0.7) / self.H)
                    if abs((self.Mx - 0.50)) < Dead_zone:
                        self.distance_y_curr = 0
                    if abs((self.My - 0.7) ) < Dead_zone:
                        self.distance_z_curr = 0

                elif(self.next_gate_idx == 15): # the big curve after the big gate
                    self.distance_y_curr =  (1.5 * (self.Mx - 0.55) / self.W)
                    self.distance_z_curr =  (1.5 * (self.My - 0.50) / self.H)
                    if abs((self.Mx - 0.55)) < Dead_zone:
                        self.distance_y_curr = 0
                    if abs((self.My - 0.50) ) < Dead_zone:
                        self.distance_z_curr = 0

                else:
                    self.distance_y_curr =  (1.5 * (self.Mx - 0.5) / self.W)
                    self.distance_z_curr =  (1.5 * (self.My - 0.48) / self.H)
                    if abs((self.Mx - 0.50)) < Dead_zone:
                        self.distance_y_curr = 0
                    if abs((self.My - 0.48) ) < Dead_zone:
                        self.distance_z_curr = 0

                self.desired_yaw_curr = math.atan2(self.distance_y_curr, self.estimate_depth) *57.2859
                
                # print(self.desired_yaw_curr)

                if self.chk_first_flag:
                    cmd_vel_x = velocity_x#velocity_x*(1+0.1*(self.estimate_depth-6))
                    cmd_vel_y = self.distance_y_curr*Vely_P_Gain
                    cmd_vel_z = self.distance_z_curr*Velz_P_Gain
                    cmd_yaw = self.desired_yaw_curr*Yaw_P_Gain
                    self.chk_first_flag = False
                else:
                    dt = self.curr_time - self.prev_time
                    df_distance_y = (self.distance_y_curr - self.distance_y_prev)/dt
                    df_distance_z = (self.distance_z_curr - self.distance_z_prev)/dt
                    df_yaw = (self.desired_yaw_curr - self.desired_yaw_prev)/dt

                    #print("[df_y, df] | ", df_distance_y, dt)

                    cmd_vel_x = velocity_x#velocity_x*(1+0.1*(self.estimate_depth-6))
                    cmd_vel_y = self.distance_y_curr*Vely_P_Gain + df_distance_y*Vely_D_Gain
                    cmd_vel_z = self.distance_z_curr*Velz_P_Gain + df_distance_z*Velz_D_Gain
                    cmd_yaw = self.desired_yaw_curr*Yaw_P_Gain + df_yaw*Yaw_D_Gain

                velocity_mag = math.sqrt(cmd_vel_x**2 + cmd_vel_y**2 + cmd_vel_z**2)

                velocity_gain = ((1-self.vision_lam) * velocity_mag + self.vision_lam * self.prev_vel) / velocity_mag
                cmd_vel_x *= velocity_gain
                cmd_vel_y *= velocity_gain
                cmd_vel_z *= velocity_gain

                velocity_vector_drone = airsim.Vector3r(cmd_vel_x, cmd_vel_y, cmd_vel_z)
                v_pose = self.airsim_client.simGetVehiclePose(vehicle_name="drone_1")
                velocity_vector_world = self.get_world_frame_vel_from_drone_frame_vel(v_pose.orientation, velocity_vector_drone)

                ####################Do Control#################
                # print("                                                     ", cmd_yaw)

                self.airsim_client.moveByVelocityAsync(velocity_vector_world.x_val, velocity_vector_world.y_val, velocity_vector_world.z_val, duration = duration, yaw_mode=airsim.YawMode(True,cmd_yaw)).join()
                #print("[E_Depth Y_vel Z_vel Yaw velocity_x] : ", self.estimate_depth, " | ", cmd_vel_y, " | ", cmd_vel_z, " | ", desired_yaw_curr, " | ", velocity_x)
                # if(self.next_gate_idx > 18):
                #     print("[Depth disY desiredYaw] : ", self.estimate_depth, " | ", self.distance_y_curr, " | ", self.desired_yaw_curr)
                #     #print("[E_Depth] : ", self.estimate_depth)

                ################update variables###############
                self.prev_time = self.curr_time
                self.distance_y_prev = self.distance_y_curr
                self.distance_z_prev = self.distance_z_curr
                self.desired_yaw_prev = self.desired_yaw_curr
                self.position_control_on = False
                self.prev_vel = math.sqrt(velocity_vector_world.x_val ** 2 + velocity_vector_world.y_val ** 2 + velocity_vector_world.z_val ** 2)
        elif(self.next_gate_idx < len(self.gate_poses_ground_truth)):
            ################################################
            # get approximate yaw vector
            ##############################################
            gate_pose = self.gate_poses_ground_truth[self.next_gate_idx].position
            gate_ori = self.gate_poses_ground_truth[self.next_gate_idx].orientation

            # next_gate_pose = self.gate_poses_ground_truth[self.next_gate_idx + 1].position
            # next_gate_ori = self.gate_poses_ground_truth[self.next_gate_idx +1].orientation

            drone_pose = self.airsim_client.simGetVehiclePose("drone_1").position
            drone_ori = self.airsim_client.simGetVehiclePose("drone_1").orientation

            error_pose_vec = gate_pose - drone_pose

            # print(error_pose_vec)

            error_pose_vec_in_drone = self.get_drone_frame_vector_from_world_frame_vector(drone_ori, error_pose_vec)
            mag_error_pose_vec = math.sqrt(error_pose_vec_in_drone.x_val ** 2 + error_pose_vec_in_drone.y_val ** 2)
            angle = math.acos(error_pose_vec_in_drone.x_val / mag_error_pose_vec)*57.2859
            if (error_pose_vec_in_drone.y_val < 0):
                angle = -angle
            ################################################
            # get approximate yaw vector
            ##############################################

            gate_pose_array = [gate_pose.x_val, gate_pose.y_val, gate_pose.z_val]
            drone_pose_array = [drone_pose.x_val, drone_pose.y_val, drone_pose.z_val]
            # next_gate_pose_array = [next_gate_pose.x_val, next_gate_pose.y_val, next_gate_pose.z_val]

            target_lambda_pose = self.get_lam_point(gate_pose_array, drone_pose_array, self.lam)
            target_lambda_z_pose = self.get_lam_point(gate_pose_array, drone_pose_array, self.lam_z)
            
            ## ground z val = 3.5
            if(target_lambda_z_pose[2] > -1):
                target_lambda_z_pose[2] = -1
            
            target_pose = gate_pose_array
            if(target_pose[2] > -1):
                target_pose[2] = -1

            # target_lambda_pose = gate_pose_array
            # next_target_pose = next_gate_pose_array

            # vel_error = self.pass_position_vec - airsim.Vector3r(target_lambda_pose[0], target_lambda_pose[1], target_pose[2])
            
            vel_error = airsim.Vector3r(target_lambda_pose[0], target_lambda_pose[1], target_pose[2]) - drone_pose
            vel_mag = math.sqrt(vel_error.x_val ** 2 + vel_error.y_val ** 2 + vel_error.z_val ** 2)

            vel_fun = 0.80 * vel_mag + (-0.2) * (abs(angle) / vel_mag) #0.543
            
            #########################################################

            drone_vel_for_start = airsim.Vector3r(6,0,0)
            world_vel_for_start = self.get_world_frame_vel_from_drone_frame_vel(drone_ori, drone_vel_for_start)

            #########################################################

            # print(drone_pose.z_val)

            if(self.next_gate_idx == 7): # the gate behind the tree 
                vel_fun = 0.75 * vel_mag + (-0.2) * (abs(angle) / vel_mag) #0.543
                self.airsim_client.moveToPositionAsync(target_lambda_pose[0], target_lambda_pose[1], target_lambda_z_pose[2], velocity = vel_fun, yaw_mode=airsim.YawMode(True,angle))
            # elif(self.next_gate_idx == 10):
            #     vel_fun = 0.3 * vel_mag + (-0.2) * (abs(angle) / vel_mag) #0.543
            #     self.airsim_client.moveToPositionAsync(target_lambda_pose[0], target_lambda_pose[1], target_lambda_z_pose[2], velocity = vel_fun, yaw_mode=airsim.YawMode(True,angle))
            elif(self.next_gate_idx == 0): # start gate
                vel_fun = 0.9 * vel_mag + (-0.2) * (abs(angle) / vel_mag) #0.543
                self.airsim_client.moveToPositionAsync(target_lambda_pose[0], target_lambda_pose[1], self.pass_position_vec.z_val - 1, velocity = vel_fun, yaw_mode=airsim.YawMode(True,angle))
            elif(self.next_gate_idx == 4):
                vel_fun = 0.9 * vel_mag + (-0.2) * (abs(angle) / vel_mag) #0.543
                self.airsim_client.moveToPositionAsync(target_lambda_pose[0], target_lambda_pose[1] - 1.5, target_pose[2], velocity = vel_fun, yaw_mode=airsim.YawMode(True,angle))
            elif(self.next_gate_idx == 8 or self.next_gate_idx == 9):
                vel_fun = 0.8 * vel_mag + (-0.2) * (abs(angle) / vel_mag) #0.543
                self.airsim_client.moveToPositionAsync(target_lambda_pose[0], target_lambda_pose[1], target_pose[2] + 1.5, velocity = vel_fun, yaw_mode=airsim.YawMode(True,angle))
            elif(self.next_gate_idx == 13):
                vel_fun = 0.7 * vel_mag + (-0.2) * (abs(angle) / vel_mag) #0.543
                self.airsim_client.moveToPositionAsync(target_lambda_pose[0], target_lambda_pose[1], target_pose[2], velocity = vel_fun, yaw_mode=airsim.YawMode(True,angle))
            elif(self.next_gate_idx == 14):
                vel_fun = 0.8 * vel_mag + (-0.2) * (abs(angle) / vel_mag) #0.543
                self.airsim_client.moveToPositionAsync(target_lambda_pose[0], target_lambda_pose[1] - 1, target_pose[2], velocity = vel_fun, yaw_mode=airsim.YawMode(True,angle))
            elif(self.next_gate_idx == 15):
                vel_fun = 0.9 * vel_mag + (-0.2) * (abs(angle) / vel_mag) #0.543
                self.airsim_client.moveToPositionAsync(target_lambda_pose[0], target_lambda_pose[1] - 1, target_pose[2] + 5, velocity = vel_fun, yaw_mode=airsim.YawMode(True,angle))

            elif(self.next_gate_idx == 20):
                vel_fun = 0.5 * vel_mag + (-0.2) * (abs(angle) / vel_mag) #0.543
                self.airsim_client.moveToPositionAsync(target_lambda_pose[0], target_lambda_pose[1], target_pose[2] + 3, velocity = vel_fun, yaw_mode=airsim.YawMode(True,angle))
            else:
                self.airsim_client.moveToPositionAsync(target_lambda_pose[0], target_lambda_pose[1], target_lambda_z_pose[2], velocity = vel_fun, yaw_mode=airsim.YawMode(True,angle))

            self.prev_vel = vel_fun


    def image_callback(self):
        #print("image callback")
        # get uncompressed fpv cam image
        request = [airsim.ImageRequest("fpv_cam", airsim.ImageType.Scene, False, False)]
        response = self.airsim_client_images.simGetImages(request)
        img_rgb_1d = np.fromstring(response[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3)
        
        path = "./picture_sc/picture_sc_" + str(self.day) + "_" + str(self.hour) + "_" + str(self.minute) + "/"
        if not(os.path.isdir(path)):
            os.makedirs(os.path.join(path))

        if(self.count % 3 == 0):
            cv2.imwrite(path+str(self.count) + ".jpg", img_rgb)
        self.count += 1
        self.gate_detection(img_rgb)

        if self.viz_image_cv2:
            cv2.imshow('Object detector', img_rgb)
            cv2.waitKey(1)




    def odometry_callback(self):
        # get uncompressed fpv cam image
        drone_state = self.airsim_client_odom.getMultirotorState()
        # in world frame:
        position = drone_state.kinematics_estimated.position
        orientation = drone_state.kinematics_estimated.orientation
        linear_velocity = drone_state.kinematics_estimated.linear_velocity
        angular_velocity = drone_state.kinematics_estimated.angular_velocity

    # call task() method every "period" seconds.
    def repeat_timer_image_callback(self, task, period):
        while self.is_image_thread_active:
            task()
            time.sleep(period)

    def repeat_timer_odometry_callback(self, task, period):
        while self.is_odometry_thread_active:
            task()
            time.sleep(period)

    def repeat_timer_control_callback(self, task, period):
        while self.is_control_thread_active:
            task()
            time.sleep(period)

    def repeat_timer_log_callback(self, task, period):
        while self.is_log_thread_active:
            task()
            time.sleep(period)

    def start_image_callback_thread(self):
        if not self.is_image_thread_active:
            self.is_image_thread_active = True
            self.image_callback_thread.start()
            print("Started image callback thread")

    def stop_image_callback_thread(self):
        if self.is_image_thread_active:
            self.is_image_thread_active = False
            self.image_callback_thread.join()
            print("Stopped image callback thread.")

    def start_odometry_callback_thread(self):
        if not self.is_odometry_thread_active:
            self.is_odometry_thread_active = True
            self.odometry_callback_thread.start()
            print("Started odometry callback thread")

    def stop_odometry_callback_thread(self):
        if self.is_odometry_thread_active:
            self.is_odometry_thread_active = False
            self.odometry_callback_thread.join()
            print("Stopped odometry callback thread.")

    def start_control_thread(self):
        if not self.is_control_thread_active:
            self.is_control_thread_active = True
            self.control_thread.start()
            print("Started control callback thread")

    def stop_control_thread(self):
        if self.is_control_thread_active:
            self.is_control_thread_active = False
            self.control_thread.join()
            print("Stopped control callback thread.")

    

def main(args):
    # ensure you have generated the neurips planning settings file by running python generate_settings_file.py
    baseline_racer = BaselineRacer(drone_name="drone_1", viz_traj=args.viz_traj, viz_traj_color_rgba=[1.0, 1.0, 0.0, 1.0], viz_image_cv2=args.viz_image_cv2)
    baseline_racer.load_level(args.level_name)
    if args.level_name == "Final_Tier_1":
        args.race_tier = 1
    if args.level_name == "Final_Tier_2":
        args.race_tier = 2
    if args.level_name == "Final_Tier_3":
        args.race_tier = 3
    baseline_racer.start_race(args.race_tier)
    baseline_racer.initialize_drone()
    baseline_racer.takeoff_with_moveOnSpline()
    baseline_racer.get_ground_truth_gate_poses()

    baseline_racer.start_image_callback_thread()
    baseline_racer.start_odometry_callback_thread()

    # baseline_racer.fly_through_all_gates_at_once_with_moveOnSpline().join()
    baseline_racer.start_control_thread()


    if(baseline_racer.next_gate_idx == 20 and baseline_racer.isPassGate()):
        print('TERMINATE PROGRAM')
        baseline_racer.stop_image_callback_thread()
        baseline_racer.stop_control_thread()
        baseline_racer.reset_race()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--level_name', type=str, choices=["Soccer_Field_Easy", "Soccer_Field_Medium", "ZhangJiaJie_Medium", "Building99_Hard", 'Final_Tier_1', 'Final_Tier_2', 'Final_Tier_3'], default="Soccer_Field_Medium")
    parser.add_argument('--enable_plot_transform', dest='plot_transform', action='store_true', default=False)
    parser.add_argument('--enable_viz_traj', dest='viz_traj', action='store_true', default=False)
    parser.add_argument('--enable_viz_image_cv2', dest='viz_image_cv2', action='store_true', default=True)
    parser.add_argument('--race_tier', type=int, choices=[1,2,3], default=1)
    args = parser.parse_args()
    main(args)
