from argparse import ArgumentParser
import airsimneurips as airsim
import cv2
import threading
import time
import utils
import numpy as np
import math
import os
import copy
import random
import log_monitor
import hyOpt_po as hyOpt

## for gate detection
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

'''                                  MAP:  Qual_Tier_2
    Gate idx 0 1    2      3 4         5                    6        7 8 9 10 11 12 13
                 going up         big left turn --------- big gate
'''

## Hyperparamters range
V_MIN = 8.5
V_MAX = 35
A_MIN = 20
A_MAX = 160
D_MIN = 3.5
D_MAX = 6.5

FINISH_GATE_IDX = 13

## for gate detection

MODEL_NAME = 'inference_graph'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

NUM_CLASSES = 1

## Load the label map.

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
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    sess = tf.Session(graph=detection_graph, config=config)

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


def L2_distance(l1, l2):
    ''' l1 = list1, l2 = list2
    '''
    return math.sqrt((l1[0] - l2[0])**2 + (l1[1] - l2[1])**2 + (l1[2] - l2[2])**2)

def L2_norm(l):
    ''' l = list
    '''
    return math.sqrt((l[0])**2 + (l[1])**2 + (l[2])**2)

def convex_combination(vec1, vec2, eta):
    ''' 0 <= eta <= 1, indicating how close it is to the vec2
        e.g. eta = 1, vec_result = vec2
        eta = 0, vec_result = vec1
    '''
    vec_result = airsim.Vector3r(0,0,0)
    vec_result.x_val = (1-eta) * vec1.x_val + eta * vec2.x_val
    vec_result.y_val = (1-eta) * vec1.y_val + eta * vec2.y_val
    vec_result.z_val = (1-eta) * vec1.z_val + eta * vec2.z_val
    return vec_result

# drone_name should match the name in ~/Document/AirSim/settings.json
class BaselineRacer(object):
    def __init__(self, drone_name="drone_1", viz_traj=True, viz_traj_color_rgba=[1.0, 0.0, 0.0, 1.0], viz_image_cv2=True):
        ## gate idx trackers
        self.last_gate_passed_idx = -1
        self.last_gate_idx_moveOnSpline_was_called_on = -1
        self.next_gate_idx = 0
        self.next_next_gate_idx = 1

        self.drone_name = drone_name
        self.gate_poses_ground_truth = None
        self.gate_object_names_sorted = None
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
        self.is_image_thread_active = False

        self.got_odom = False
        self.odometry_callback_thread = threading.Thread(target=self.repeat_timer_odometry_callback, args=(self.odometry_callback, 0.5))
        self.is_odometry_thread_active = False

        self.MAX_NUMBER_OF_GETOBJECTPOSE_TRIALS = 10 # see https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing/issues/383
        self.finished_race = False
        self.terminated_program = False

        ################### gate detection result variables #################
        self.img_mutex = threading.Lock()
        self.W = 0
        self.H = 0
        self.Mx = 0
        self.My = 0
        self.detect_flag = False
        self.previous_detect_flag = False

        ################# Hyper-parameter Optimization#####################
        self.hyper_opt = hyOpt.hyOpt(FINISH_GATE_IDX + 1)
        self.hyper_opt.best_hyper.set_v_range((V_MIN, V_MAX))
        self.hyper_opt.best_hyper.set_a_range((A_MIN, A_MAX))
        self.hyper_opt.best_hyper.set_d_range((D_MIN, D_MAX))
        self.hyper_opt.best_hyper.init_hypers(12, 50, 3.5)
        self.hyper_opt.best_hyper.init_time()
        self.use_new_hyper_for_next_race(self.hyper_opt.best_hyper)

        # if the simulation crashes, continue from last iteration by putting best hyperparameters here
        self.hyper_opt.best_hyper.v = np.array([23.36, 11.3, 32.19, 22.46, 13.37, 25.65, 12.0, 27.25, 25.33, 12.0, 12.0, 12.0, 21.09, 30.56])
        self.hyper_opt.best_hyper.a = np.array([70.71, 117.34, 146.34, 91.64, 43.16, 99.92, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
        self.hyper_opt.best_hyper.d = np.array([3.5, 3.5, 6.14, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 2.0])
        self.hyper_opt.best_hyper.time = np.array([6.08, 8.01, 10.5, 12.76, 15.22, 18.07, 33.86, 37.66, 39.83, 42.57, 44.57, 47.46, 49.97, 52.58])
        self.logging_file_name = "po5_2.txt"
        self.iteration = 1
    

    # loads desired level
    def load_level(self, level_name, sleep_sec=2.0):
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

    def dummy_reset(self):
        self.airsim_client.reset()
        time.sleep(0.1)
        self.airsim_client.simResetRace()
        time.sleep(0.1)
        self.airsim_client.enableApiControl(vehicle_name=self.drone_name)
        self.airsim_client.arm(vehicle_name=self.drone_name)


    # arms drone, enable APIs, set default traj tracker gains
    def initialize_drone(self):
        self.airsim_client.enableApiControl(vehicle_name=self.drone_name)
        self.airsim_client.arm(vehicle_name=self.drone_name)

        traj_tracker_gains = airsim.TrajectoryTrackerGains(kp_cross_track=11.0, kd_cross_track=4.0,
                                                            kp_vel_cross_track=3.0, kd_vel_cross_track=0.0,
                                                            kp_along_track=0.4, kd_along_track=0.0,
                                                            kp_vel_along_track=0.04, kd_vel_along_track=0.0,
                                                            kp_z_track=8.3, kd_z_track=3.5,
                                                            kp_vel_z=3, kd_vel_z=0.8,
                                                            kp_yaw=3.0, kd_yaw=0.5)



        self.airsim_client.setTrajectoryTrackerGains(traj_tracker_gains, vehicle_name=self.drone_name)
        time.sleep(0.2)
    
    def initialize_drone_hyper_parameter(self, hyper_parameter):
        self.curr_hyper = copy.deepcopy(hyper_parameter)

    def reset_drone_parameter(self):
        # gate idx trackers
        self.last_gate_passed_idx = -1
        self.last_gate_idx_moveOnSpline_was_called_on = -1
        self.next_gate_idx = 0
        self.next_next_gate_idx = 1

        self.finished_race = False
        self.terminated_program = False

    def takeoffAsync(self):
        self.airsim_client.takeoffAsync().join()

    # like takeoffAsync(), but with moveOnSpline()
    def takeoff_with_moveOnSpline(self, takeoff_height=1.0):
        start_position = self.airsim_client.simGetVehiclePose(vehicle_name=self.drone_name).position
        takeoff_waypoint = airsim.Vector3r(start_position.x_val, start_position.y_val, start_position.z_val-takeoff_height)

        self.airsim_client.moveOnSplineAsync([takeoff_waypoint], vel_max=15.0, acc_max=5.0, add_position_constraint=True, add_velocity_constraint=False, 
            add_acceleration_constraint=False, viz_traj=self.viz_traj, viz_traj_color_rgba=self.viz_traj_color_rgba, vehicle_name=self.drone_name).join()

    # stores gate ground truth poses as a list of airsim.Pose() objects in self.gate_poses_ground_truth
    def get_ground_truth_gate_poses(self):
        gate_names_sorted_bad = sorted(self.airsim_client.simListSceneObjects("Gate.*"))
        # gate_names_sorted_bad is of the form `GateN_GARBAGE`. for example:
        # ['Gate0', 'Gate10_21', 'Gate11_23', 'Gate1_3', 'Gate2_5', 'Gate3_7', 'Gate4_9', 'Gate5_11', 'Gate6_13', 'Gate7_15', 'Gate8_17', 'Gate9_19']
        # we sort them by their ibdex of occurence along the race track(N), and ignore the unreal garbage number after the underscore(GARBAGE)
        gate_indices_bad = [int(gate_name.split('_')[0][4:]) for gate_name in gate_names_sorted_bad]
        gate_indices_correct = sorted(range(len(gate_indices_bad)), key=lambda k: gate_indices_bad[k])
        self.gate_object_names_sorted = [gate_names_sorted_bad[gate_idx] for gate_idx in gate_indices_correct]
        self.n_gate = len(self.gate_object_names_sorted)
        self.gate_poses_ground_truth = []
        for gate_name in self.gate_object_names_sorted:
            curr_pose = self.airsim_client.simGetObjectPose(gate_name)
            counter = 0
            while (math.isnan(curr_pose.position.x_val) or math.isnan(curr_pose.position.y_val) or math.isnan(curr_pose.position.z_val)) and (counter < self.MAX_NUMBER_OF_GETOBJECTPOSE_TRIALS):
                print(f"DEBUG: {gate_name} position is nan, retrying...")
                counter += 1
                curr_pose = self.airsim_client.simGetObjectPose(gate_name)
            assert not math.isnan(curr_pose.position.x_val), f"ERROR: {gate_name} curr_pose.position.x_val is still {curr_pose.position.x_val} after {counter} trials"
            assert not math.isnan(curr_pose.position.y_val), f"ERROR: {gate_name} curr_pose.position.y_val is still {curr_pose.position.y_val} after {counter} trials"
            assert not math.isnan(curr_pose.position.z_val), f"ERROR: {gate_name} curr_pose.position.z_val is still {curr_pose.position.z_val} after {counter} trials"
            self.gate_poses_ground_truth.append(curr_pose)

    # this is utility function to get a velocity constraint which can be passed to moveOnSplineVelConstraints()
    # the "scale" parameter scales the gate facing vector accordingly, thereby dictating the speed of the velocity constraint
    def get_gate_facing_vector_from_quaternion(self, airsim_quat, scale = 1.0):
        import numpy as np
        # convert gate quaternion to rotation matrix
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

    def is_gate_passed(self):
        self.next_gate_xyz = [self.gate_poses_ground_truth[self.next_gate_idx].position.x_val,
                              self.gate_poses_ground_truth[self.next_gate_idx].position.y_val,
                              self.gate_poses_ground_truth[self.next_gate_idx].position.z_val]
        dist_from_next_gate = L2_distance(self.curr_xyz, self.next_gate_xyz)
        if dist_from_next_gate < self.hyper_opt.curr_hyper.d[self.next_gate_idx]:
            return True
        else:
            return False
    
    def update_gate_idx_trackers(self):
        self.last_gate_passed_idx += 1
        self.next_gate_idx += 1
        self.next_next_gate_idx += 1
        # print("Update next_gate_idx to %d" % self.next_gate_idx)
    
    def is_race_finished(self):
        '''                                                             Soccer_Field_Medium
            Gate idx 0 1 2   3      4  5  6  7  8   9  10   11               12               13        14       15 16 17 18 19 20   21     22 23 24
                           curved        down             far right     b4 big turn left    mid-air  sharp down                    sharp up
        '''
        terminate_condition = self.last_gate_passed_idx == FINISH_GATE_IDX
        if terminate_condition:
            print("     FINISH the race")
        return terminate_condition

    def is_drone_stucked(self):
        early_terminate_condition = L2_norm(self.curr_lin_vel) < 0.05
        if early_terminate_condition:
            print("     EARLY TERMINATION: drone stucked")
        return early_terminate_condition

    def is_slower_than_last_race(self):
        early_terminate_condition = log_monitor.get_current_race_time() > self.hyper_opt.best_hyper.time[-1]
        if early_terminate_condition:
            print("     EARLY TERMINATION: slower than the best racorded time")
        return early_terminate_condition

    def is_drone_missed_some_gate(self):
        early_terminate_condition = log_monitor.check_gate_missed()
        if early_terminate_condition:
            print("     EARLY TERMINATION: drone missed some gate")
        return early_terminate_condition

    def is_drone_collied(self):
        early_terminate_condition = log_monitor.check_collision()
        if early_terminate_condition:
            print("     EARLY TERMINATION: drone collided")
        return early_terminate_condition

    def gate_detection(self, img_rgb):
        THRESHOULD = 0.97
        with self.img_mutex:
            #### gate detection
            frame_expanded = np.expand_dims(img_rgb, axis=0)
            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})
            index = np.squeeze(scores >= THRESHOULD)
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
            else:
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
                min_score_thresh=THRESHOULD)

    def image_callback(self):
        # get uncompressed fpv cam image
        request = [airsim.ImageRequest("fpv_cam", airsim.ImageType.Scene, False, False)]
        response = self.airsim_client_images.simGetImages(request)
        img_rgb_1d = np.fromstring(response[0].image_data_uint8, dtype=np.uint8) 
        img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3)
        self.gate_detection(img_rgb)

        if self.viz_image_cv2:
            cv2.imshow("img_rgb", img_rgb)
            cv2.waitKey(1)

    
    def odometry_callback(self):
        # in world frame:
        self.drone_state = self.airsim_client_odom.getMultirotorState()
        drone_position = self.drone_state.kinematics_estimated.position
        drone_velocity = self.drone_state.kinematics_estimated.linear_velocity
        self.curr_lin_vel = [drone_velocity.x_val, drone_velocity.y_val, drone_velocity.z_val]
        self.curr_xyz = [drone_position.x_val, drone_position.y_val, drone_position.z_val]
        self.got_odom = True

        if (self.finished_race == False):
            if self.is_gate_passed():
                self.update_gate_idx_trackers()
            
            # condition to terminate the race
            if self.is_race_finished() or \
               self.is_drone_stucked() or \
               self.is_slower_than_last_race() or \
               self.is_drone_missed_some_gate():
            #    self.is_drone_collied():
                self.finished_race = True
                time.sleep(1.0)
                self.airsim_client.moveByVelocityAsync(0, 0, 0, 2).join()   # stop the drone

                return

            ''' Control Part'''
            if (self.detect_flag == True):
                ''' Go to the center of the gate'''
                # self.airsim_client.cancelLastTask()
                self.fly_to_next_gate_with_moveOnSpline()
            else:
                ''' Go to prior of the gate'''
                noisy_position_of_next_gate = self.gate_poses_ground_truth[self.next_gate_idx].position
                target_position = convex_combination(drone_position, noisy_position_of_next_gate, 0.5)

                self.fly_to_next_point_with_moveOnSpline(target_position)

        elif (self.finished_race == True and L2_norm(self.curr_lin_vel) < 0.5):
            # race is finished
            
            self.finished_race == False
            self.terminated_program = True
            time.sleep(0.5)
            
            temp = [log_monitor.get_score_at_gate(str(i)) for i in range(1, FINISH_GATE_IDX + 2)]
            current_race_time = [round(score[0] + score[1], 2) for score in temp]

            print(f"best: {self.hyper_opt.best_hyper.time.tolist()}")
            print(f"curr: {current_race_time}")

            self.hyper_opt.save_curr_time(current_race_time)
            self.dummy_reset()
            self.race_again()
        else:
            pass

    def log_curr_iter_data(self):
        ## For the current iteration, write the best/current hyperparameter/time to the text file
        data_logging.write(f"\niteration: {self.iteration}\n")
        data_logging.flush()
        data_logging.write(f"best: {self.hyper_opt.best_hyper.time.tolist()}\n")
        data_logging.flush()
        data_logging.write(f"time: {self.hyper_opt.curr_hyper.time.tolist()}\n")
        data_logging.flush()

        data_logging.write(f"current_hyper_parameter\n")
        data_logging.flush()
        data_logging.write(f"v: {self.hyper_opt.curr_hyper.v.tolist()}\n")
        data_logging.flush()
        data_logging.write(f"a: {self.hyper_opt.curr_hyper.a.tolist()}\n")
        data_logging.flush()
        data_logging.write(f"d: {self.hyper_opt.curr_hyper.d.tolist()}\n")
        data_logging.flush()

        data_logging.write(f"BEST_hyper_parameter\n")
        data_logging.flush()
        data_logging.write(f"v: {self.hyper_opt.best_hyper.v.tolist()}\n")
        data_logging.flush()
        data_logging.write(f"a: {self.hyper_opt.best_hyper.a.tolist()}\n")
        data_logging.flush()
        data_logging.write(f"d: {self.hyper_opt.best_hyper.d.tolist()}\n")
        data_logging.flush()

    def get_new_hyper_for_next_race(self):
        idx = self.hyper_opt.update_best_hyper()
        if (idx == -1):
            new_hyper = self.hyper_opt.random_mutation_from_best(num_mutation=2)
        else:
            new_hyper = self.hyper_opt.random_mutation_from_best(num_mutation=2, end_idx=idx)
        
        return new_hyper

    def use_new_hyper_for_next_race(self, new_hyper):
        self.hyper_opt.curr_hyper = copy.deepcopy(new_hyper)

    def race_again(self):
        # data logging
        self.log_curr_iter_data()

        new_hyper = self.get_new_hyper_for_next_race()
        self.use_new_hyper_for_next_race(new_hyper)

        self.iteration = self.iteration + 1
        self.reset_drone_parameter()
        self.start_race(1)

        print(f"================ iteration: {self.iteration} ================")
        self.airsim_client.disableApiControl(vehicle_name="drone_2")
        self.airsim_client.disarm(vehicle_name="drone_2")
        self.takeoff_with_moveOnSpline()
        self.get_ground_truth_gate_poses()


    def fly_to_next_gate_with_moveOnSpline(self):
        # print(self.gate_poses_ground_truth[self.next_gate_idx].position)
        return self.airsim_client.moveOnSplineAsync([self.gate_poses_ground_truth[self.next_gate_idx].position], 
                                                    vel_max=self.hyper_opt.curr_hyper.v[self.next_gate_idx],
                                                    acc_max=self.hyper_opt.curr_hyper.a[self.next_gate_idx], 
                                                    add_position_constraint=True, 
                                                    add_velocity_constraint=True, 
                                                    add_acceleration_constraint=True, 
                                                    replan_from_lookahead=False,
                                                    viz_traj=self.viz_traj, 
                                                    viz_traj_color_rgba=self.viz_traj_color_rgba, 
                                                    vehicle_name=self.drone_name)


    def fly_to_next_point_with_moveOnSpline(self, point):
        # print(self.gate_poses_ground_truth[self.next_gate_idx].position)
        return self.airsim_client.moveOnSplineAsync([point],
                                                    vel_max=self.hyper_opt.curr_hyper.v[self.next_gate_idx],
                                                    acc_max=self.hyper_opt.curr_hyper.a[self.next_gate_idx], 
                                                    add_position_constraint=True, 
                                                    add_velocity_constraint=True, 
                                                    add_acceleration_constraint=True, 
                                                    replan_from_lookahead=False,
                                                    viz_traj=self.viz_traj, 
                                                    viz_traj_color_rgba=self.viz_traj_color_rgba, 
                                                    vehicle_name=self.drone_name)

    # call task() method every "period" seconds.
    def repeat_timer_image_callback(self, task, period):
        while self.is_image_thread_active:
            task()
            time.sleep(period)

    def repeat_timer_odometry_callback(self, task, period):
        while self.is_odometry_thread_active:
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
            # self.odometry_callback_thread.join()
            print("Stopped odometry callback thread.")


def main(args):
    # ensure you have generated the neurips planning settings file by running python generate_settings_file.py
    
    baseline_racer.load_level(args.level_name)
    baseline_racer.start_image_callback_thread()
    baseline_racer.start_race(args.race_tier)
    baseline_racer.get_ground_truth_gate_poses()
    baseline_racer.initialize_drone()
    
    # don't want opponent drone
    baseline_racer.airsim_client.disableApiControl(vehicle_name="drone_2")
    baseline_racer.airsim_client.disarm(vehicle_name="drone_2")
    # hyper parameter initialization

    # baseline_racer.initialize_drone_hyper_parameter(baseline_racer.hyper_opt.best_hyper)
    
    baseline_racer.takeoff_with_moveOnSpline()
    baseline_racer.start_odometry_callback_thread()

    print(f"================ iteration: 1 ================")



if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--level_name', type=str, choices=["Soccer_Field_Easy", "Soccer_Field_Medium", "ZhangJiaJie_Medium", "Building99_Hard", 
        "Qualifier_Tier_1", "Qualifier_Tier_2", "Qualifier_Tier_3", "Final_Tier_1", "Final_Tier_2", "Final_Tier_3"], default="Qualifier_Tier_2")
    parser.add_argument('--planning_baseline_type', type=str, choices=["all_gates_at_once","all_gates_one_by_one"], default="all_gates_at_once")
    parser.add_argument('--planning_and_control_api', type=str, choices=["moveOnSpline", "moveOnSplineVelConstraints"], default="moveOnSpline")
    parser.add_argument('--enable_viz_traj', dest='viz_traj', action='store_true', default=False)
    parser.add_argument('--enable_viz_image_cv2', dest='viz_image_cv2', action='store_true', default=True)
    parser.add_argument('--race_tier', type=int, choices=[1,2,3], default=1)
    args = parser.parse_args()
    baseline_racer = BaselineRacer(drone_name="drone_1", viz_traj=args.viz_traj, viz_traj_color_rgba=[1.0, 1.0, 1.0, 1.0], viz_image_cv2=args.viz_image_cv2)
    log_monitor = log_monitor.LogMonitor()
    data_logging = open(baseline_racer.logging_file_name, "a")

    main(args)
