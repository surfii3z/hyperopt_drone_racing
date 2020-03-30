from argparse import ArgumentParser
import airsimneurips as airsim
import cv2
import threading
import time
import utils
import numpy as np
import math
import os

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

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
    def __init__(self, drone_name = "drone_1", viz_traj=True, viz_traj_color_rgba=[1.0, 0.0, 0.0, 1.0], viz_image_cv2=True):
        # gate idx trackers
        self.last_gate_passed_idx = -1
        self.last_gate_idx_moveOnSpline_was_called_on = -1
        self.next_gate_idx = 0

        self.last_point_idx_moveOnSpline_was_called_on = -1
        self.next_gate_idx = 0

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

        ###################gate detection result variables#################
        self.img_mutex = threading.Lock()
        self.W = 0
        self.H = 0
        self.Mx = 0
        self.My = 0
        self.detect_flag = False
        self.previous_detect_flag = False

    # loads desired level
    def load_level(self, level_name, sleep_sec = 2.0):
        self.level_name = level_name
        self.airsim_client.simLoadLevel(self.level_name)
        time.sleep(2)
        
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
        n_gate = len(self.gate_poses_ground_truth)
        self.vel_max = np.ones(n_gate) * 30.0 * 3
        self.acc_max = np.ones(n_gate) * 15.0 * 3
        self.dist = np.ones(n_gate) * 3        

        # set default values for trajectory tracker gains 
        traj_tracker_gains = airsim.TrajectoryTrackerGains(kp_cross_track=5.0, kd_cross_track=1.0, 
                                                            kp_vel_cross_track=3.0, kd_vel_cross_track=0.0, 
                                                            kp_along_track=0.4, kd_along_track=0.0, 
                                                            kp_vel_along_track=0.04, kd_vel_along_track=0.0, 
                                                            kp_z_track=2.0, kd_z_track=0.0, 
                                                            kp_vel_z=0.4, kd_vel_z=0.0, 
                                                            kp_yaw=3.0, kd_yaw=0.1)

        self.airsim_client.setTrajectoryTrackerGains(traj_tracker_gains, vehicle_name=self.drone_name)
        time.sleep(0.2)
    
    def reset_drone_parameter(self):
        # gate idx trackers
        self.last_gate_passed_idx = -1
        self.last_gate_idx_moveOnSpline_was_called_on = -1
        self.next_gate_idx = 0

        self.finished_race = False
        self.terminated_program = False

    def takeoffAsync(self):
        self.airsim_client.takeoffAsync().join()

    # like takeoffAsync(), but with moveOnSpline()
    def takeoff_with_moveOnSpline(self, takeoff_height = 1.0):
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

    def is_gate_passed(self):
        self.next_gate_xyz = [self.gate_poses_ground_truth[self.next_gate_idx].position.x_val, 
                              self.gate_poses_ground_truth[self.next_gate_idx].position.y_val,
                              self.gate_poses_ground_truth[self.next_gate_idx].position.z_val]
        dist_from_next_gate = L2_distance(self.curr_xyz, self.next_gate_xyz)
        if dist_from_next_gate < self.dist[self.next_gate_idx]:
            return True
        else:
            return False
    
    def update_gate_idx_trackers(self):
        self.last_gate_passed_idx += 1
        self.next_gate_idx += 1
        print("Update next_gate_idx to %d" % self.next_gate_idx)
    
    def is_race_finished(self):
        # is the last gate in the track passed
        return (self.last_gate_passed_idx == len(self.gate_poses_ground_truth)-1)

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
            
            if self.is_race_finished():
                self.finished_race = True
                return

            # if not(self.last_gate_idx_moveOnSpline_was_called_on == self.next_gate_idx) and not self.finished_race:
            #     self.fly_to_next_gate_with_moveOnSpline()
            #     self.last_gate_idx_moveOnSpline_was_called_on = self.next_gate_idx

            ''' Control Part'''
            if (self.detect_flag == True):
                ''' Go to the center of the gate'''
                self.airsim_client.cancelLastTask()
                self.fly_to_next_gate_with_moveOnSpline()
            else:
                ''' Go to prior of the gate'''
                noisy_position_of_next_gate = self.gate_poses_ground_truth[self.next_gate_idx].position
                target_position = convex_combination(drone_position, noisy_position_of_next_gate, 0.5)

                if self.last_gate_passed_idx == -1:
                    if (self.last_gate_idx_moveOnSpline_was_called_on == -1):
                        self.airsim_client.cancelLastTask()
                        self.fly_to_first_point_with_moveOnSpline(target_position)
                        self.last_gate_idx_moveOnSpline_was_called_on = 0
                        return
                else:
                    self.fly_to_next_point_with_moveOnSpline(target_position)

                
            # in case of collision, the drone will stop
            # if L2_norm(self.curr_lin_vel) < 0.1 and self.last_gate_idx_moveOnSpline_was_called_on != 0:
            #     print("Collsion case call")
            #     self.fly_to_next_gate_with_moveOnSpline()

        elif (self.finished_race == True and L2_norm(self.curr_lin_vel) < 0.5):
            # race is finished
            self.reset_race()
            self.finished_race == False
            self.terminated_program = True
            time.sleep(0.5)
            # self.race_again()
        else:
            pass


    def fly_to_first_gate_with_moveOnSpline(self):
        # print(self.gate_poses_ground_truth[self.next_gate_idx].position)
        # The drone starts from not moving, don't add vel and acc constraints
        return self.airsim_client.moveOnSplineAsync([self.gate_poses_ground_truth[self.next_gate_idx].position], 
                                                     vel_max=self.vel_max[self.next_gate_idx],
                                                     acc_max=self.acc_max[self.next_gate_idx], 
                                                     add_position_constraint=True, 
                                                     add_velocity_constraint=False, 
                                                     add_acceleration_constraint=False, 
                                                     viz_traj=self.viz_traj, 
                                                     viz_traj_color_rgba=self.viz_traj_color_rgba, 
                                                     vehicle_name=self.drone_name)
    
    def fly_to_first_point_with_moveOnSpline(self, point):
        return self.airsim_client.moveOnSplineAsync([point], 
                                                    vel_max=self.vel_max[self.next_gate_idx],
                                                    acc_max=self.acc_max[self.next_gate_idx], 
                                                    add_position_constraint=True, 
                                                    add_velocity_constraint=False, 
                                                    add_acceleration_constraint=False, 
                                                    viz_traj=self.viz_traj, 
                                                    viz_traj_color_rgba=self.viz_traj_color_rgba, 
                                                    vehicle_name=self.drone_name)
    
    def fly_to_next_gate_with_moveOnSpline(self):
        # print(self.gate_poses_ground_truth[self.next_gate_idx].position)
        return self.airsim_client.moveOnSplineAsync([self.gate_poses_ground_truth[self.next_gate_idx].position], 
                                                    vel_max=self.vel_max[self.next_gate_idx],
                                                    acc_max=self.acc_max[self.next_gate_idx], 
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
                                                    vel_max=self.vel_max[self.next_gate_idx],
                                                    acc_max=self.acc_max[self.next_gate_idx], 
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

    def race_again(self):
        self.start_race(1)
        self.reset_drone_parameter()
        self.takeoff_with_moveOnSpline()


def main(args):
    # ensure you have generated the neurips planning settings file by running python generate_settings_file.py
    
    baseline_racer.load_level(args.level_name)
    baseline_racer.start_image_callback_thread()
    baseline_racer.start_race(args.race_tier)
    baseline_racer.get_ground_truth_gate_poses()
    baseline_racer.initialize_drone()
    baseline_racer.takeoff_with_moveOnSpline()
    baseline_racer.start_odometry_callback_thread()

    # Comment out the following if you observe the python script exiting prematurely, and resetting the race 
    # baseline_racer.stop_image_callback_thread()
    # baseline_racer.stop_odometry_callback_thread()
    # baseline_racer.reset_race()


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--level_name', type=str, choices=["Soccer_Field_Easy", "Soccer_Field_Medium", "ZhangJiaJie_Medium", "Building99_Hard", 
        "Qualifier_Tier_1", "Qualifier_Tier_2", "Qualifier_Tier_3", "Final_Tier_1", "Final_Tier_2", "Final_Tier_3"], default="Soccer_Field_Medium")
    parser.add_argument('--planning_baseline_type', type=str, choices=["all_gates_at_once","all_gates_one_by_one"], default="all_gates_at_once")
    parser.add_argument('--planning_and_control_api', type=str, choices=["moveOnSpline", "moveOnSplineVelConstraints"], default="moveOnSpline")
    parser.add_argument('--enable_viz_traj', dest='viz_traj', action='store_true', default=True)
    parser.add_argument('--enable_viz_image_cv2', dest='viz_image_cv2', action='store_true', default=True)
    parser.add_argument('--race_tier', type=int, choices=[1,2,3], default=1)
    args = parser.parse_args()
    baseline_racer = BaselineRacer(drone_name="drone_1", viz_traj=args.viz_traj, viz_traj_color_rgba=[1.0, 1.0, 0.0, 1.0], viz_image_cv2=args.viz_image_cv2)
    main(args)
