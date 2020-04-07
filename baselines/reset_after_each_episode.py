
from argparse import ArgumentParser
import airsimneurips as airsim
import threading
import utils
from baseline_racer import BaselineRacer

def dummy_reset(baseline_racer, drone_names, tier):
    baseline_racer.airsim_client.simPause()
    baseline_racer.airsim_client.reset()
    for drone_name in drone_names:
        baseline_racer.airsim_client.enableApiControl(vehicle_name=drone_name)
        baseline_racer.airsim_client.arm(vehicle_name=drone_name)
    baseline_racer.airsim_client.simUnPause() # unpause sim to simresetrace works as it's supposed to
    baseline_racer.airsim_client.simResetRace()
    baseline_racer.airsim_client.simStartRace(tier=tier)
    baseline_racer.takeoff_with_moveOnSpline()
    baseline_racer.get_ground_truth_gate_poses()
    baseline_racer.fly_through_all_gates_at_once_with_moveOnSpline().join()

def main(args):
    # ensure you have generated the neurips planning settings file by running python generate_settings_file.py
    baseline_racer = BaselineRacer(drone_name="drone_1", viz_traj=args.viz_traj, viz_image_cv2=False)
    baseline_racer.load_level(args.level_name)

    drone_names = ["drone_1", "drone_2"]

    # for training binaries
    if args.race_tier == 2:
        drone_names = ["drone_1"]

    # for qualification binaries, enforce correct tier regardless of user unit
    if args.level_name == "Qualifier_Tier_1":
        args.race_tier = 1
        drone_names = ["drone_1", "drone_2"]

    if args.level_name == "Qualifier_Tier_2":
        args.race_tier = 1 # for testing so drone_1 goes thru all gates
        # args.race_tier = 2
        drone_names = ["drone_1"]

    if args.level_name == "Qualifier_Tier_3":
        args.race_tier = 1 # for testing so drone_1 goes thru all the gates
        # args.race_tier = 3
        drone_names = ["drone_1", "drone_2"]

    baseline_racer.start_race(args.race_tier)
    baseline_racer.initialize_drone()
    baseline_racer.takeoff_with_moveOnSpline()
    baseline_racer.get_ground_truth_gate_poses()
    baseline_racer.fly_through_all_gates_at_once_with_moveOnSpline().join()
    dummy_reset(baseline_racer, drone_names, args.race_tier)
    # baseline_racer.reset_race()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--level_name', type=str, choices=["Soccer_Field_Easy", "Soccer_Field_Medium", "ZhangJiaJie_Medium", "Building99_Hard", 
        "Qualifier_Tier_1", "Qualifier_Tier_2", "Qualifier_Tier_3"], default="Soccer_Field_Easy")
    parser.add_argument('--enable_viz_traj', dest='viz_traj', action='store_true', default=True)
    parser.add_argument('--race_tier', type=int, choices=[1,2,3], default=1)
    args = parser.parse_args()

    main(args)