import torch
import argparse
import os
import logging
import pandas as pd
import tensorflow as tf
from utils.simulator import *
from utils.test_utils import *
from model.planner import MotionPlanner
from model.predictor import Predictor
from waymo_open_dataset.protos import scenario_pb2

def closed_loop_test():
    # logging
    log_path = f"./testing_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+'test.log')

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Use integrated planning module: {}".format(args.use_planning))
    logging.info("Use device: {}".format(args.device))

    # test file
    scenarios = tf.data.TFRecordDataset(args.test_file)

    # set up simulator
    simulator = Simulator(150) # temporal horizon 15s    

    # load model
    predictor = Predictor(50).to(args.device)
    predictor.load_state_dict(torch.load(args.model_path, map_location=args.device))
    predictor.eval()

    # cache results
    collisions, off_routes, progress = [], [], []
    Accs, Jerks, Lat_Accs = [], [], []
    Human_Accs, Human_Jerks, Human_Lat_Accs = [], [], []
    similarity_3s, similarity_5s, similarity_10s = [], [], []

    # set up planner
    if args.use_planning:
        trajectory_len, feature_len = 50, 9
        planner = MotionPlanner(trajectory_len, feature_len, device=args.device, test=True)

    # iterate scenarios in the test file
    for scenario in scenarios:
        parsed_data = scenario_pb2.Scenario()
        parsed_data.ParseFromString(scenario.numpy())
        simulator.load_scenario(parsed_data)
        logging.info(f'Scenario: {simulator.scenario_id}')

        obs = simulator.reset()
        done = False

        while not done:
            logging.info(f'Time: {simulator.timestep-19}')
            ego = torch.from_numpy(obs[0]).to(args.device)
            neighbors = torch.from_numpy(obs[1]).to(args.device)
            lanes = torch.from_numpy(obs[2]).to(args.device)
            crosswalks = torch.from_numpy(obs[3]).to(args.device)
            ref_line = torch.from_numpy(obs[4]).to(args.device)
            current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]

            # predict
            with torch.no_grad():
                plans, predictions, scores, cost_function_weights = predictor(ego, neighbors, lanes, crosswalks)
                plan, prediction = select_future(plans, predictions, scores)

            # plan
            if args.use_planning:
                planner_inputs = {
                    "control_variables": plan.view(-1, 100),
                    "predictions": prediction,
                    "ref_line_info": ref_line,
                    "current_state": current_state
                }

                for i in range(feature_len):
                    planner_inputs[f'cost_function_weight_{i+1}'] = cost_function_weights[:, i].unsqueeze(0)

                with torch.no_grad():
                    final_values, info = planner.layer.forward(planner_inputs, optimizer_kwargs={'track_best_solution': True})
                    plan = info.best_solution['control_variables'].view(-1, 50, 2).to(args.device)

            plan_traj = bicycle_model(plan, ego[:, -1])[:, :, :3]
            plan_traj = plan_traj.cpu().numpy()[0]
            prediction = prediction.cpu().numpy()[0]

            # take one step
            obs, done, info = simulator.step(plan_traj, prediction)
            logging.info(f'Collision: {info[0]}, Off-route: {info[1]}')

            # render
            if args.render:
                simulator.render()

        # calculate metrics
        collisions.append(info[0])
        off_routes.append(info[1])
        progress.append(simulator.calculate_progress())

        dynamics = simulator.calculate_dynamics()
        acc = np.mean(np.abs(dynamics[0]))
        jerk = np.mean(np.abs(dynamics[1])) 
        lat_acc = np.mean(np.abs(dynamics[2]))
        Accs.append(acc)
        Jerks.append(jerk)
        Lat_Accs.append(lat_acc)

        error, human_dynamics = simulator.calculate_human_likeness()
        h_acc = np.mean(np.abs(human_dynamics[0]))
        h_jerk = np.mean(np.abs(human_dynamics[1])) 
        h_lat_acc = np.mean(np.abs(human_dynamics[2]))
        Human_Accs.append(h_acc)
        Human_Jerks.append(h_jerk)
        Human_Lat_Accs.append(h_lat_acc)

        similarity_3s.append(error[29])
        similarity_5s.append(error[49])
        similarity_10s.append(error[99])

        # save animation
        if args.save:
            simulator.save_animation(log_path)

    # save metircs
    df = pd.DataFrame(data={'collision':collisions, 'off_route':off_routes, 'progress': progress,
                            'Acc':Accs, 'Jerk':Jerks, 'Lat_Acc':Lat_Accs, 
                            'Human_Acc':Human_Accs, 'Human_Jerk':Human_Jerks, 'Human_Lat_Acc':Human_Lat_Accs,
                            'Human_L2_3s':similarity_3s, 'Human_L2_5s':similarity_5s, 'Human_L2_10s':similarity_10s})
    df.to_csv(f"./testing_log/{args.name}/{args.test_file.split('/')[-1]}.csv")

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Closed-loop Test')
    parser.add_argument('--name', type=str, help='log name (default: "Test1")', default="Test1")
    parser.add_argument('--test_file', type=str, help='path to the test file')
    parser.add_argument('--model_path', type=str, help='path to saved model')
    parser.add_argument('--use_planning', action="store_true", help='if use integrated planning module (default: False)', default=False)
    parser.add_argument('--render', action="store_true", help='if render the scene (default: False)', default=False)
    parser.add_argument('--save', action="store_true", help='if save animation (default: False)', default=False)
    parser.add_argument('--device', type=str, help='run on which device (default: cpu)', default='cpu')
    args = parser.parse_args()

    # Run
    closed_loop_test()
