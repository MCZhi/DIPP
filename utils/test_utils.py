import logging
import torch
import matplotlib.pyplot as plt
import scipy.spatial as T
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.affinity import affine_transform, rotate
from data_process import *

def initLogging(log_file: str, level: str = "INFO"):
    logging.basicConfig(filename=log_file, filemode='w',
                        level=getattr(logging, level, None),
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())

class TestDataProcess(DataProcess):
    def __init__(self):
        self.num_neighbors = 10
        self.hist_len = 20
        self.future_len = 50

    def process_frame(self, timestep, sdc_id, tracks):      
        ego = self.ego_process(sdc_id, timestep, tracks) 
        neighbors, neighbors_to_predict = self.neighbors_process(sdc_id, timestep, tracks)
        agent_map = np.zeros(shape=(1+self.num_neighbors, 6, 100, 17), dtype=np.float32)
        agent_map_crosswalk = np.zeros(shape=(1+self.num_neighbors, 4, 100, 3), dtype=np.float32)

        agent_map[0], agent_map_crosswalk[0] = self.map_process(ego, timestep, type=1)
        for i in range(self.num_neighbors):
            if neighbors[i, -1, 0] != 0:
                agent_map[i+1], agent_map_crosswalk[i+1] = self.map_process(neighbors[i], timestep)

        ref_line = self.route_process(sdc_id, timestep, self.current_xyh, tracks)

        ground_truth = self.ground_truth_process(sdc_id, timestep, tracks)
        gt_future_states = ground_truth.copy()
        ego, neighbors, map, map_crosswalk, ref_line, ground_truth = self.normalize_data(ego, neighbors, agent_map, agent_map_crosswalk, ref_line, ground_truth, viz=False)

        ego = np.expand_dims(ego, axis=0)
        neighbors = np.expand_dims(neighbors, axis=0)
        map_lanes = np.expand_dims(map, axis=0)
        map_crosswalk = np.expand_dims(map_crosswalk, axis=0)
        ref_line = np.expand_dims(ref_line, axis=0)        
        
        return ego, neighbors, map_lanes, map_crosswalk, ref_line, neighbors_to_predict, ground_truth, gt_future_states

def select_future(plans, predictions, scores):
    best_mode = torch.argmax(scores, dim=-1)
    plan = torch.stack([plans[i, m] for i, m in enumerate(best_mode)])
    prediction = torch.stack([predictions[i, m] for i, m in enumerate(best_mode)])

    return plan, prediction

def CTRV_model(agents):
    prediction = []
    dt = 0.1

    for i in range(agents.shape[1]):
        turn_rate = (agents[0, i, -1, 2] - agents[0, i, -2, 2]) / dt
        velocity = torch.hypot(agents[0, i, -1, 3], agents[0, i, -1, 4])
        turn_rate = turn_rate.repeat(50).clip(-0.5, 0.5)
        velocity = velocity.repeat(50)
        theta = agents[0, i, -1, 2] + torch.cumsum(turn_rate * dt, dim=-1)
        theta = torch.fmod(theta, 2*torch.pi)
        
        # x and y coordniate
        x = agents[0, i, -1, 0] + torch.cumsum(velocity * torch.cos(theta) * dt, dim=-1)
        y = agents[0, i, -1, 1] + torch.cumsum(velocity * torch.sin(theta) * dt, dim=-1)
        traj = torch.stack([x, y, theta], dim=-1)
        prediction.append(traj)

    prediction = torch.stack(prediction).unsqueeze(0)

    return prediction

def bicycle_model(control, current_state):
    dt = 0.1 # discrete time period [s]
    max_delta = 0.6 # vehicle's steering limits [rad]
    max_a = 5 # vehicle's accleration limits [m/s^2]

    x_0 = current_state[:, 0] # vehicle's x-coordinate [m]
    y_0 = current_state[:, 1] # vehicle's y-coordinate [m]
    theta_0 = current_state[:, 2] # vehicle's heading [rad]
    v_0 = torch.hypot(current_state[:, 3], current_state[:, 4]) # vehicle's velocity [m/s]
    L = 3.089 # vehicle's wheelbase [m]
    a = control[:, :, 0].clamp(-max_a, max_a) # vehicle's accleration [m/s^2]
    delta = control[:, :, 1].clamp(-max_delta, max_delta) # vehicle's steering [rad]

    # speed
    v = v_0.unsqueeze(1) + torch.cumsum(a * dt, dim=1)
    v = torch.clamp(v, min=0)

    # angle
    d_theta = v * torch.tan(delta) / L
    theta = theta_0.unsqueeze(1) + torch.cumsum(d_theta * dt, dim=-1)
    theta = torch.fmod(theta, 2*torch.pi)
    
    # x and y coordniate
    x = x_0.unsqueeze(1) + torch.cumsum(v * torch.cos(theta) * dt, dim=-1)
    y = y_0.unsqueeze(1) + torch.cumsum(v * torch.sin(theta) * dt, dim=-1)
    
    # output trajectory
    traj = torch.stack([x, y, theta, v], dim=-1)

    return traj

def physical_model(control, current_state, dt=0.1):
    dt = 0.1 # discrete time period [s]
    max_d_theta = 0.5 # vehicle's change of angle limits [rad/s]
    max_a = 5 # vehicle's accleration limits [m/s^2]

    x_0 = current_state[:, 0] # vehicle's x-coordinate
    y_0 = current_state[:, 1] # vehicle's y-coordinate
    theta_0 = current_state[:, 2] # vehicle's heading [rad]
    v_0 = torch.hypot(current_state[:, 3], current_state[:, 4]) # vehicle's velocity [m/s]
    a = control[:, :, 0].clip(-max_a, max_a) # vehicle's accleration [m/s^2]
    d_theta = control[:, :, 1].clip(-max_d_theta, max_d_theta) # vehicle's heading change rate [rad/s]

    # speed
    v = v_0.unsqueeze(1) + torch.cumsum(a * dt, dim=1)
    v = torch.clamp(v, min=0)

    # angle
    theta = theta_0.unsqueeze(1) + torch.cumsum(d_theta * dt, dim=-1)
    theta = torch.fmod(theta, 2*torch.pi)
    
    # x and y coordniate
    x = x_0.unsqueeze(1) + torch.cumsum(v * torch.cos(theta) * dt, dim=-1)
    y = y_0.unsqueeze(1) + torch.cumsum(v * torch.sin(theta) * dt, dim=-1)

    # output trajectory
    traj = torch.stack([x, y, theta, v], dim=-1)

    return traj

def map_process(map_feature, map_type):
    if map_type == 'lane':
        polyline = np.array([(map_point.x, map_point.y) for map_point in map_feature.polyline], dtype=np.float32)

        return polyline, map_feature.entry_lanes

    elif map_type == 'road_line':
        polyline = np.array([(map_point.x, map_point.y) for map_point in map_feature.polyline], dtype=np.float32)

        if map_feature.type == 1:
            plt.plot(polyline[:, 0], polyline[:, 1], 'w', linestyle='dashed', linewidth=2)
        elif map_feature.type == 2:
            plt.plot(polyline[:, 0], polyline[:, 1], 'w', linestyle='solid', linewidth=2)
        elif map_feature.type == 3:
            plt.plot(polyline[:, 0], polyline[:, 1], 'w', linestyle='solid', linewidth=3)
        elif map_feature.type == 4:
            plt.plot(polyline[:, 0], polyline[:, 1], 'xkcd:yellow', linestyle='dashed', linewidth=2)
        elif map_feature.type == 5:
            plt.plot(polyline[:, 0], polyline[:, 1], 'xkcd:yellow', linestyle='dashed', linewidth=3)
        elif map_feature.type == 6:
            plt.plot(polyline[:, 0], polyline[:, 1], 'xkcd:yellow', linestyle='solid', linewidth=2)
        elif map_feature.type == 7:
            plt.plot(polyline[:, 0], polyline[:, 1], 'xkcd:yellow', linestyle='solid', linewidth=3)
        elif map_feature.type == 8:
            plt.plot(polyline[:, 0], polyline[:, 1], 'xkcd:yellow', linestyle='dotted', linewidth=2)
        else:
            plt.plot(polyline[:, 0], polyline[:, 1], 'k', linewidth=1)

        return polyline

    elif map_type == 'road_edge':
        polyline = np.array([(map_point.x, map_point.y) for map_point in map_feature.polyline], dtype=np.float32)

        if map_feature.type == 1:
            plt.plot(polyline[:, 0], polyline[:, 1], 'k', linewidth=3)
        elif map_feature.type == 2:
            plt.plot(polyline[:, 0], polyline[:, 1], 'k', linewidth=2)
        else:
            plt.plot(polyline[:, 0], polyline[:, 1], 'k', linewidth=1)

        return polyline

    elif map_type == 'stop_sign':
        position = np.array([map_feature.position.x, map_feature.position.y], dtype=np.float32)
        plt.gca().add_patch(plt.Circle(position, 2, color='r'))

        return position

    elif map_type == 'crosswalk':
        polyline = polygon_completion(map_feature.polygon).astype(np.float32)
        plt.plot(polyline[:, 0], polyline[:, 1], 'b', linewidth=4)

        return polyline

    elif map_type == 'speed_bump':
        polyline = polygon_completion(map_feature.polygon).astype(np.float32)
        plt.plot(polyline[:, 0], polyline[:, 1], 'xkcd:orange', linewidth=4)

        return polyline

    else:
        raise TypeError

def polygon_completion(polygon):
    polyline_x = []
    polyline_y = []

    for i in range(len(polygon)):
        if i+1 < len(polygon):
            next = i+1
        else:
            next = 0

        dist_x = polygon[next].x - polygon[i].x
        dist_y = polygon[next].y - polygon[i].y
        dist = np.linalg.norm([dist_x, dist_y])
        interp_num = np.ceil(dist)*2
        interp_index = np.arange(2+interp_num)
        point_x = np.interp(interp_index, [0, interp_index[-1]], [polygon[i].x, polygon[next].x]).tolist()
        point_y = np.interp(interp_index, [0, interp_index[-1]], [polygon[i].y, polygon[next].y]).tolist()
        polyline_x.extend(point_x[:-1])
        polyline_y.extend(point_y[:-1])
        
    return np.stack([polyline_x, polyline_y], axis=1)

def traffic_signal_process(lanes, traffic_signal):
    stop_point = traffic_signal.stop_point

    if traffic_signal.state in [1, 4, 7]:
        state = 'r' 
    elif traffic_signal.state in [2, 5, 8]:
        state = 'y'
    elif traffic_signal.state in [3, 6]:
        state = 'g'
    else:
        state = None

    if state:
        light = plt.Circle((stop_point.x, stop_point.y), 1.2, color=state)
        plt.gca().add_patch(light)
        
def wrap_to_pi(theta):
    return (theta+np.pi) % (2*np.pi) - np.pi

def transform(out_coord, ori_coord, include_curr=False):
    line = LineString(out_coord[:, :2])
    line = rotate(line, ori_coord[2], origin=(0, 0), use_radians=True)
    line = affine_transform(line, [1, 0, 0, 1, ori_coord[0], ori_coord[1]])

    if include_curr:
        line = np.insert(line.coords, 0, ori_coord[:2], axis=0)
    else:
        line = np.array(line.coords)

    return line

def return_circle_list(x, y, l, w, yaw):
    r = w/np.sqrt(2)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    if l < 4.0:
        c1 = [x-(l-w)/2*cos_yaw, y-(l-w)/2*sin_yaw]
        c2 = [x+(l-w)/2*cos_yaw, y+(l-w)/2*sin_yaw]
        c = [c1, c2]
    elif l >= 4.0 and l < 8.0:
        c0 = [x, y]
        c1 = [x-(l-w)/2*cos_yaw, y-(l-w)/2*sin_yaw]
        c2 = [x+(l-w)/2*cos_yaw, y+(l-w)/2*sin_yaw]
        c = [c0, c1, c2]
    else:
        c0 = [x, y]
        c1 = [x-(l-w)/2*cos_yaw, y-(l-w)/2*sin_yaw]
        c2 = [x+(l-w)/2*cos_yaw, y+(l-w)/2*sin_yaw]
        c3 = [x-(l-w)/2*cos_yaw/2, y-(l-w)/2*sin_yaw/2]
        c4 = [x+(l-w)/2*cos_yaw/2, y+(l-w)/2*sin_yaw/2]
        c = [c0, c1, c2, c3, c4]

    for i in range(len(c)):
        c[i] = np.stack(c[i], axis=-1)

    c = np.stack(c, axis=-2)

    return c

def return_collision_threshold(w1, w2):
    return (w1 + w2) / np.sqrt(3.8)

def check_collision(ego_center_points, neighbor_center_points, sizes):
    collision = False

    for t in range(ego_center_points.shape[0]):
        if check_collision_step(ego_center_points[t], neighbor_center_points[:, t], sizes):
            collision = True

    return collision 

def check_collision_step(ego_center_points, neighbor_center_points, sizes):
    collision = []
    plan_x, plan_y, plan_yaw, plan_l, plan_w = ego_center_points[0], ego_center_points[1], ego_center_points[2], sizes[0, 0], sizes[0, 1]
    ego_vehicle = return_circle_list(plan_x, plan_y, plan_l, plan_w, plan_yaw)  

    for i in range(neighbor_center_points.shape[0]):
        neighbor_length = sizes[i+1, 0]
        neighbor_width = sizes[i+1, 1]

        if neighbor_center_points[i, 0] != 0:
            neighbor_vehicle = return_circle_list(neighbor_center_points[i, 0], neighbor_center_points[i, 1], neighbor_length, neighbor_width, neighbor_center_points[i, 2])
            distance = [np.linalg.norm(ego_vehicle[i]-neighbor_vehicle[j], axis=-1) for i in range(ego_vehicle.shape[0]) for j in range(neighbor_vehicle.shape[0])]
            distance = np.stack(distance, axis=-1)
            threshold = return_collision_threshold(plan_w, neighbor_width)
            collision.append(np.any(distance < threshold))

    return np.any(collision)

def check_dynamics(traj):
    traj = np.array(traj)
    v_x, v_y, theta = np.diff(traj[:, 0]) / 0.1, np.diff(traj[:, 1]) / 0.1, traj[1:, 2]
    lon_speed = v_x * np.cos(theta) + v_y * np.sin(theta)
    lat_speed = v_y * np.cos(theta) - v_x * np.sin(theta)
    acc = np.diff(lon_speed) / 0.1
    jerk = np.diff(lon_speed, n=2) / 0.01
    lat_acc = np.diff(lat_speed) / 0.1

    return np.mean(np.abs(acc)), np.mean(np.abs(jerk)), np.mean(np.abs(lat_acc))

def check_traffic(traj, ref_line):
    red_light = False
    off_route = False

    # project to frenet
    distance_to_ref = T.distance.cdist(traj[:, :2], ref_line[:, :2])
    s_ego = np.argmin(distance_to_ref, axis=-1)
    distance_to_route = np.min(distance_to_ref, axis=-1)

    if np.any(distance_to_route > 5):
        off_route = True

    # get stop point 
    stop_point = np.where(ref_line[:, -1]==0)[0]

    if stop_point.any() and np.any(s_ego > np.min(stop_point)):
        red_light = True

    return red_light, off_route

def check_similarity(traj, gt):
    error = np.linalg.norm(traj[:, :2] - gt[:, :2], axis=-1)
    
    return error

def check_prediction(trajs, gt):
    ADE = []
    FDE = []
    mask = np.not_equal(gt[:, :, :2], 0)

    for i in range(10):
        if mask[i, 0, 0]:
            error = np.linalg.norm(trajs[i, :, :2] - gt[i, :, :2], axis=-1) 
            error = error * mask[i, :, 0]
            ADE.append(np.mean(error))
            FDE.append(error[-1])

    return np.mean(ADE), np.mean(FDE)
