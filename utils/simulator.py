import numpy as np
import tensorflow as tf
from data_process import DataProcess
from utils.test_utils import *
import matplotlib.animation as animation
import matplotlib.pyplot as plt

class Simulator(DataProcess):
    def __init__(self, timespan):
        self.num_neighbors = 10
        self.hist_len = 20
        self.future_len = 50
        self.timespan = timespan      
        
    def load_scenario(self, scenario):
        self.scenario = scenario
        self.scenario_id = scenario.scenario_id
        self.sdc_id = scenario.sdc_track_index
        self.timesteps = scenario.timestamps_seconds
        self.tracks = scenario.tracks
        self.map_features = scenario.map_features
        self.dynamic_map_states = scenario.dynamic_map_states
        self.build_map(scenario.map_features, scenario.dynamic_map_states)

    def reset(self):
        plt.close('all')
        self.timestep = 19
        self.scene_imgs = [] 
        self.sdc_trajectory = []
        self.sdc_gt_trajectory = []
        
        track = self.tracks[self.sdc_id].states
        self.sdc_state = np.array((track[self.timestep].center_x, track[self.timestep].center_y, track[self.timestep].heading, track[self.timestep].velocity_x, 
                                   track[self.timestep].velocity_y, track[self.timestep].length, track[self.timestep].width, track[self.timestep].height))
        self.sdc_hist_states = np.array([[track[t].center_x, track[t].center_y, track[t].heading, track[t].velocity_x, 
                                          track[t].velocity_y, track[t].length, track[t].width, track[t].height] for t in range(self.timestep+1)])
        self.sdc_route = np.array([(state.center_x, state.center_y, state.heading) for state in track])

        obs = self.feed_data(self.timestep)

        return obs

    def step(self, plan, prediction):     
        # update timestep 
        self.timestep += 1

        # transform to global
        xy = transform(plan, self.sdc_state)
        self.plan = xy

        self.prediction = []
        for i in range(len(self.neighbors_to_predict)):
            predict_traj = transform(prediction[i], self.sdc_state)
            self.prediction.append(predict_traj)
        
        # update sdc state
        velocity = (xy[0] - self.sdc_state[:2]) / 0.1
        heading = plan[0, 2].clip(-0.1, 0.1) + self.sdc_state[2]
        self.sdc_state = np.concatenate([xy[0], [heading], velocity, self.sdc_state[-3:]])
        self.sdc_trajectory.append(self.sdc_state[:3])
        self.sdc_gt_trajectory.append(self.sdc_route[self.timestep])
        self.sdc_hist_states = np.concatenate([self.sdc_hist_states[1:], np.expand_dims(self.sdc_state, 0)])

        # update neighbors states
        self.neighbors_states = {'OfI_neighbors':[], 'background_neighbors':[]}

        # objects of interest
        for i in self.neighbors_to_predict:
            self.neighbors_states['OfI_neighbors'].append((self.tracks[i].object_type, self.tracks[i].states[self.timestep]))
        
        # background
        for i, track in enumerate(self.tracks):
            if i != self.sdc_id and track.states[self.timestep].valid:
                if i not in self.neighbors_to_predict:
                    self.neighbors_states['background_neighbors'].append(track.states[self.timestep])
        
        # process data
        obs = self.feed_data(self.timestep, True)
        if obs is None:
            return None, True, (False, False)

        # check
        collision = self.check_collision()
        off_route = self.check_off_route()
        done = collision or off_route or self.timestep > (self.timespan+19) or self.timestep >= len(self.timesteps)-1
        info = (collision, off_route)

        return obs, done, info

    def feed_data(self, timestep, override=False):
        if not override:
            ego = self.ego_process(self.sdc_id, timestep, self.tracks)
        else:
            ego = self.sdc_hist_states.copy()
            self.current_xyh = self.sdc_state[:3].copy()

        neighbors, neighbors_to_predict = self.neighbors_process(self.sdc_id, timestep, self.tracks)
        agent_map = np.zeros(shape=(1+self.num_neighbors, 6, 100, 17), dtype=np.float32)
        agent_map_crosswalk = np.zeros(shape=(1+self.num_neighbors, 4, 100, 3), dtype=np.float32)

        agent_map[0], agent_map_crosswalk[0] = self.map_process(ego, timestep, type=1)
        for i in range(self.num_neighbors):
            if neighbors[i, -1, 0] != 0:
                agent_map[i+1], agent_map_crosswalk[i+1] = self.map_process(neighbors[i], timestep)

        ref_line = self.route_process(self.sdc_id, timestep, self.current_xyh, self.tracks)
        if ref_line is None:
            return None
        else:
            self.ref_line = ref_line.copy()

        ground_truth = self.ground_truth_process(self.sdc_id, timestep, self.tracks)
        gt_future_states = ground_truth.copy()
        ego, neighbors, map, map_crosswalk, ref_line, ground_truth = self.normalize_data(ego, neighbors, agent_map, agent_map_crosswalk, ref_line, ground_truth, viz=False)

        ego = np.expand_dims(ego, axis=0).astype(np.float32)
        neighbors = np.expand_dims(neighbors, axis=0)
        map_lanes = np.expand_dims(map, axis=0)
        map_crosswalk = np.expand_dims(map_crosswalk, axis=0)
        ref_line = np.expand_dims(ref_line, axis=0)        

        self.neighbors_to_predict = neighbors_to_predict
        self.ground_truth = ground_truth
        self.gt_future_states = gt_future_states
        
        return ego, neighbors, map_lanes, map_crosswalk, ref_line

    def check_collision(self):
        collision = []
        ego_x, ego_y, ego_yaw, ego_l, ego_w = self.sdc_state[0], self.sdc_state[1], self.sdc_state[2], self.sdc_state[-3], self.sdc_state[-2]
        ego_circles = return_circle_list(ego_x, ego_y, ego_l, ego_w, ego_yaw)

        for neighbor in self.neighbors_states['OfI_neighbors']:
            neighbor_state = neighbor[1]
            neighbor_length = neighbor_state.length
            neighbor_width = neighbor_state.width           
            neighbor_circles = return_circle_list(neighbor_state.center_x, neighbor_state.center_y, neighbor_length, neighbor_width, neighbor_state.heading)
            distance = [np.linalg.norm(ego_circles[i]-neighbor_circles[j], axis=-1) for i in range(ego_circles.shape[0]) for j in range(neighbor_circles.shape[0])]
            distance = np.stack(distance, axis=-1)
            threshold = return_collision_threshold(ego_w, neighbor_width)
            collision.append(np.any((distance < threshold)))

        return np.any(collision)

    def check_off_route(self):
        distance_to_route = T.distance.cdist(self.sdc_state[None, :2], self.ref_line[:, :2])
        distance = np.min(distance_to_route, axis=-1)[0]
        off_route = distance > 5

        return off_route

    def calculate_progress(self):
        progress = 0
        
        for t in range(1, len(self.sdc_trajectory)):
            dx = self.sdc_trajectory[t][0] - self.sdc_trajectory[t-1][0]
            dy = self.sdc_trajectory[t][1] - self.sdc_trajectory[t-1][1]
            progress += np.hypot(dx, dy)

        return progress

    def calculate_dynamics(self, traj=None):
        traj = traj if traj else self.sdc_trajectory
        traj = np.array(traj)
        v_x, v_y, theta = np.diff(traj[:, 0]) / 0.1, np.diff(traj[:, 1]) / 0.1, traj[:-1, 2]
        lon_speed = v_x * np.cos(theta) + v_y * np.sin(theta)
        lat_speed = v_y * np.cos(theta) - v_x * np.sin(theta)
        acc = np.diff(lon_speed) / 0.1
        jerk = np.diff(lon_speed, n=2) / 0.01
        lat_acc = np.diff(lat_speed) / 0.1

        return acc, jerk, lat_acc
    
    def calculate_human_likeness(self):
        sdc_trajectory = np.array(self.sdc_trajectory)

        if len(self.sdc_trajectory) < 100:
            repeated_last_point = np.repeat(sdc_trajectory[np.newaxis, -1], 100-sdc_trajectory.shape[0], axis=0)
            sdc_trajectory = np.append(sdc_trajectory, repeated_last_point, axis=0)

        error = np.linalg.norm(sdc_trajectory[:100, :2] - np.array(self.sdc_route)[20:120, :2], axis=-1)
        human = self.calculate_dynamics(self.sdc_gt_trajectory)

        return error, human

    def render(self):
        plt.ion()
        ax = plt.gca()
        fig = plt.gcf()
        dpi = 100
        size_inches = 800 / dpi
        fig.set_size_inches([size_inches, size_inches])
        fig.set_dpi(dpi)
        fig.set_tight_layout(True)
        fig.set_facecolor('xkcd:grey') 

        # map
        for vector in self.map_features:
            vector_type = vector.WhichOneof("feature_data")
            vector = getattr(vector, vector_type)
            polyline = map_process(vector, vector_type)

        # sdc
        agent_color = ['r', 'm', 'b', 'g'] # [sdc, vehicle, pedestrian, cyclist]
        color = agent_color[0]
        rect = plt.Rectangle((self.sdc_state[0]-self.sdc_state[-3]/2, self.sdc_state[1]-self.sdc_state[-2]/2), 
                              self.sdc_state[-3], self.sdc_state[-2], linewidth=2, color=color, alpha=0.8, zorder=3,
                              transform=mpl.transforms.Affine2D().rotate_around(*(self.sdc_state[0], self.sdc_state[1]), self.sdc_state[2]) + ax.transData)
        ax.add_patch(rect)
        ax.plot(self.plan[::7, 0], self.plan[::7, 1], linewidth=2, color=color, marker='.', markersize=6, zorder=4)

        # neighbors
        for i, neighbor in enumerate(self.neighbors_states['OfI_neighbors']):
            type = neighbor[0]
            state = neighbor[1]
            color = agent_color[type]
            rect = plt.Rectangle((state.center_x-state.length/2, state.center_y-state.width/2), 
                                  state.length, state.width, linewidth=2, color=color, alpha=0.7, zorder=3,
                                  transform=mpl.transforms.Affine2D().rotate_around(*(state.center_x, state.center_y), state.heading) + ax.transData)
            ax.add_patch(rect)
            ax.plot(self.prediction[i][::7, 0], self.prediction[i][::7, 1], linewidth=2, color=color, marker='.', markersize=6, zorder=3)        

        for i, neighbor in enumerate(self.neighbors_states['background_neighbors']):
            rect = plt.Rectangle((neighbor.center_x-neighbor.length/2, neighbor.center_y-neighbor.width/2), 
                                  neighbor.length, neighbor.width, linewidth=2, color='k', alpha=0.6, zorder=3,
                                  transform=mpl.transforms.Affine2D().rotate_around(*(neighbor.center_x, neighbor.center_y), neighbor.heading) + ax.transData)
            ax.add_patch(rect)

        # dynamic_map_states
        for signal in self.dynamic_map_states[self.timestep].lane_states:
            traffic_signal_process(self.lanes, signal)

        # show plot
        ax.axis([-100 + self.sdc_state[0], 100 + self.sdc_state[0], -100 + self.sdc_state[1], 100 + self.sdc_state[1]])
        ax.set_aspect('equal')
        ax.grid(False)
        ax.margins(0) 
        ax.axis('off') 
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))   
        self.scene_imgs.append(data)   
        plt.pause(0.1)
        plt.clf()

    @staticmethod
    def create_animation(images):
        plt.ioff()
        fig, ax = plt.subplots()
        dpi = 100
        size_inches = 800 / dpi
        fig.set_size_inches([size_inches, size_inches])
        plt.ion()

        def animate_func(i):
            ax.imshow(images[i])
            fig.set_tight_layout(True)
            ax.grid(False)
            ax.margins(0)
            ax.axis('off')

        anim = animation.FuncAnimation(fig, animate_func, frames=len(images), interval=100)

        return anim
    
    def save_animation(self, path='.'):
        anim = self.create_animation(self.scene_imgs)
        anim.save(path+f'{self.scenario_id}.mp4')
    
