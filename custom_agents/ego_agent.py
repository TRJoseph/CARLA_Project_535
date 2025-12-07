import carla
import math
import re
import numpy as np
from llama_cpp import Llama

from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption

# --- PERCEPTION HELPER ---
def get_world_obstacles(world, ego_actor, search_radius=30.0):
    ego_loc = ego_actor.get_location()
    obstacles = []
    
    actors = world.get_actors()
    vehicles = actors.filter('vehicle.*')
    walkers = actors.filter('walker.pedestrian.*')
    all_actors = list(vehicles) + list(walkers)

    for actor in all_actors:
        if actor.id == ego_actor.id: continue
        if ego_loc.distance(actor.get_location()) > search_radius: continue

        vel = actor.get_velocity()
        v_mag = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        
        if v_mag > 0.1:
            heading_rad = math.atan2(vel.y, vel.x)
        else:
            heading_rad = math.radians(actor.get_transform().rotation.yaw)
        
        bbox = actor.bounding_box.extent
        radius = math.sqrt(bbox.x**2 + bbox.y**2)

        obstacles.append((
            actor.id,
            round(actor.get_location().x, 2), 
            round(actor.get_location().y, 2), 
            round(v_mag, 2), 
            round(heading_rad, 2), 
            round(radius, 2)
        ))

    return tuple(obstacles)

def interpolate_trajectory(waypoints, resolution=1.0):

    dense_queue = []
    
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i] # (x, y, v, h)
        p2 = waypoints[i+1]
        
        loc1 = carla.Location(x=p1[0], y=p1[1])
        loc2 = carla.Location(x=p2[0], y=p2[1])
        
        dist = loc1.distance(loc2)
        
        if dist < 0.1: continue 
        
        num_steps = int(dist / resolution)
        
        for n in range(num_steps):
            alpha = n / num_steps
            x = p1[0] * (1 - alpha) + p2[0] * alpha
            y = p1[1] * (1 - alpha) + p2[1] * alpha
            
            yaw_rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            yaw_deg = math.degrees(yaw_rad)
            
            trans = carla.Transform(
                carla.Location(x=x, y=y, z=0.0), 
                carla.Rotation(pitch=0, yaw=yaw_deg, roll=0)
            )
            dense_queue.append(trans)
            
    return dense_queue

class EgoAgent(BasicAgent):
    def __init__(self, vehicle_actor, model_path=None, target_speed=20, debug=False):
        
        super().__init__(vehicle_actor, target_speed=target_speed)
        
        self.obstacle_memory = {}
        
        self.llm = None
        if model_path:
            print(f"Loading LLM from {model_path}...")
            try:
                self.llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=-1, 
                    n_ctx=8192,
                    verbose=False
                )
            except Exception as e:
                print(f"Failed to load LLM: {e}")
        else:
            print("WARNING: No model_path provided. LLM features disabled.")

    def _world_to_body(self, ego_trans, world_loc):

       
        dx = world_loc.x - ego_trans.location.x
        dy = world_loc.y - ego_trans.location.y
        
        yaw = math.radians(ego_trans.rotation.yaw)
        
        local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
        local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
        
        return (local_x, local_y)

    def _body_to_world(self, ego_trans, local_x, local_y):

        yaw = math.radians(ego_trans.rotation.yaw)
        
        dx = local_x * math.cos(yaw) - local_y * math.sin(yaw)
        dy = local_x * math.sin(yaw) + local_y * math.cos(yaw)
        
        world_x = ego_trans.location.x + dx
        world_y = ego_trans.location.y + dy
        
        return carla.Location(x=world_x, y=world_y, z=ego_trans.location.z)
            
    def _should_trigger_llm(self, current_obstacles, current_timestamp):

        trigger = False
        threshold_meters = 1.0
        dt = 0.5

        for obs in current_obstacles:
            obj_id, cx, cy, cv, ch, cr = obs

            if obj_id not in self.obstacle_memory:
                trigger = True
                print(f"[Trigger] New CLOSE Obstacle Detected! ID: {obj_id}")
                continue
            
            prev_data = self.obstacle_memory[obj_id]
            
            prev_vx = prev_data['v'] * math.cos(prev_data['h'])
            prev_vy = prev_data['v'] * math.sin(prev_data['h'])
                
            pred_x = prev_data['x'] + (prev_vx * dt)
            pred_y = prev_data['y'] + (prev_vy * dt)
                
            error = math.sqrt((cx - pred_x)**2 + (cy - pred_y)**2)
                
            if error > threshold_meters:
                print(f"[Trigger] Deviation Detected! ID: {obj_id} Err: {error:.2f}m")
                trigger = True

        new_memory = {}
        for obs in current_obstacles:
            obj_id, cx, cy, cv, ch, cr = obs
            new_memory[obj_id] = {
                'x': cx, 'y': cy, 'v': cv, 'h': ch, 't': current_timestamp
            }
        self.obstacle_memory = new_memory
        
        return trigger
    
    def get_lane_context(self, ego_wp):
        
        lane_width = ego_wp.lane_width
        left_wp = ego_wp.get_left_lane()
        right_wp = ego_wp.get_right_lane()
        
        # 1. Analyze Left Lane (Negative Y)
        left_info = "None (Barrier)"
        if left_wp:
            if left_wp.lane_type == carla.LaneType.Driving:
                # Check for oncoming
                if (ego_wp.lane_id * left_wp.lane_id) < 0:
                    left_info = "Oncoming Traffic (Danger)"
                else:
                    left_info = "Overtaking Lane (Safe)"
            elif left_wp.lane_type == carla.LaneType.Shoulder:
                left_info = "Shoulder"
            elif left_wp.lane_type == carla.LaneType.Sidewalk:
                left_info = "Sidewalk"
        
        # 2. Analyze Right Lane (Positive Y)
        right_info = "None (Barrier)"
        if right_wp:
            if right_wp.lane_type == carla.LaneType.Driving:
                if (ego_wp.lane_id * right_wp.lane_id) < 0:
                    right_info = "Oncoming Traffic (Danger)"
                else:
                    right_info = "Driving Lane (Safe)"
            elif right_wp.lane_type == carla.LaneType.Shoulder:
                right_info = "Shoulder (Emergency Only)"
            elif right_wp.lane_type == carla.LaneType.Sidewalk:
                right_info = "Sidewalk"

        boundary = lane_width / 2.0
        
        context = (
            f"- Lane Width: {lane_width:.1f}m\n"
            f"- Left Boundary (y=-{boundary:.1f}): {left_info}\n"
            f"- Right Boundary (y=+{boundary:.1f}): {right_info}"
        )
        return context
    
    def consult_llm(self, local_obstacles, local_traj, lane_context):
        print(">>> PAUSING FOR LLM (BODY FRAME)...")
        
        margin = 3.0 

        system_instruction = f"""You are a trajectory optimizer.
        
        ### 1. COORDINATE SYSTEM
        - **X**: Forward (Positive).
        - **Y**: **LEFT is NEGATIVE (-)**. **RIGHT is POSITIVE (+)**.
        - **Ego**: Starts at (0,0).
        
        ### 2. MATH RULES
        - If Target Y is POSITIVE, intermediate points MUST act like: 0 -> 0.5 -> 1.0 -> Target.
        - If Target Y is NEGATIVE, intermediate points MUST act like: 0 -> -0.5 -> -1.0 -> Target.
        
        ### 3. REQUIRED REASONING STEPS
        Output a "Calculations" block before the final list:
        A. Identify Closest Obstacle (X, Y).
        B. Define Conflict Zone: [Obs_Y - {margin}, Obs_Y + {margin}].
        C. **DECISION**: Pick "Target Y" outside the zone.
           - Check Lane Context (Don't hit Oncoming/Left if possible).
           - IF Obs is Negative (Left), Swerve POSITIVE (Right).
        D. **PATH CHECK**: 
           - "I decided to swerve [DIRECTION]. Therefore, my Y values must be [SIGN]."
        
        ### 4. OUTPUT
        "FINAL_TRAJECTORY: ((x, y, v, h), ...)"
        (Generate 6 points. X should increase by ~2m-4m per point).
        
        after you have done your calculations, go through the reasoning steps again to make sure it follows the requirements

        ### EXAMPLE
        Input: Obstacle at (15, -0.5). Context: Right is Safe.
        Calculations:
        - Obs Y is -0.5 (Left).
        - Conflict Zone: [-2.5, +1.5].
        - Decision: Swerve Right to Y = +2.0.
        - Path Check: Swerve is Positive. Points must go 0 -> +2.0.
        FINAL_TRAJECTORY: ((5, 0.5, 5, 0.05), (10, 1.2, 5, 0.1), (15, 2.0, 5, 0), (20, 1.5, 5, -0.05), (25, 0, 5, 0))
        """

        user_content = (
            f"LANE CONTEXT:\n{lane_context}\n"
            f"OBSTACLES (x,y,v,h,r): {local_obstacles}\n"
            f"CURRENT TRAJECTORY (x,y,v,h): {local_traj}"
        )

        combined_prompt = f"{system_instruction}\n\nUSER DATA:\n{user_content}\n\nRESPONSE:"
        messages = [{"role": "user", "content": combined_prompt}]

        try:
            output = self.llm.create_chat_completion(
                messages=messages, 
                max_tokens=8192, 
                temperature=0.1,
                top_p=0.9
            )
            response_text = output["choices"][0]["message"]["content"]
            print(f"[LLM REASONING]:\n{response_text}\n")
            
            match = re.search(r"FINAL_TRAJECTORY:\s*(\(\(.*\)\))", response_text, re.DOTALL)
            if not match: 
                match = re.search(r"(\(\(.*\)\))", response_text, re.DOTALL)
            
            if match:
                traj_str = match.group(1)
                return eval(traj_str)
        except Exception as e:
            print(f"LLM Error: {e}")
        return None
    
    def run_step(self):
        
        while len(self._local_planner._waypoints_queue) < 10:
            self._local_planner._compute_next_waypoints(k=10)
        
        queue = list(self._local_planner._waypoints_queue)[:6]
        if not queue: return super().run_step()

        if self.llm:
            snapshot = self._vehicle.get_world().get_snapshot()
            current_time = snapshot.timestamp.elapsed_seconds
            raw_obstacles = get_world_obstacles(self._vehicle.get_world(), self._vehicle, search_radius=15.0)
            
            if self._should_trigger_llm(raw_obstacles, current_time):
                
                ego_trans = self._vehicle.get_transform()
                
                body_obstacles = []
                for o in raw_obstacles:
                    w_loc = carla.Location(x=o[1], y=o[2])
                    b_x, b_y = self._world_to_body(ego_trans, w_loc)
                    b_h = o[4] - math.radians(ego_trans.rotation.yaw)
                    body_obstacles.append((round(b_x, 2), round(b_y, 2), o[3], round(b_h, 2), o[5]))
                
                body_traj = []
                for wp, _ in queue:
                    loc = wp.transform.location
                    b_x, b_y = self._world_to_body(ego_trans, loc)
                    yaw = math.radians(wp.transform.rotation.yaw)
                    b_h = yaw - math.radians(ego_trans.rotation.yaw)
                    body_traj.append((round(b_x, 2), round(b_y, 2), round(self._target_speed/3.6, 2), round(b_h, 2)))
                
                current_map = self._vehicle.get_world().get_map()
                ego_wp = current_map.get_waypoint(self._vehicle.get_location(), project_to_road=True)
                lane_context_str = self.get_lane_context(ego_wp)

                modified_body_traj = self.consult_llm(tuple(body_obstacles), tuple(body_traj), lane_context_str)
                
                if modified_body_traj:
                    print(">>> APPLYING NEW TRAJECTORY...")
                    
                    world_traj = []
                    for pt in modified_body_traj:
                        w_loc = self._body_to_world(ego_trans, pt[0], pt[1])
                        w_yaw = pt[3] + math.radians(ego_trans.rotation.yaw)
                        world_traj.append((w_loc.x, w_loc.y, pt[2], w_yaw))
                        
                    dense_transforms = interpolate_trajectory(world_traj, resolution=0.5)
                    
                    new_queue = []
                    
                    for transform in dense_transforms:
                        transform.location.z = ego_trans.location.z
                        
                        wp = current_map.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Any)
                        
                        wp.transform.location = transform.location
                        wp.transform.rotation = transform.rotation
                        new_queue.append((wp, RoadOption.LANEFOLLOW))
                    
                    self._local_planner._waypoints_queue.clear()
                    self._local_planner._waypoints_queue.extend(new_queue)

        return super().run_step()