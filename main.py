import random
import carla
import time
import yaml
from core.spawn_utils import *
from core.autopilot import *
from core.cleanup import *
from carla_env import CarlaEnv

def merge_dict(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and key in base:
            base[key] = merge_dict(base[key], value)
        else:
            base[key] = value
    return base

def main():
    base_cfg = yaml.safe_load(open("config/base.yaml"))

    # TO CHANGE SCENARIO FOR DIFFERENT TESTS CHANGE THIS PATH TO ONE OF THE SCENARIOS LOCATED IN THE CONFIG/ DIRECTORY
    scenario_cfg = yaml.safe_load(open("config/town10_first_env.yaml"))

    c_env = CarlaEnv(merge_dict(base_cfg, scenario_cfg))
    try:
        if c_env.config["debug"]["draw_spawn_points"]:
            c_env.draw_world_spawn_points()

        c_env.reset()

        spectator = c_env.world.get_spectator()
        transform = c_env.ego_vehicle.spawn_point
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20), carla.Rotation(pitch=-90)))

        npc_vehicles = spawn_vehicles(c_env.world, c_env.config["simulation"]["num_vehicles"])
        
        c_env.world.tick()
        if(c_env.config["simulation"]["npc_vehicle_ap"]):
            toggle_autopilot(npc_vehicles)

        c_env.world.tick()

        print("Simulation running... Press Ctrl+C to stop.")

        while True:
            time.sleep(c_env.config["simulation"]["time_step"])
            c_env.step_forward()
    except KeyboardInterrupt:
        print("Stopping simulation...")
    finally:
        print("Cleaning up...")
        c_env.cleanup()
        c_env.step_forward()


if __name__ == '__main__':
    main()