import random
import carla
import numpy as np
import cv2

class RGBCam:
    def __init__(self, world, bp_id, host_actor, spawn_transform=None):
        self.world = world
        self.bp_lib = world.get_blueprint_library()
        self.bp = self.bp_lib.find(bp_id)
        self.host_actor = host_actor
        self.spawn_transform = spawn_transform


        # image settings
        self.save_to_disk = False
        self.show_cam = False
        self.IM_HEIGHT = 480
        self.IM_WIDTH = 640
        

        self.actor = None

    def process_image(self, image: carla.Image):
        if not self.save_to_disk:
            i = np.array(image.raw_data)
            i2 = i.reshape((self.IM_HEIGHT, self.IM_WIDTH, 4))
            i3 = i2[:, :, :3]
            if(self.show_cam):
                cv2.imshow("", i3)
                cv2.waitKey(1)
            return i3/255.0
        else:
            image.save_to_disk(f'out/{image.frame:06d}.png')
            return

    def spawn(self) -> carla.Actor:
        transform = self.spawn_transform
        self.actor = self.world.spawn_actor(self.bp, transform, attach_to=self.host_actor)
        return self.actor
    
    def start_listening(self):
        self.actor.listen(lambda image: self.process_image(image))
    
    def set_attribute(self, id: str, value) -> None:
        self.bp.set_attribute(id, f"{value}")
    
    def destroy(self):
        if self.actor:
            self.actor.stop()
            self.actor.destroy()
            self.actor = None