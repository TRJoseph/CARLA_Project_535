import random
import carla
import numpy as np
import cv2

class SEMCAM:
    def __init__(self, world, bp_id, host_actor, spawn_transform=None):
        self.world = world
        self.bp_lib = world.get_blueprint_library()
        self.bp = self.bp_lib.find(bp_id)
        self.host_actor = host_actor
        self.spawn_transform = spawn_transform


    # sem_cam = None
    # sem_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    # sem_bp.set_attribute("image_size_x",str(1920))
    # sem_bp.set_attribute("image_size_y",str(1080))
    # sem_bp.set_attribute("fov",str(105))
    # sem_location = carla.Location(2,0,1)
    # sem_rotation = carla.Rotation(0,180,0)
    # sem_transform = carla.Transform(sem_location,sem_rotation)
    # sem_cam = world.spawn_actor(sem_bp,sem_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
    # # This time, a color converter is applied to the image, to get the semantic segmentation view
    # sem_cam.listen(lambda image: image.save_to_disk('tutorial/new_sem_output/%.6d.jpg' % image.frame,carla.ColorConverter.CityScapesPalette))