'''
file: donkey_sim.py
author: Tawn Kramer
date: 2018-08-31
'''

import time
import math
import logging
import base64
from threading import Thread
from io import BytesIO
import types

import numpy as np
from PIL import Image

from donkey_gym.core.fps import FPSTimer
from donkey_gym.core.message import IMesgHandler
from donkey_gym.core.sim_client import SimClient
from donkey_gym.envs.donkey_ex import SimFailed

from config import INPUT_DIM, ROI, THROTTLE_REWARD_WEIGHT, MAX_THROTTLE, MIN_THROTTLE, \
    REWARD_CRASH, CRASH_SPEED_WEIGHT

logger = logging.getLogger(__name__)


class DonkeyUnitySimContoller():
    """
    Wrapper for communicating with unity simulation.

    :param level: (int) Level index
    :param port: (int) Port to use for communicating with the simulator
    :param max_cte_error: (float) Max cross track error before reset
    """
    
    def __init__(self, conf):

        logger.setLevel(conf["log_level"])

        self.address = (conf["host"], conf["port"])

        self.handler = DonkeyUnitySimHandler(conf=conf)

        self.client = SimClient(self.address, self.handler)

    def set_car_config(self, body_style, body_rgb, car_name, font_size):
        self.handler.send_car_config(body_style, body_rgb, car_name, font_size)

    def set_cam_config(self, **kwargs):
        self.handler.send_cam_config(**kwargs)

    def set_reward_fn(self, reward_fn):
        self.handler.set_reward_fn(reward_fn)

    def set_episode_over_fn(self, ep_over_fn):
        self.handler.set_episode_over_fn(ep_over_fn)

    def wait_until_loaded(self):
        while not self.handler.loaded:
            logger.warning("waiting for sim to start..")
            time.sleep(3.0)

    def reset(self):
        self.handler.reset()

    def get_sensor_size(self):
        """
        :return: (int, int, int)
        """
        return self.handler.get_sensor_size()

    def take_action(self, action):
        self.handler.take_action(action)

    def observe(self):
        """
        :return: (np.ndarray)
        """
        return self.handler.observe()

    def quit(self):
        self.client.stop()

    def render(self, mode):
        pass

    def is_game_over(self):
        return self.handler.is_game_over()

    def calc_reward(self, done):
        return self.handler.calc_reward(done)


class DonkeyUnitySimHandler(IMesgHandler):
    """
    Socket message handler.

    :param level: (int) Level ID
    :param max_cte_error: (float) Max cross track error before reset
    """
    def __init__(self, conf):
        self.conf = conf
        self.iSceneToLoad = conf["level"]
        self.loaded = False
        self.max_cte = conf["max_cte"]
        # VAE
        #self.timer = FPSTimer()
        self.timer = FPSTimer(N=0)

        # sensor size - height, width, depth TODO cehck with INPUT_DIM
        #self.camera_img_size = conf["cam_resolution"]
        self.camera_img_size = INPUT_DIM
        self.image_array = np.zeros(self.camera_img_size)
        # VAE
        self.verbose = True
        self.last_obs = None
        self.last_throttle = 0.0
        
        self.hit = "none"
        self.cte = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.speed = 0.0
        self.over = False
        # VAE
        self.steering_angle = 0.0
        self.current_step = 0
        self.speed = 0
        self.steering = None

        self.fns = {'telemetry': self.on_telemetry,
                    "scene_selection_ready": self.on_scene_selection_ready,
                    "scene_names": self.on_recv_scene_names,
                    "car_loaded": self.on_car_loaded,
                    "cross_start": self.on_cross_start,
                    "race_start": self.on_race_start,
                    "race_stop": self.on_race_stop,
                    "DQ": self.on_DQ,
                    "ping": self.on_ping,
                    "aborted": self.on_abort,
                    "need_car_config": self.on_need_car_config}

    def on_connect(self, client):
        logger.debug("socket connected")
        self.client = client

    def on_disconnect(self):
        logger.debug("socket disconnected")
        self.client = None

    def on_abort(self, message):
        self.client.stop()

    def on_need_car_config(self, message):
        logger.info("on need car config")
        self.loaded = True
        self.send_config(self.conf)

    def extract_keys(self, dct, lst):
        ret_dct = {}
        for key in lst:
            if key in dct:
                ret_dct[key] = dct[key]
        return ret_dct

    def send_config(self, conf):
        logger.info("sending car config.")
        self.set_car_config(conf)
        self.set_racer_bio(conf)
        cam_config = self.extract_keys(conf, ["img_w", "img_h", "img_d", "img_enc", "fov", "fish_eye_x", "fish_eye_y", "offset_x", "offset_y", "offset_z", "rot_x"])
        self.send_cam_config(**cam_config)
        logger.info("done sending car config.")

    def set_car_config(self, conf):
        if "body_style" in conf :
            self.send_car_config(conf["body_style"], conf["body_rgb"], conf["car_name"], conf["font_size"])

    def set_racer_bio(self, conf):
        self.conf = conf
        if "bio" in conf :
            self.send_racer_bio(conf["racer_name"], conf["car_name"], conf["bio"], conf["country"])
        else :
            print("no racer bio in conf")

    def on_recv_message(self, message):
        if 'msg_type' not in message:
            logger.warn('expected msg_type field')
            return
        msg_type = message['msg_type']
        logger.debug("got message :" + msg_type)
        if msg_type in self.fns:
            self.fns[msg_type](message)
        else:
            logger.warning(f'unknown message type {msg_type}')

    ## ------- Env interface ---------- ##

    def reset(self):
        logger.debug("reseting")
        if self.verbose:
            print("resetting")
        self.send_control(0.0, 0.0)
        self.send_reset_car()
        self.timer.reset()
        time.sleep(1)
        self.image_array = np.zeros(self.camera_img_size)
        self.last_obs = self.image_array
        self.hit = "none"
        self.cte = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.speed = 0.0
        self.over = False

        # VAE
        #self.last_obs = None
        self.current_step = 0
        self.steering_angle = 0.0
        self.last_throttle = 0.0

        #self.speed = 0
        self.steering = None


    def get_sensor_size(self):
        return self.camera_img_size

    def take_action(self, action):
        # VAE
        if self.verbose:
            print("act", end='')

        throttle = action[1]
        self.last_throttle = throttle
        self.current_step += 1

        self.send_control(action[0], action[1])

    def observe(self):
        while self.last_obs is self.image_array:
            time.sleep(1.0 / 120.0)

        self.last_obs = self.image_array
        observation = self.image_array
        done = self.is_game_over()
        reward = self.calc_reward(done)
        # VAE
        info = {'pos': (self.x, self.y, self.z), 'cte': self.cte, 'speed': self.speed, 'hit': self.hit}
        #info = {}

        #self.timer.on_frame()

        return observation, reward, done, info

    def is_game_over(self):
        return self.over

    ## ------ RL interface ----------- ##

    def set_reward_fn(self, reward_fn):
        """
        allow users to set their own reward function
        """
        self.calc_reward = types.MethodType(reward_fn, self)
        logger.debug("custom reward fn set.")

    def calc_reward(self, done):
        '''
        if done:
            return -1.0

        if self.cte > self.max_cte:
            return -1.0

        if self.hit != "none":
            return -2.0
        '''
        # going fast close to the center of lane yeilds best reward
        return (1.0 - (math.fabs(self.cte) / self.max_cte)) * self.speed
        
        '''
        # VAE
        if done:
            # penalize the agent for getting off the road fast
            norm_throttle = (self.last_throttle - MIN_THROTTLE) / (MAX_THROTTLE - MIN_THROTTLE)
            return REWARD_CRASH - CRASH_SPEED_WEIGHT * norm_throttle
        # 1 per timesteps + throttle
        throttle_reward = THROTTLE_REWARD_WEIGHT * (self.last_throttle / MAX_THROTTLE)
        return 1 + throttle_reward
        '''

    ## ------ Socket interface ----------- ##

    def on_telemetry(self, data):

        if self.verbose:
            #print("telemetry msg rcv'd")
            print('.', end='')

        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))

        # always update the image_array as the observation loop will hang if not changing.
        #VAE self.image_array = np.asarray(image)

        # Resize and crop image
        image = np.array(image)
        # Save original image for render
        self.original_image = np.copy(image)
        # Resize if using higher resolution images
        # image = cv2.resize(image, CAMERA_RESOLUTION)
        # Region of interest
        r = ROI
        image = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        # Convert RGB to BGR
        image = image[:, :, ::-1]
        self.image_array = image
        # Here resize is not useful for now (the image have already the right dimension)
        # self.image_array = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        self.x = data["pos_x"]
        self.y = data["pos_y"]
        self.z = data["pos_z"]
        self.speed = data["speed"]

        # VAE
        self.steering_angle = data['steering_angle']

        # Cross track error not always present.
        # Will be missing if path is not setup in the given scene.
        # It should be setup in the 4 scenes available now.
        if "cte" in data:
            self.cte = data["cte"]

        # don't update hit once session over
        if self.over:
            return

        self.hit = data["hit"]

        #don't have to, but to clean up the print, delete the image string.
        del data["image"]


        self.determine_episode_over()

    def on_cross_start(self, data):        
        logger.info(f"crossed start line: lap_time {data['lap_time']}")

    def on_race_start(self, data):
        logger.debug(f"race started")

    def on_race_stop(self, data):
        logger.debug(f"race stoped")

    def on_DQ(self, data):
        logger.info(f"racer DQ")
        self.over = True

    def on_ping(self, message):
        """
        no reply needed at this point. Server sends these as a keep alive to make sure clients haven't gone away.
        """
        pass

    def set_episode_over_fn(self, ep_over_fn):
        """
        allow userd to define their own episode over function
        """
        self.determine_episode_over = types.MethodType(ep_over_fn, self)
        logger.debug("custom ep_over fn set.")

    def determine_episode_over(self):
        # we have a few initial frames on start that are sometimes very large CTE when it's behind
        # the path just slightly. We ignore those.
        if math.fabs(self.cte) > 2 * self.max_cte:
            pass
        elif math.fabs(self.cte) > self.max_cte:
            logger.debug(f"game over: cte {self.cte}")
            self.over = True
        elif self.hit != "none":
            logger.debug(f"game over: hit {self.hit}")
            self.over = True

    def on_scene_selection_ready(self, data):
        logger.debug("SceneSelectionReady ")
        self.send_get_scene_names()

    def on_car_loaded(self, data):
        logger.debug("car loaded")
        self.loaded = True

    def on_recv_scene_names(self, data):
        if data:
            names = data['scene_names']
            logger.debug(f"SceneNames: {names}")
            if self.verbose:
                print("SceneNames:", names)
                print("loading scene", self.iSceneToLoad, names[self.iSceneToLoad])
            self.send_load_scene(names[self.iSceneToLoad])

    def send_control(self, steer, throttle):
        if not self.loaded:
            return
        msg = {'msg_type': 'control', 'steering': steer.__str__(), 'throttle': throttle.__str__(), 'brake': '0.0'}
        #msg = {'msg_type': 'control', 'steering': '0.0', 'throttle': '0.5', 'brake': '0.0'} #testing
        self.queue_message(msg)

    def send_reset_car(self):
        msg = {'msg_type': 'reset_car'}
        self.queue_message(msg)

    def send_get_scene_names(self):
        msg = {'msg_type': 'get_scene_names'}
        self.queue_message(msg)

    def send_load_scene(self, scene_name):
        if self.verbose:
            print("Car Loaded")
        msg = {'msg_type': 'load_scene', 'scene_name': scene_name}
        self.queue_message(msg)

    def send_car_config(self, body_style, body_rgb, car_name, font_size):
        """
        # body_style = "donkey" | "bare" | "car01" choice of string
        # body_rgb  = (128, 128, 128) tuple of ints
        # car_name = "string less than 64 char"
        """
        msg = {'msg_type': 'car_config',
            'body_style': body_style,
            'body_r' : body_rgb[0].__str__(),
            'body_g' : body_rgb[1].__str__(),
            'body_b' : body_rgb[2].__str__(),
            'car_name': car_name,
            'font_size' : font_size.__str__() }
        self.blocking_send(msg)
        time.sleep(0.1)

    def send_racer_bio(self, racer_name, car_name, bio, country):
        # body_style = "donkey" | "bare" | "car01" choice of string
        # body_rgb  = (128, 128, 128) tuple of ints
        # car_name = "string less than 64 char"
        msg = {'msg_type': 'racer_info',
            'racer_name': racer_name,
            'car_name' : car_name,
            'bio' : bio,
            'country' : country }
        self.blocking_send(msg)
        time.sleep(0.1)

    def send_exit_scene(self):
        """
        Go back to scene selection.
        """
        msg = {'msg_type': 'exit_scene'}
        self.blocking_send(msg)

    def send_cam_config(self, img_w=0, img_h=0, img_d=0, img_enc=0, fov=0, fish_eye_x=0, fish_eye_y=0, offset_x=0, offset_y=0, offset_z=0, rot_x=0):
        """ Camera config
            set any field to Zero to get the default camera setting.
            offset_x moves camera left/right
            offset_y moves camera up/down
            offset_z moves camera forward/back
            rot_x will rotate the camera
            with fish_eye_x/y == 0.0 then you get no distortion
            img_enc can be one of JPG|PNG|TGA
        """
        msg = {"msg_type" : "cam_config",
               "fov" : str(fov),
               "fish_eye_x" : str(fish_eye_x),
               "fish_eye_y" : str(fish_eye_y),
               "img_w" : str(img_w),
               "img_h" : str(img_h),
               "img_d" : str(img_d),
               "img_enc" : str(img_enc),
               "offset_x" : str(offset_x),
               "offset_y" : str(offset_y),
               "offset_z" : str(offset_z),
               "rot_x" : str(rot_x) }
        self.blocking_send(msg)
        time.sleep(0.1)

    def blocking_send(self, msg):
        if self.client is None:
            logger.debug(f'skiping: \n {msg}')
            return

        logger.debug(f'blocking send \n {msg}')
        self.client.send_now(msg)

    def queue_message(self, msg):
        if self.client is None:
            logger.debug(f'skiping: \n {msg}')
            return

        logger.debug(f'sending queued \n {msg}')

        # if self.verbose:
        #     print('sending queued', msg)
        self.client.queue_message(msg)
