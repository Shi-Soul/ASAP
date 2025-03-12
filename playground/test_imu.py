


from unitree_dds_wrapper.idl import unitree_hg
from unitree_dds_wrapper.publisher import Publisher
from unitree_dds_wrapper.subscription import Subscription
from unitree_dds_wrapper.utils.crc import crc32
import numpy as np
import struct
import copy
import threading
import time
from termcolor import colored
import os
import sys
import math
from scipy.spatial.transform import Rotation as R
import datetime
import pickle
import signal
from unitree_dds_wrapper.utils.joystick import LogicJoystick, Joystick


topic = "rt/lowstate"


# unitree_dds_wrapper/python/unitree_dds_wrapper/utils/joystick.py
lowstate_subscriber = Subscription(unitree_hg.msg.dds_.LowState_, topic)



# class IMUState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.IMUState_"):
#     quaternion: types.array[types.float32, 4]
#     gyroscope: types.array[types.float32, 3]
#     accelerometer: types.array[types.float32, 3]
#     rpy: types.array[types.float32, 3]
#     temperature: types.int16
# 


while True:
    if lowstate_subscriber.msg is not None:
        imu_state = lowstate_subscriber.msg.imu_state
        # print(imu_state.rpy)
        print(f"{imu_state.rpy[0]:.6f} | {imu_state.rpy[1]:.6f} | {imu_state.rpy[2]:.6f}")
        # 这个rpy的 zero pos 应该是和开机时的base坐标系绑定的, IMU绑在torso上
        
        
    time.sleep(0.0003)

