


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


joy = Joystick()
# unitree_dds_wrapper/python/unitree_dds_wrapper/utils/joystick.py
lowstate_subscriber = Subscription(unitree_hg.msg.dds_.LowState_, topic)


"""

    # Buttons
    self.back = Button()  -> SELECT
    self.start = Button() -> START
    self.LS = Button() 
    self.RS = Button()  
    self.LB = Button()  -> L1
    self.RB = Button()  -> R1
    self.A = Button()
    self.B = Button()
    self.X = Button()
    self.Y = Button()
    self.up = Button() 
    self.down = Button()
    self.left = Button()
    self.right = Button()
    self.F1 = Button() -> 不知道
    self.F2 = Button() -> 不知道

    # Axes
    self.LT = Axis() -> L2
    self.RT = Axis() -> R2
    self.lx = Axis() 
    self.ly = Axis() 
    self.rx = Axis() 
    self.ry = Axis() 
"""


while True:
    if lowstate_subscriber.msg is not None:
        # print(lowstate_subscriber.msg.wireless_remote)
        joy.extract(lowstate_subscriber.msg.wireless_remote)
        print(f"lx={joy.lx.data:.06f} ly={joy.ly.data:.06f}  rx={joy.rx.data:.06f} ry={joy.ry.data:.06f} A={joy.A.pressed}")
        # print(f"lt={joy.LT.data:.06f} rt={joy.RT.data:.06f}")
        # print(joy.left.pressed)print(joy.left.pressed)
        # print(joy.left.pressed, joy.right.pressed, joy.up.pressed, joy.down.pressed)
        # print(joy.LS.pressed, joy.RS.pressed, joy.LB.pressed, joy.RB.pressed)
        # print(joy.back.pressed, joy.start.pressed, joy.F1.pressed, joy.F2.pressed)
        
        # LB = L1, RB= R1; LS, RS 不是L2 R2
        # L2 = LT, R2 = RT
        # back = select, start=start, F1,F2 不知道是什么
    time.sleep(0.0003)

