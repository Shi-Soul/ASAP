






from typing import Union, List
import numpy as np
import time
import torch
import signal

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize # type: ignore
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize # type: ignore
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_ # type: ignore
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_ # type: ignore    
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG # type: ignore
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo # type: ignore
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG # type: ignore
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo # type: ignore
from unitree_sdk2py.utils.crc import CRC # type: ignore

from humanoidverse.utils.real.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from humanoidverse.utils.real.rotation_helper import get_gravity_orientation, transform_imu_data, quaternion_to_euler_array
from humanoidverse.utils.real.remote_controller import RemoteController, KeyMap
from humanoidverse.envs.env_utils.history_handler import HistoryHandler
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
from humanoidverse.utils.helpers import parse_observation
import math
from typing import Dict
import numpy as np
from omegaconf import OmegaConf
from loguru import logger
import os

from ..urcirobot import URCIRobot
np2torch = lambda x: torch.tensor(x, dtype=torch.float32)
torch2np = lambda x: x.cpu().numpy()
def wrap_to_pi_float(angles:float):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])




class LowLevelMagic:
    low_state: LowStateHG
    low_cmd: LowCmdHG
    joystick: RemoteController
    
    
    def __init__(self, cfg: OmegaConf):
        
        def signal_handler(sig, frame):
            logger.info("Ctrl+C  Exiting safely...")
            self.safe_exit()
        signal.signal(signal.SIGINT, signal_handler)
        
        config = cfg.deploy
        self.joystick = RemoteController()
        ChannelFactoryInitialize(0, config.net)
        
        
        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = config.mode_machine
            # self.mode_machine_ = 0
            # breakpoint() # check the mode machine

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.receive_state_handler, 10)

        elif config.msg_type == "go":
            raise ValueError("Not implemented for go msg type. This code is designed for hg msg type.")
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        # self._wait() #DEBUG:

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=config.weak_motor)
            
    def safe_exit(self):
        logger.info("Real Robot Exiting...")
        os._exit(1)
        
    def sanity_check(self):
        for motor_idx in range(len(self.low_cmd.motor_cmd)):
            if ((np.abs(self.low_cmd.motor_cmd[motor_idx].q - self.low_state.motor_state[motor_idx].q) > 1.5 ) or 
                (np.abs(self.low_cmd.motor_cmd[motor_idx].dq) > 20)):

                logger.error(f"Action of joint {motor_idx} is too large."
                             f"target q\t: {self.low_cmd.motor_cmd[motor_idx].q} "
                             f"target dq\t: {self.low_cmd.motor_cmd[motor_idx].dq} "
                             f"q\t\t: {self.low_state.motor_state[motor_idx].q} "
                             f"dq\t\t: {self.low_state.motor_state[motor_idx].dq}")
                self.safe_exit()
        
    def send_cmd(self):
        self.sanity_check()
        self.low_cmd.crc = CRC().Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

    def LowStateCallback(self):
        
        if self.joystick.button[KeyMap.B] == 1: # quick stop
            self.safe_exit()
        pass
    
    def receive_state_handler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.joystick.set(self.low_state.wireless_remote)
        
        self.sanity_check()
        self.LowStateCallback()
        
    def _wait(self):
        while self.low_state.tick == 0:
            time.sleep(0.1)
            logger.info("Waiting for the robot to connect...")
        logger.info("Successfully connected to the robot.")
        

    # @dataclass
    # @annotate.final
    # @annotate.autoid("sequential")
    # class LowCmd_(idl.IdlStruct, typename="unitree_hg.msg.dds_.LowCmd_"):
    #     mode_pr: types.uint8
    #     mode_machine: types.uint8
    #     motor_cmd: types.array['unitree_sdk2py.idl.unitree_hg.msg.dds_.MotorCmd_', 35]
    #     reserve: types.array[types.uint32, 4]
    #     crc: types.uint32
    
    
    # @dataclass
    # @annotate.final
    # @annotate.autoid("sequential")
    # class MotorCmd_(idl.IdlStruct, typename="unitree_hg.msg.dds_.MotorCmd_"):
    #     mode: types.uint8
    #     q: types.float32
    #     dq: types.float32
    #     tau: types.float32
    #     kp: types.float32
    #     kd: types.float32
    #     reserve: types.uint32
    
    # 
    # @dataclass
    # @annotate.final
    # @annotate.autoid("sequential")
    # class LowState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.LowState_"):
    #     version: types.array[types.uint32, 2]
    #     mode_pr: types.uint8
    #     mode_machine: types.uint8
    #     tick: types.uint32
    #     imu_state: 'unitree_sdk2py.idl.unitree_hg.msg.dds_.IMUState_'
    #     motor_state: types.array['unitree_sdk2py.idl.unitree_hg.msg.dds_.MotorState_', 35]
    #     wireless_remote: types.array[types.uint8, 40]
    #     reserve: types.array[types.uint32, 4]
    #     crc: types.uint32


    # @dataclass
    # @annotate.final
    # @annotate.autoid("sequential")
    # class MotorState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.MotorState_"):
    #     mode: types.uint8
    #     q: types.float32
    #     dq: types.float32
    #     ddq: types.float32
    #     tau_est: types.float32
    #     temperature: types.array[types.int16, 2]
    #     vol: types.float32
    #     sensor: types.array[types.uint32, 2]
    #     motorstate: types.uint32
    #     reserve: types.array[types.uint32, 4]



class RealRobot(URCIRobot, LowLevelMagic):
    REAL=True
    
    def __init__(self, cfg: OmegaConf):
        URCIRobot.__init__(self, cfg)
        LowLevelMagic.__init__(self, cfg)
        
        
        
        self.num_real_dofs: int = 29
        assert self.num_real_dofs == 29, "Only 29 dofs are supported for now"
        
        self.dof_idx_23_to_29: List[int] = cfg.deploy.dof_idx_23_to_29
        self.dof_idx_locked: List[int] = [i for i in range(0, 29) if i not in self.dof_idx_23_to_29]
        self.locked_kp: float = cfg.deploy.locked_kp
        self.locked_kd: float = cfg.deploy.locked_kd
        
        logger.info("Initializing **Real** Robot")
        logger.info("Task Name: {}".format(cfg.log_task_name))
        logger.info("Robot Type: {}".format(cfg.robot.asset.robot_type))
        
        self.Reset()
    
    def _make_buffer(self):
        super()._make_buffer()
        
        self.q_real = np.zeros(self.num_real_dofs)
        self.dq_real = np.zeros(self.num_real_dofs)
        self.ddq_real = np.zeros(self.num_real_dofs)
        self.tau_est = np.zeros(self.num_real_dofs)

        self.kp_real = np.zeros(self.num_real_dofs)
        self.kd_real = np.zeros(self.num_real_dofs)
        
        self.dof_init_pose_real = np.zeros(self.num_real_dofs)
        
        # 填充有效关节的kp/kd
        for i, j in enumerate(self.dof_idx_23_to_29):
            self.kp_real[j] = self.kp[i]
            self.kd_real[j] = self.kd[i]
            self.dof_init_pose_real[j] = self.dof_init_pose[i]
            
        # 填充锁定关节的kp/kd
        for j in self.dof_idx_locked:
            self.kp_real[j] = self.locked_kp
            self.kd_real[j] = self.locked_kd
    
    def GetState(self):
        for j in range(self.num_real_dofs):
            self.q_real[j] = self.low_state.motor_state[j].q
            self.dq_real[j] = self.low_state.motor_state[j].dq
            self.ddq_real[j] = self.low_state.motor_state[j].ddq
            self.tau_est[j] = self.low_state.motor_state[j].tau_est
            
        self.q = self.q_real[self.dof_idx_23_to_29]
        self.dq = self.dq_real[self.dof_idx_23_to_29] 
        breakpoint() # check the grammar
        
        # quat, omega
        self.quat = self.low_state.imu_state.quaternion
        self.omega = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
        self.gvec = get_gravity_orientation(self.quat)
        
        self.rpy = quaternion_to_euler_array(self.quat)
        self.rpy[self.rpy > math.pi] -= 2 * math.pi
        
        
        self.cmd[0] = self.joystick.ly *0.3
        self.cmd[1] = self.joystick.lx * -1 * 0.3
        self.cmd[3] += self.joystick.ry * -1 * self.dt * 0.1
        
        if self.heading_cmd:
            self.cmd[2] = np.clip(0.5*wrap_to_pi_float(self.cmd[3] - self.rpy[2]), -1., 1.)
        else:
            self.cmd[2] = self.joystick.rx * -1 * 0.3
            
        if self.is_motion_tracking:
            self.motion_time = (self.timer) * self.dt 
            self.ref_motion_phase = self.motion_time / self.motion_len
            
        breakpoint() # check the syntax of self.cmd
        
        
        breakpoint() # check the valid value 
        # TODO: add real-time logging for the state
    
    def Reset(self):
        self.ToZeroTorque()
        self.MoveToDefaultPose()
        self.KeepDefaultPose()
        
        self.act[:] = 0
        self.history_handler.reset([0])
        self.timer: int = 0
        
        self.UpdateObs()
    
    def ApplyAction(self, action:np.ndarray):
        self.timer+=1
        self.act = action.copy()
        target_q = np.clip(action, -self.clip_action_limit, self.clip_action_limit) * self.action_scale + self.dof_init_pose
        
        self.SetPose(target_q)
        self.UpdateObs()
    
    def Obs(self)->Dict[str, np.ndarray]:
        return {'actor_obs': torch2np(self.obs_buf_dict['actor_obs']).reshape(1, -1)}
            
    
    def LowStateCallback(self):
        """
            B: low level Quick Stop
            X: high level Reset
            
            lx,ly: velocity command
            rx: yaw directino command
        """
        # handle joystick keyboard
        LowLevelMagic.LowStateCallback(self)
            
        
        if self.joystick.button[KeyMap.X] == 1: # quick stop
            self._ref_pid = -2
            
        pass
            
    # self.low_cmd.motor_cmd is designed to be **Stateless**
    # Each time you call the following CMD function,
    # you should assume that the previous low_cmd is randomly generated
    def SetPose(self, pose: np.ndarray):
        assert pose.shape == (self.num_dofs,)
        
        for i, j in enumerate(self.dof_idx_23_to_29):
            self.low_cmd.motor_cmd[j].q = pose[i]
            self.low_cmd.motor_cmd[j].qd = 0
            self.low_cmd.motor_cmd[j].tau = 0
            self.low_cmd.motor_cmd[j].kp = self.kp_real[j]  # 使用预计算值
            self.low_cmd.motor_cmd[j].kd = self.kd_real[j]  # 使用预计算值
            
        # 锁定关节的kp/kd
        for j in self.dof_idx_locked:
            self.low_cmd.motor_cmd[j].q = 0
            self.low_cmd.motor_cmd[j].qd = 0
            self.low_cmd.motor_cmd[j].tau = 0
            self.low_cmd.motor_cmd[j].kp = self.kp_real[j]  # 使用预计算值
            self.low_cmd.motor_cmd[j].kd = self.kd_real[j]  # 使用预计算值
            
        self.send_cmd()
    
    def ToZeroTorque(self):
        logger.info("Enter zero torque state.")
        logger.info("Waiting for the start signal...")
        while self.joystick.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd()
            time.sleep(self.dt)
    
    MoveToDefaultPose = lambda self: self.MoveToPose29Dof(self.dof_init_pose_real, 2.0)
    KeepDefaultPose = lambda self: self.KeepPose29Dof(self.dof_init_pose_real)

    def MoveToPose29Dof(self, target_pose: np.ndarray, duration: float):
        logger.info("Moving to specified pose.")
        num_step = int(duration / self.dt)
        
        self.GetState()
        start_dof_pos = self.q_real.copy()
        end_dof_pos = target_pose.copy()  # Use the provided target pose
        
        # Move to the specified pose
        for i in range(num_step + 1):
            alpha = i / num_step
            for j in range(self.num_real_dofs):
                self.low_cmd.motor_cmd[j].q = start_dof_pos[j] * (1 - alpha) + end_dof_pos[j] * alpha
                self.low_cmd.motor_cmd[j].qd = 0
                self.low_cmd.motor_cmd[j].kp = self.kp_real[j]  # Use pre-calculated values
                self.low_cmd.motor_cmd[j].kd = self.kd_real[j]  # Use pre-calculated values
                self.low_cmd.motor_cmd[j].tau = 0
            self.send_cmd()
            time.sleep(self.dt)

    def KeepPose29Dof(self, pose: np.ndarray):
        logger.info("Entering pose hold state.")
        logger.info("Waiting for the Button A signal...")
        
        # Maintain the current pose
        current_pose = pose.copy()
        
        while self.joystick.button[KeyMap.A] != 1:
            for j in range(self.num_real_dofs):
                self.low_cmd.motor_cmd[j].q = current_pose[j]
                self.low_cmd.motor_cmd[j].qd = 0
                self.low_cmd.motor_cmd[j].kp = self.kp_real[j]  # Use pre-calculated values
                self.low_cmd.motor_cmd[j].kd = self.kd_real[j]  # Use pre-calculated values
                self.low_cmd.motor_cmd[j].tau = 0
            self.send_cmd()
            time.sleep(self.dt)







