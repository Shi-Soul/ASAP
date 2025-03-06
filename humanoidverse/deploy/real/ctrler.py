






from typing import Union
import numpy as np
import time
import torch

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
from humanoidverse.utils.real.rotation_helper import get_gravity_orientation, transform_imu_data
from humanoidverse.utils.real.remote_controller import RemoteController, KeyMap
from humanoidverse.envs.env_utils.history_handler import HistoryHandler
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
from humanoidverse.utils.helpers import parse_observation

from typing import Dict
import numpy as np
from omegaconf import OmegaConf
from loguru import logger


from ..urcirobot import URCIRobot

class LowLevelMagic:
    def __init__(self, cfg: OmegaConf):
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
            breakpoint() # check the mode machine

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            raise NotImplementedError("Not implemented")
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
        self._wait()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=config.weak_motor)
            
    
    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.joystick.set(self.low_state.wireless_remote)
        
    def _wait(self):
        while self.low_state.tick == 0:
            time.sleep(0.01)
            print("Waiting for the robot to connect...")
        print("Successfully connected to the robot.")
        
    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)


class RealRobot(URCIRobot, LowLevelMagic):
    def __init__(self, cfg: OmegaConf):
        
        self.cfg = cfg
        self.device = "cpu"
        self.dt = cfg.deploy.ctrl_dt
        self.timer = 0
        
        self.num_actions = cfg.robot.actions_dim
        self.cmd = np.array([0, 0, 0, 0])
        
        self.clip_action_limit = cfg.robot.control.action_clip_value
        self.clip_observations = cfg.env.config.normalization.clip_observations
        self.action_scale = cfg.robot.control.action_scale
        
        logger.info("Initializing **Real** Robot")
        logger.info("Task Name: {}".format(cfg.log_task_name))
        logger.info("Robot Type: {}".format(cfg.robot.asset.robot_type))
        
        self._make_init_pose()
        self._make_buffer()
        if cfg.log_task_name == "motion_tracking":
            self.is_motion_tracking = True
            self._make_motionlib()
        else:
            self.is_motion_tracking = False
        
        super().__init__(cfg)
        
        
        raise NotImplementedError("Not implemented")
    
    def Reset(self):
        raise NotImplementedError("Not implemented")
    
    def ApplyAction(self, action:np.ndarray):
        raise NotImplementedError("Not implemented")
    
    def GetState(self):
        raise NotImplementedError("Not implemented")
    
    def Obs(self)->Dict[str, np.ndarray]:
        raise NotImplementedError("Not implemented")
    
    

    def UpdateObs(self):
        self.GetState()
        
        
        self.obs_buf_dict_raw = {}
        self.hist_obs_dict = {}
        
        noise_extra_scale = 1.
        for obs_key, obs_config in self.cfg.obs.obs_dict.items():
            if not obs_key=='actor_obs': continue
            self.obs_buf_dict_raw[obs_key] = dict()

            parse_observation(self, obs_config, self.obs_buf_dict_raw[obs_key], self.cfg.obs.obs_scales, self.cfg.obs.noise_scales, noise_extra_scale)
        
        # Compute history observations
        history_obs_list = self.history_handler.history.keys()
        parse_observation(self, history_obs_list, self.hist_obs_dict, self.cfg.obs.obs_scales, self.cfg.obs.noise_scales, noise_extra_scale)
        
        
        self.obs_buf_dict = dict()
        
        for obs_key, obs_config in self.cfg.obs.obs_dict.items():
            if not obs_key=='actor_obs': continue
            obs_keys = sorted(obs_config)
            # print("obs_keys", obs_keys)            
            self.obs_buf_dict[obs_key] = torch.cat([self.obs_buf_dict_raw[obs_key][key] for key in obs_keys], dim=-1)
            
            
        clip_obs = self.clip_observations
        for obs_key, obs_val in self.obs_buf_dict.items():
            if not obs_key=='actor_obs': continue
            self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs)

        for key in self.history_handler.history.keys():
            self.history_handler.add(key, self.hist_obs_dict[key])
            
    # TODO:
    def zero_torque_state(self):
        raise NotImplementedError("Not implemented")
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.joystick.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.dt)

    def move_to_default_pos(self):
        raise NotImplementedError("Not implemented")
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.dt)
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.dt)

    def default_pos_state(self):
        raise NotImplementedError("Not implemented")
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.joystick.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.dt)

    def run(self):
        raise NotImplementedError("Not implemented")
        self.counter += 1
        # Get the current joint position and velocity
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # create observation
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale
        period = 0.8
        count = self.counter * self.config.control_dt
        phase = count % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        self.cmd[0] = self.joystick.ly
        self.cmd[1] = self.joystick.lx * -1
        self.cmd[2] = self.joystick.rx * -1

        num_actions = self.config.num_actions
        self.obs[:3] = ang_vel
        self.obs[3:6] = gravity_orientation
        self.obs[6:9] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        self.obs[9 : 9 + num_actions] = qj_obs
        self.obs[9 + num_actions : 9 + num_actions * 2] = dqj_obs
        self.obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action
        self.obs[9 + num_actions * 3] = sin_phase
        self.obs[9 + num_actions * 3 + 1] = cos_phase

        # Get the action from the policy network
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        
        # transform action to target_dof_pos
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.dt)








