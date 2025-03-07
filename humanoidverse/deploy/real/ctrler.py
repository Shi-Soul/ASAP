






from typing import Union, List
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
np2torch = lambda x: torch.tensor(x, dtype=torch.float32)
torch2np = lambda x: x.cpu().numpy()


class LowLevelMagic:
    low_state: LowStateHG
    low_cmd: LowCmdHG
    joystick: RemoteController
    
    
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
        raise NotImplementedError("TODO: add sanity check")
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)


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
    def __init__(self, cfg: OmegaConf):
        
        self.cfg: OmegaConf = cfg
        self.device: str = "cpu"
        self.dt: float = cfg.deploy.ctrl_dt
        self.timer: int = 0
        
        self.num_real_dofs: int = len(self.low_cmd.motor_cmd)
        self.cmd: np.ndarray = np.array([0, 0, 0, 0])
        assert self.num_real_dofs == 29, "Only 29 dofs are supported for now"
        
        self.clip_action_limit: float = cfg.robot.control.action_clip_value
        self.clip_observations: float = cfg.env.config.normalization.clip_observations
        self.action_scale: float = cfg.robot.control.action_scale
        self.dof_idx_23_to_29: List[int] = cfg.deploy.dof_idx_23_to_29
        self.dof_idx_locked: List[int] = [i for i in range(23, 29) if i not in self.dof_idx_23_to_29]
        self.locked_kp: float = cfg.deploy.locked_kp
        self.locked_kd: float = cfg.deploy.locked_kd
        
        logger.info("Initializing **Real** Robot")
        logger.info("Task Name: {}".format(cfg.log_task_name))
        logger.info("Robot Type: {}".format(cfg.robot.asset.robot_type))
        
        self._make_init_pose()
        self._make_buffer()
        if cfg.log_task_name == "motion_tracking":
            self.is_motion_tracking: bool = True
            self._make_motionlib()
        else:
            self.is_motion_tracking: bool = False
        
        super().__init__(cfg)
        
        self.Reset()
        raise NotImplementedError("Not implemented")
    
    def _make_buffer(self):
        super()._make_buffer()
        
        self.q_real = np.zeros(self.num_real_dofs)
        self.dq_real = np.zeros(self.num_real_dofs)
        self.ddq_real = np.zeros(self.num_real_dofs)
        self.tau_est = np.zeros(self.num_real_dofs)

        self.kp_real = np.zeros(self.num_real_dofs)
        self.kd_real = np.zeros(self.num_real_dofs)
        
        # 填充有效关节的kp/kd
        for i, j in enumerate(self.dof_idx_23_to_29):
            self.kp_real[j] = self.kp[i]
            self.kd_real[j] = self.kd[i]
            
        # 填充锁定关节的kp/kd
        for j in self.dof_idx_locked:
            self.kp_real[j] = self.locked_kp
            self.kd_real[j] = self.locked_kd
    
    def GetState(self):
        # q,dq
        # self.tau_est = np.zeros(self.num_real_dofs)
        for i,j in enumerate(self.dof_idx_23_to_29):
            self.q[i] = self.low_state.motor_state[j].q
            self.dq[i] = self.low_state.motor_state[j].dq
        
        # quat, omega
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
        gravity_orientation = get_gravity_orientation(quat)
        
        # rpy = quaternion_to_euler_array(self.quat)
        # self.rpy[self.rpy > math.pi] -= 2 * math.pi
        
        raise NotImplementedError("Not implemented")
    
    def Reset(self):
        self.ToZeroTorque()
        self.ToDefaultPose()
        self.KeepDefaultPose()
        
        self.act[:] = 0
        self.history_handler.reset([0])
        self.timer: int = 0
        
        self.UpdateObs()
    
    def ApplyAction(self, action:np.ndarray):
        self.act = action.copy()
        target_q = np.clip(action, -self.clip_action_limit, self.clip_action_limit) * self.action_scale + self.dof_init_pose
        
        self.SetPose(target_q)
        self.UpdateObs()
    
    def Obs(self)->Dict[str, np.ndarray]:
        return {'actor_obs': torch2np(self.obs_buf_dict['actor_obs']).reshape(1, -1)}
    
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
            
        self.send_cmd(self.low_cmd)
    
    def ToZeroTorque(self):
        # raise NotImplementedError("Not implemented")
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.joystick.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.dt)
    
    def ToDefaultPose(self):
        print("Moving to default pos.")
        total_time: float = 2
        num_step = int(total_time / self.dt)
        
        start_dof_pos = np.zeros(self.num_real_dofs, dtype=np.float32)
        end_dof_pos = np.zeros(self.num_real_dofs, dtype=np.float32)# 锁定关节位置设为0
        
        for i, j in enumerate(self.dof_idx_23_to_29):
            end_dof_pos[j] = self.dof_init_pose[i]
            
        # 移动到默认姿态
        for i in range(num_step + 1):
            alpha = i / num_step
            for j in range(self.num_real_dofs):
                self.low_cmd.motor_cmd[j].q = start_dof_pos[j] * (1 - alpha) + end_dof_pos[j] * alpha
                self.low_cmd.motor_cmd[j].qd = 0
                self.low_cmd.motor_cmd[j].kp = self.kp_real[j]  # 直接使用预计算值
                self.low_cmd.motor_cmd[j].kd = self.kd_real[j]  # 直接使用预计算值
                self.low_cmd.motor_cmd[j].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.dt)
    def KeepDefaultPose(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        
        end_dof_pos = np.zeros(self.num_real_dofs, dtype=np.float32)
        
        # 初始化目标位置
        for i, j in enumerate(self.dof_idx_23_to_29):
            end_dof_pos[j] = self.dof_init_pose[i]
            
        # 保持默认姿态
        while self.joystick.button[KeyMap.A] != 1:
            for j in range(self.num_real_dofs):
                self.low_cmd.motor_cmd[j].q = end_dof_pos[j]
                self.low_cmd.motor_cmd[j].qd = 0
                self.low_cmd.motor_cmd[j].kp = self.kp_real[j]  # 直接使用预计算值
                self.low_cmd.motor_cmd[j].kd = self.kd_real[j]  # 直接使用预计算值
                self.low_cmd.motor_cmd[j].tau = 0
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








