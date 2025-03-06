# Unified Robot Control Interface for HumanoidVerse
# Weiji Xie @ 2025.03.04

REAL       :bool    = False
BYPASS_ACT :bool    = False
HEADING_CMD:bool    = True

LOG        :bool    = True
RENDER     :bool    = True
PLOT       :bool    = False
defcmd = [0.0, 0.0, 0.0, 0.0]


if not REAL:
    import mujoco, mujoco_viewer
    import glfw 
import os
import sys
from pathlib import Path

import torch
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from humanoidverse.utils.logging import HydraLoggerBridge
from humanoidverse.envs.env_utils.history_handler import HistoryHandler
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
from humanoidverse.utils.helpers import parse_observation
from humanoidverse.deploy import URCIRobot
from scipy.spatial.transform import Rotation as R
import logging
from utils.config_utils import *  # noqa: E402, F403
# add argparse arguments

from typing import Dict
from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from loguru import logger


import onnxruntime as ort
import numpy as np


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


class MujocoRobot(URCIRobot):
    ACT_EMA: bool = False # Noise
    RAND_NOISE: bool = False
    RAND_DELAY: bool = False
    RAND_MASK : bool = False
    
    ema_alpha = 0.1  # EMA smoothing factor
    noise_ratio = 3e-2
    delay_ratio = (0.4, 0.8)
    # delay_ratio = (0.5, 1.5)
    # delay_ratio = (0.5, 2.0)
    mask_ratio = 0.8
    mk_rand_noise = lambda tens, ratio: tens * (1 + np.random.randn(*tens.shape).astype(tens.dtype) * ratio) # type: ignore
    print_torque = lambda tau: print(f"tau (norm, max) = {np.linalg.norm(tau):.2f}, \t{np.max(tau):.2f}", end='\r')
    
    def __init__(self, cfg):
        self.cfg:OmegaConf = cfg
        self.device="cpu"
        
        self.decimation = cfg.simulator.config.sim.control_decimation
        self.sim_dt = 1/cfg.simulator.config.sim.fps
        self.dt = self.decimation * self.sim_dt
        self.subtimer = 0
        
        self.clip_action_limit = cfg.robot.control.action_clip_value
        self.clip_observations = cfg.env.config.normalization.clip_observations
        self.action_scale = cfg.robot.control.action_scale
        
        
        self.model = mujoco.MjModel.from_xml_path(os.path.join(cfg.robot.asset.asset_root, cfg.robot.asset.xml_file)) # type: ignore
        # self.model = mujoco.MjModel.from_xml_path('/home/bai/ASAP/humanoidverse/data/robots/g1/g1_23dof_lock_wrist_phys.xml')
        # self.model = mujoco.MjModel.from_xml_path('/home/bai/ASAP/humanoidverse/data/robots/g1_asap/g1_29dof_anneal_23dof.xml')
        self.data = mujoco.MjData(self.model) # type: ignore
        self.model.opt.timestep = self.sim_dt
        
        self.cmd = np.array(defcmd)
        self.num_actions = cfg.robot.actions_dim
        self.num_ctrl = self.data.ctrl.shape[0]
        assert self.num_ctrl == self.num_actions, f"Number of control DOFs {self.num_ctrl} does not match number of actions {self.num_actions}"
        
        logger.info("Initializing Mujoco Robot")
        logger.info("Task Name: {}".format(cfg.log_task_name))
        logger.info("Robot Type: {}".format(cfg.robot.asset.robot_type))
        
        self._make_init_pose()
        self._make_buffer()
        if cfg.log_task_name == "motion_tracking":
            self.is_motion_tracking = True
            self._make_motionlib()
        else:
            self.is_motion_tracking = False
        
        self.Reset()
        
        mujoco.mj_step(self.model, self.data) # type: ignore    
        if RENDER:
            self.__make_viewer()

    def __make_viewer(self):
        ...
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.cam.lookat[:] = np.array([0,0,0.8])
        self.viewer.cam.distance = 3.0        
        self.viewer.cam.azimuth = 30                         # 可根据需要调整角度
        self.viewer.cam.elevation = -30                      # 负值表示从上往下看
        def _key_callback(window, key, scancode, action, mods):
            if action == glfw.PRESS:
                #  Keyboard mapping:
                #  ----------------
                #  |   K L ; '    |
                #  |   , . /      |
                #  ----------------

                if key == glfw.KEY_COMMA:
                    self.cmd[1] += 0.1
                elif key == glfw.KEY_SLASH:
                    self.cmd[1] -= 0.1
                elif key == glfw.KEY_L: #
                    self.cmd[0] += 0.1
                elif key == glfw.KEY_PERIOD:
                    self.cmd[0] -= 0.1
                elif key == glfw.KEY_K:
                    if HEADING_CMD:
                        self.cmd[3] = wrap_to_pi_float(self.cmd[3]+np.pi/20)
                    else:
                        self.cmd[2] += 0.1
                elif key == glfw.KEY_SEMICOLON:
                    if HEADING_CMD:                            
                        self.cmd[3] = wrap_to_pi_float(self.cmd[3]-np.pi/20)
                    else:
                        self.cmd[2] -= 0.1
                elif key == glfw.KEY_APOSTROPHE:
                    self.cmd = np.array(defcmd)
                elif key == glfw.KEY_ENTER:
                    self.Reset()
                print(self.cmd)
            self.viewer._key_callback(window, key, scancode, action, mods)
        glfw.set_key_callback(self.viewer.window, _key_callback)

    def Reset(self):
        # raise NotImplementedError("Not implemented")
        self.data.qpos[:3] = np.array(self.cfg.robot.init_state.pos)
        self.data.qpos[3:7] = np.array(self.cfg.robot.init_state.rot)
        self.data.qpos[7:] = self.dof_init_pose
        self.data.qvel[:]   = 0
        
        self.act[:] = 0
        self.history_handler.reset([0])

        self.subtimer = 0
        
        self.GetState()
        ...

    @staticmethod
    def pd_control(target_q, q, kp, target_dq, dq, kd):
        '''Calculates torques from position commands
        '''
        # if MujocoRobot.RAND_NOISE:
        #     kp,kd = MujocoRobot.mk_rand_noise(np.array([kp, kd]), MujocoRobot.noise_ratio)
        return (target_q - q) * kp + (target_dq - dq) * kd

    def GetState(self):
        '''Extracts physical states from the mujoco data structure
        '''
        data = self.data
        self.q = data.qpos.astype(np.double)[7:] # 19 dim
            # 3 dim base pos + 4 dim quat + 12 dim actuation angles
        self.dq = data.qvel.astype(np.double)[6:] # 18 dim ?????
            # 3 dim base vel + 3 dim omega + 12 dim actuation vel
        
        self.pos = data.qpos.astype(np.double)[:3]
        self.quat = data.qpos.astype(np.double)[3:7]
        self.vel = data.qvel.astype(np.double)[:3]
        self.omega = data.qvel.astype(np.double)[3:6]
        
        r = R.from_quat(self.quat)
        self.rpy = quaternion_to_euler_array(self.quat)
        self.rpy[self.rpy > math.pi] -= 2 * math.pi
        self.gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    
        
        if HEADING_CMD:
            self.cmd[2] = np.clip(0.5*wrap_to_pi_float(self.cmd[3] - self.rpy[2]), -1., 1.)
            
        if self.is_motion_tracking:
            self.motion_time = (self.subtimer/self.decimation) * self.dt 
            self.ref_motion_phase = self.motion_time / self.motion_len
            
        # breakpoint()
    
        
    def ApplyAction(self, action): 
        self.act = action.copy()
        target_q = np.clip(action, -self.clip_action_limit, self.clip_action_limit) * self.action_scale + self.dof_init_pose
        
        for i in range(self.decimation):
            self.GetState()
            
            tau = self.pd_control(target_q, self.q, self.kp,
                            0, self.dq, self.kd)  # Calc torques
            
            # if self.RAND_NOISE: tau = MujocoRobot.mk_rand_noise(tau, MujocoRobot.noise_ratio)
            tau = np.clip(tau, -self.tau_limit, self.tau_limit)  # Clamp torques
            
            
            # self.print_torque(tau)
            # tau*=0
            # print(np.linalg.norm(target_q-self.q), np.linalg.norm(self.dq), np.linalg.norm(tau))
            # self.data.qpos[:3] = np.array([0,0,1])
            self.data.ctrl[:] = tau

            mujoco.mj_step(self.model, self.data) # type: ignore
            
            if np.any(self.data.contact.pos[:,2] > 0.01):
                names_list = self.model.names.decode('utf-8').split('\x00')[:40]
                res = np.zeros((6,1),dtype=np.float64)
                geom_name = lambda x: (names_list[self.model.geom_bodyid[x] + 1])
                geom_force = lambda x:mujoco.mj_contactForce(self.model,self.data,x,res) #type:ignore
                
                for contact in self.data.contact:
                    if contact.pos[2] > 0.01 and contact.geom1 != 0 and contact.geom2 != 0:
                        geom1_name = geom_name(contact.geom1)
                        geom2_name = geom_name(contact.geom2)
                        print(f"Warning!!! Collision between '{geom1_name,contact.geom1}' and '{geom2_name,contact.geom2}' at position {contact.pos}.")
                    
                # breakpoint()
            if RENDER:
                if self.viewer.is_alive:
                    self.viewer.render()
                else:
                    raise Exception("Mujoco Robot Exit")
            self.subtimer += 1


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
            
        # breakpoint()

    @property
    def Obs(self):
        
        self.UpdateObs()
        # return {k: torch2np(v) for k, v in self.obs_buf_dict.items()}
        return {'actor_obs': torch2np(self.obs_buf_dict['actor_obs']).reshape(1, -1)}

    ######################### Observations #########################
    def _get_obs_command_lin_vel(self):
        return np2torch(self.cmd[:2])
    
    def _get_obs_command_ang_vel(self):
        return np2torch(self.cmd[2:3])
    
    def _get_obs_actions(self,):
        return np2torch(self.act)
    
    def _get_obs_base_pos_z(self,):
        return np2torch(self.pos[2:3])
    
    def _get_obs_feet_contact_force(self,):
        raise NotImplementedError("Not implemented")
        return self.data.contact.force[:, :].view(self.num_envs, -1)
          
    
    def _get_obs_base_lin_vel(self,):
        return np2torch(self.vel)
    
    def _get_obs_base_ang_vel(self,):
        return np2torch(self.omega)
    
    def _get_obs_projected_gravity(self,):
        return np2torch(self.gvec)
    
    def _get_obs_dof_pos(self,):
        return np2torch(self.q - self.dof_init_pose)
    
    def _get_obs_dof_vel(self,):
        return np2torch(self.dq)
    
    
    def _get_obs_ref_motion_phase(self):
        logger.info(f"Phase: {self.ref_motion_phase}")
        return torch.tensor(self.ref_motion_phase).reshape(1,)
    
    def _get_obs_dif_local_rigid_body_pos(self):
        raise NotImplementedError("Not implemented")
        return self._obs_dif_local_rigid_body_pos
    
    def _get_obs_local_ref_rigid_body_pos(self):
        raise NotImplementedError("Not implemented")
        return self._obs_local_ref_rigid_body_pos
    
    def _get_obs_vr_3point_pos(self):
        raise NotImplementedError("Not implemented")
        return self._obs_vr_3point_pos
    
    def _get_obs_history(self,):
        assert "history" in self.cfg.obs.obs_auxiliary.keys()
        history_config = self.cfg.obs.obs_auxiliary['history']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1).reshape(-1)
    
    def _get_obs_short_history(self,):
        assert "short_history" in self.cfg.obs.obs_auxiliary.keys()
        history_config = self.cfg.obs.obs_auxiliary['short_history']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1).reshape(-1)
    
    def _get_obs_long_history(self,):
        assert "long_history" in self.cfg.obs.obs_auxiliary.keys()
        history_config = self.cfg.obs.obs_auxiliary['long_history']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1).reshape(-1)
    
    def _get_obs_history_actor(self,):
        assert "history_actor" in self.cfg.obs.obs_auxiliary.keys()
        history_config = self.cfg.obs.obs_auxiliary['history_actor']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1).reshape(-1)
    
    def _get_obs_history_critic(self,):
        assert "history_critic" in self.cfg.obs.obs_auxiliary.keys()
        history_config = self.cfg.obs.obs_auxiliary['history_critic']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1).reshape(-1)

from wjxtools import pdb_decorator
@hydra.main(config_path="config", config_name="base_eval")
@pdb_decorator
def main(override_config: OmegaConf):
    def setup_logging():
    
        # logging to hydra log file
        hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "eval.log")
        logger.remove()
        logger.add(hydra_log_path, level="DEBUG")

        # Get log level from LOGURU_LEVEL environment variable or use INFO as default
        console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
        logger.add(sys.stdout, level=console_log_level, colorize=True)

        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger().addHandler(HydraLoggerBridge())

        os.chdir(hydra.utils.get_original_cwd())
        
    def setup_simulator(config: OmegaConf):    
        # simulator_type = config.simulator['_target_'].split('.')[-1]
        simulator_type = config.simulator.config.name
        
        if simulator_type == 'real':
            from humanoidverse.deploy.real import RealRobot
            RobotCls = RealRobot
        elif simulator_type == 'mujoco':
            RobotCls = MujocoRobot
        else:
            raise NotImplementedError(f"Simulator type {simulator_type} not implemented")
        
        from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
        from humanoidverse.utils.helpers import pre_process_config
        import torch
        from humanoidverse.utils.inference_helpers import export_policy_as_jit, export_policy_as_onnx, export_policy_and_estimator_as_onnx

        return RobotCls, BaseAlgo, pre_process_config, torch, export_policy_as_jit, export_policy_as_onnx, export_policy_and_estimator_as_onnx
        
    def get_config(override_config: OmegaConf):
    
        if override_config.checkpoint is not None:
            has_config = True
            checkpoint = Path(override_config.checkpoint)
            config_path = checkpoint.parent / "config.yaml"
            if not config_path.exists():
                config_path = checkpoint.parent.parent / "config.yaml"
                if not config_path.exists():
                    has_config = False
                    logger.error(f"Could not find config path: {config_path}")

            if has_config:
                logger.info(f"Loading training config file from {config_path}")
                with open(config_path) as file:
                    train_config = OmegaConf.load(file)

                if train_config.eval_overrides is not None:
                    train_config = OmegaConf.merge(
                        train_config, train_config.eval_overrides
                    )

                config = OmegaConf.merge(train_config, override_config)
            else:
                config = override_config
        else:
            raise NotImplementedError("Not implemented")
            if override_config.eval_overrides is not None:
                config = override_config.copy()
                eval_overrides = OmegaConf.to_container(config.eval_overrides, resolve=True)
                for arg in sys.argv[1:]:
                    if not arg.startswith("+"):
                        key = arg.split("=")[0]
                        if key in eval_overrides:
                            del eval_overrides[key]
                config.eval_overrides = OmegaConf.create(eval_overrides)
                config = OmegaConf.merge(config, eval_overrides)
            else:
                config = override_config
                
        ckpt_num = config.checkpoint.split('/')[-1].split('_')[-1].split('.')[0]
        config.env.config.save_rendering_dir = str(checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}")
        config.env.config.ckpt_dir = str(checkpoint.parent) # commented out for now, might need it back to save motion
        OmegaConf.set_struct(config, False)
        
        return config, checkpoint
    
    def setup_logging2(config: OmegaConf):
        eval_log_dir = Path(config.eval_log_dir)
        eval_log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving eval logs to {eval_log_dir}")
        with open(eval_log_dir / "config.yaml", "w") as file:
            OmegaConf.save(config, file)

    def load_policy(config: OmegaConf, checkpoint: Path):
        assert checkpoint.suffix == '.onnx', f"File {checkpoint} is not a .onnx file."

        session = ort.InferenceSession(checkpoint, providers=['CPUExecutionProvider'])  # 使用CPU

        actor_dim = config.robot.algo_obs_dim_dict['actor_obs']
        action_dim = config.env.config.robot.actions_dim

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        example_input = np.random.randn(1, actor_dim).astype(np.float32)
        try_inferr = session.run([output_name], {input_name: example_input})
        assert try_inferr[0].shape == (1, action_dim), f"Action shape {try_inferr[0].shape} does not match expected shape (1, {action_dim})."
        
        def policy_fn(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
            # assert obs.shape == (1, actor_dim), f"Observation shape {obs.shape} does not match expected shape (1, {actor_dim})."
            result = session.run([output_name], obs_dict)
            # obs = obs_dict[input_name]
            # result = session.run([output_name], {input_name: obs_dict})
            # result = session.run([output_name], {input_name: obs})
            return result[0]
        return policy_fn
        
    
    setup_logging()
    
    config, checkpoint = get_config(override_config)
    
    setup_logging2(config)
    
    RobotCls, BaseAlgo, pre_process_config, torch, \
        export_policy_as_jit, export_policy_as_onnx, export_policy_and_estimator_as_onnx = \
                                setup_simulator(config)
    
    pre_process_config(config)
    # device = config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    
    policy_fn = load_policy(config, checkpoint)
    
    robot:URCIRobot = RobotCls(config)
    
    # breakpoint()
    while True:
        action = policy_fn(robot.Obs)[0]
        # action = np.zeros(robot.num_actions)
        
        robot.ApplyAction(action)
    
    
if __name__ == "__main__":
    main()
