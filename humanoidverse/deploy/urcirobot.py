from typing import Dict, List, Tuple, Callable, Optional
import numpy as np
from omegaconf import OmegaConf
from loguru import logger
from humanoidverse.envs.env_utils.history_handler import HistoryHandler
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
from humanoidverse.utils.helpers import parse_observation
import time
import torch

np2torch = lambda x: torch.tensor(x, dtype=torch.float32)
torch2np = lambda x: x.cpu().numpy()


class URCIRobot:
    REAL: bool
    BYPASS_ACT: bool
    
    dt: float # big dt, not small dt
    clip_observations: float
    cfg: OmegaConf
    
    q: np.ndarray
    dq: np.ndarray
    # pos: np.ndarray
    # vel: np.ndarray
    quat: np.ndarray
    omega: np.ndarray
    gvec: np.ndarray
    rpy: np.ndarray
    
    act: np.ndarray
    
    _obs_cfg_obs: Optional[OmegaConf]=None
    _ref_pid = 0 # reference policy id
    
    # TODO: merging __init__ for both Mujoco and Real, sharing more code
    def __init__(self, cfg: OmegaConf):
        self.BYPASS_ACT = cfg.deploy.BYPASS_ACT
        
        self.cfg: OmegaConf = cfg
        self.device: str = "cpu"
        self.dt: float = cfg.deploy.ctrl_dt
        self.timer: int = 0
        
        
        self.num_actions = cfg.robot.actions_dim
        self.heading_cmd = cfg.deploy.heading_cmd   
        self.clip_action_limit: float = cfg.robot.control.action_clip_value
        self.clip_observations: float = cfg.env.config.normalization.clip_observations
        self.action_scale: float = cfg.robot.control.action_scale
        
        self._make_init_pose()
        self._make_buffer()
        if cfg.log_task_name == "motion_tracking":
            self.is_motion_tracking = True
            self._make_motionlib()
        else:
            self.is_motion_tracking = False
        
    
    # TODO: add a bottom for Reset and continue running
    def routing(self, cfg_policies: List[Tuple[OmegaConf, Callable]]):
        self._check_init()
        self.cmd[3]=self.rpy[2]
        cur_pid = -1

        while True:
            t1 = time.time()
            # TODO: auto change policy when a motion tracking is done
            
            if cur_pid != self._ref_pid:
                self._ref_pid %= len(cfg_policies)
                assert self._ref_pid >= 0 and self._ref_pid < len(cfg_policies), f"Invalid policy id: {self._ref_pid}"
                cur_pid = self._ref_pid
                self._obs_cfg_obs = cfg_policies[cur_pid][0]
                policy_fn = cfg_policies[cur_pid][1]
                
                # TODO: cleaning state, history about 'ref_motion_phase'
                self.timer=0 # TODO unify it.
                self.history_handler.history['ref_motion_phase']*=0
                
                self.UpdateObsWoHistory()
                
                breakpoint()
            
            action = policy_fn(self.Obs())[0]
            
            if self.BYPASS_ACT: action = np.zeros_like(action)
            
            self.ApplyAction(action)
            
            t2 = time.time()
            
            if self.REAL:
            # if True:
            #     print(f"t2-t1 = {t2-t1}")
                remain_dt = self.dt - (t2-t1)
                if remain_dt > 0:
                    time.sleep(remain_dt)
                else:
                    logger.warning(f"Warning! delay = {t2-t1} longer than policy_dt = {self.dt} , skip sleeping")
        
        ...
    
    def looping(self, policy_fn):
        self._check_init()
        self.cmd[3]=self.rpy[2]

        while True:
            t1 = time.time()
            
            action = policy_fn(self.Obs())[0]
            
            if self.BYPASS_ACT: action = np.zeros_like(action)
            
            self.ApplyAction(action)
            
            t2 = time.time()
            
            if self.REAL:
            # if True:
            #     print(f"t2-t1 = {t2-t1}")
                remain_dt = self.dt - (t2-t1)
                if remain_dt > 0:
                    time.sleep(remain_dt)
                else:
                    logger.warning(f"Warning! delay = {t2-t1} longer than policy_dt = {self.dt} , skip sleeping")
            # print(f"time: {t2-t1}")
            # time.sleep(self.dt)
        ...
    
    def Reset(self):
        raise NotImplementedError("Not implemented")
    
    def ApplyAction(self, action:np.ndarray):
        raise NotImplementedError("Not implemented")
    
    def Obs(self)->Dict[str, np.ndarray]:
        raise NotImplementedError("Not implemented")
    
    def GetState(self):
        raise NotImplementedError("Not implemented")
    
    # TODO: Decompose UpdateObs into two parts: 1) UpdateObsWoHistory, 2) UpdateObsWithHistory
    def UpdateObsWoHistory(self):
        self.GetState()
        
        obs_cfg_obs = self.cfg.obs if self._obs_cfg_obs is None else self._obs_cfg_obs
        
        self.obs_buf_dict_raw = {}
        
        noise_extra_scale = 1.
        for obs_key, obs_config in obs_cfg_obs.obs_dict.items():
            if not obs_key=='actor_obs': continue
            self.obs_buf_dict_raw[obs_key] = dict()

            parse_observation(self, obs_config, self.obs_buf_dict_raw[obs_key], obs_cfg_obs.obs_scales, obs_cfg_obs.noise_scales, noise_extra_scale)
        
        self.obs_buf_dict = dict()
        
        for obs_key, obs_config in obs_cfg_obs.obs_dict.items():
            if not obs_key=='actor_obs': continue
            obs_keys = sorted(obs_config)
            # print("obs_keys", obs_keys)            
            self.obs_buf_dict[obs_key] = torch.cat([self.obs_buf_dict_raw[obs_key][key] for key in obs_keys], dim=-1)
            
            
        clip_obs = self.clip_observations
        for obs_key, obs_val in self.obs_buf_dict.items():
            if not obs_key=='actor_obs': continue
            self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs)

    def UpdateObs(self):
        self.GetState()
        
        hist_cfg_obs = self.cfg.obs
        obs_cfg_obs = self.cfg.obs if self._obs_cfg_obs is None else self._obs_cfg_obs
        
        self.obs_buf_dict_raw = {}
        self.hist_obs_dict = {}
        
        noise_extra_scale = 1.
        for obs_key, obs_config in obs_cfg_obs.obs_dict.items():
            if not obs_key=='actor_obs': continue
            self.obs_buf_dict_raw[obs_key] = dict()

            parse_observation(self, obs_config, self.obs_buf_dict_raw[obs_key], obs_cfg_obs.obs_scales, obs_cfg_obs.noise_scales, noise_extra_scale)
        
        # Compute history observations
        history_obs_list = self.history_handler.history.keys()
        parse_observation(self, history_obs_list, self.hist_obs_dict, hist_cfg_obs.obs_scales, hist_cfg_obs.noise_scales, noise_extra_scale)
        
        
        self.obs_buf_dict = dict()
        
        for obs_key, obs_config in obs_cfg_obs.obs_dict.items():
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
    
    # TODO: better _check_init
    def _check_init(self):
        assert self.dt is not None, "dt is not set"
        assert self.dt>0 and self.dt < 0.1, "dt is not in the valid range"
        assert self.cfg is not None or not isinstance(self.cfg, OmegaConf), "cfg is not set"
        
        assert self.num_dofs is not None, "num_dofs is not set"
        assert self.num_dofs == 23, "In policy level, only 23 dofs are supported for now"
        assert self.kp is not None and type(self.kp) == np.ndarray and self.kp.shape == (self.num_dofs,), "kp is not set"
        assert self.kd is not None and type(self.kd) == np.ndarray and self.kd.shape == (self.num_dofs,), "kd is not set"
        
        assert (self.dof_init_pose is not None and type(self.dof_init_pose) == np.ndarray and 
                    self.dof_init_pose.shape == (self.num_dofs,)), "dof_init_pose is not set"
        
        assert self.tau_limit is not None and type(self.tau_limit) == np.ndarray and self.tau_limit.shape == (self.num_dofs,), "tau_limit is not set"
        
        assert self.BYPASS_ACT is not None, "BYPASS_ACT is not set"
        assert self.BYPASS_ACT in [True, False], "BYPASS_ACT is not a boolean, got {self.BYPASS_ACT}"
    
    def _make_init_pose(self):
        cfg_init_state = self.cfg.robot.init_state
        self.body_names = self.cfg.robot.body_names
        self.dof_names = self.cfg.robot.dof_names
        self.num_bodies = len(self.body_names)
        self.num_dofs = len(self.dof_names)
        assert self.num_dofs == 23, "Only 23 dofs are supported for now"
        
        
        dof_init_pose = cfg_init_state.default_joint_angles
        dof_effort_limit_list = self.cfg.robot.dof_effort_limit_list
        
        self.dof_init_pose = np.array([dof_init_pose[name] for name in self.dof_names])
        self.tau_limit = np.array(dof_effort_limit_list)
        
        
        self.kp = np.zeros(self.num_dofs)
        self.kd = np.zeros(self.num_dofs)
        
        
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            found = False
            for dof_name in self.cfg.robot.control.stiffness.keys():
                if dof_name in name:
                    self.kp[i] = self.cfg.robot.control.stiffness[dof_name]
                    self.kd[i] = self.cfg.robot.control.damping[dof_name]
                    found = True
                    logger.debug(f"PD gain of joint {name} were defined, setting them to {self.kp[i]} and {self.kd[i]}")
            if not found:
                raise ValueError(f"PD gain of joint {name} were not defined. Should be defined in the yaml file.")
        
    def _make_buffer(self):
        self.cmd: np.ndarray = np.array(self.cfg.deploy.defcmd)
        
        self.q = np.zeros(self.num_dofs)
        self.dq = np.zeros(self.num_dofs)
        self.quat = np.zeros(4)
        self.omega = np.zeros(3)
        self.gvec = np.zeros(3)
        self.rpy = np.zeros(3)
        
        self.act = np.zeros(self.num_dofs)
        
        
        self.history_handler = HistoryHandler(1, self.cfg.obs.obs_auxiliary, self.cfg.obs.obs_dims, self.device)
        ...
        
    def _make_motionlib(self):
        self.cfg.robot.motion.step_dt = self.dt
        self._motion_lib = MotionLibRobot(self.cfg.robot.motion, num_envs=1, device=self.device)
        self._motion_lib.load_motions(random_sample=False)
        
        self._motion_id = 0
        self.motion_len = self._motion_lib.get_motion_length(self._motion_id)
        # breakpoint()
        ...
        
        
    
    ######################### Observations #########################
    def _get_obs_command_lin_vel(self):
        return np2torch(self.cmd[:2])
    
    def _get_obs_command_ang_vel(self):
        return np2torch(self.cmd[2:3])
    
    def _get_obs_actions(self,):
        return np2torch(self.act)
    
    def _get_obs_base_pos_z(self,):
        # raise NotImplementedError("Not Implemented")
        return np2torch(self.pos[2:3])
    
    def _get_obs_feet_contact_force(self,):
        raise NotImplementedError("Not implemented")
        return self.data.contact.force[:, :].view(self.num_envs, -1)
          
    
    def _get_obs_base_lin_vel(self,):
        # raise NotImplementedError("Not Implemented")
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
        # logger.info(f"Phase: {self.ref_motion_phase}")
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
        obs_cfg_obs = self.cfg.obs if self._obs_cfg_obs is None else self._obs_cfg_obs
        assert "history" in obs_cfg_obs.obs_auxiliary.keys()
        history_config = obs_cfg_obs.obs_auxiliary['history']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1).reshape(-1)
    
    def _get_obs_short_history(self,):
        obs_cfg_obs = self.cfg.obs if self._obs_cfg_obs is None else self._obs_cfg_obs
        assert "short_history" in obs_cfg_obs.obs_auxiliary.keys()
        history_config = obs_cfg_obs.obs_auxiliary['short_history']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1).reshape(-1)
    
    def _get_obs_long_history(self,):
        obs_cfg_obs = self.cfg.obs if self._obs_cfg_obs is None else self._obs_cfg_obs
        assert "long_history" in obs_cfg_obs.obs_auxiliary.keys()
        history_config = obs_cfg_obs.obs_auxiliary['long_history']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1).reshape(-1)
    
    def _get_obs_history_actor(self,):
        obs_cfg_obs = self.cfg.obs if self._obs_cfg_obs is None else self._obs_cfg_obs
        assert "history_actor" in obs_cfg_obs.obs_auxiliary.keys()
        history_config = obs_cfg_obs.obs_auxiliary['history_actor']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1).reshape(-1)
    
    def _get_obs_history_critic(self,):
        obs_cfg_obs = self.cfg.obs if self._obs_cfg_obs is None else self._obs_cfg_obs
        assert "history_critic" in obs_cfg_obs.obs_auxiliary.keys()
        history_config = obs_cfg_obs.obs_auxiliary['history_critic']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1).reshape(-1)