from typing import Dict, List, Tuple, Callable, Optional
import numpy as np
from omegaconf import OmegaConf
from loguru import logger
from humanoidverse.envs.env_utils.history_handler import HistoryHandler
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
from humanoidverse.utils.helpers import parse_observation
import time
import torch

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
    
    def __init__(self, cfg: OmegaConf):
        self.BYPASS_ACT = cfg.deploy.BYPASS_ACT
    
    def routing(self, cfg_policies: List[Tuple[OmegaConf, Callable]]):
        policy_id = 0
        
        self._obs_cfg_obs = cfg_policies[policy_id][0]
        policy_fn = cfg_policies[policy_id][1]
        self.UpdateObsWoHistory()
        
        
        self.looping(policy_fn)
        
        raise NotImplementedError("Not implemented")
        ...
    
    def looping(self, policy_fn):
        self._check_init()
        
        while True:
            t1 = time.time()
            
            action = policy_fn(self.Obs())[0]
            
            if self.BYPASS_ACT: action = np.zeros_like(action)
            
            self.ApplyAction(action)
            
            t2 = time.time()
            
            if self.REAL:
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
    
    def UpdateObsWoHistory(self):
        
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
        