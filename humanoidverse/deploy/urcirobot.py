from typing import Dict
import numpy as np
from omegaconf import OmegaConf
from loguru import logger
from humanoidverse.envs.env_utils.history_handler import HistoryHandler
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot


class URCIRobot:
    def __init__(self, cfg: OmegaConf):
        raise NotImplementedError("Not implemented")
    
    def Reset(self):
        raise NotImplementedError("Not implemented")
    
    def ApplyAction(self, action:np.ndarray):
        raise NotImplementedError("Not implemented")
    
    def GetState(self):
        raise NotImplementedError("Not implemented")
    
    def Obs(self)->Dict[str, np.ndarray]:
        raise NotImplementedError("Not implemented")
    
    def _make_init_pose(self):
        cfg_init_state = self.cfg.robot.init_state
        self.body_names = self.cfg.robot.body_names
        self.dof_names = self.cfg.robot.dof_names
        self.num_bodies = len(self.body_names)
        self.num_dofs = len(self.dof_names)
        
        
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
        self.act = np.zeros(self.num_actions)
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
        