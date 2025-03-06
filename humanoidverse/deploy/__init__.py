from typing import Dict
from .real import *
import numpy as np
from omegaconf import OmegaConf


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
    