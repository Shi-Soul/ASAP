# Unified Robot Control Interface for HumanoidVerse
# Weiji Xie @ 2025.03.04

REAL       :bool    = False


import os
import sys
from pathlib import Path

import torch
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, ListConfig, DictConfig
from humanoidverse.utils.logging import HydraLoggerBridge
from humanoidverse.utils.helpers import parse_observation
from humanoidverse.deploy import URCIRobot
from scipy.spatial.transform import Rotation as R
import logging
from utils.config_utils import *  # noqa: E402, F403
# add argparse arguments

from typing import Dict, Optional
from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from loguru import logger


import onnxruntime as ort
import numpy as np


from wjxtools import pdb_decorator
@hydra.main(config_path="config", config_name="base_eval")
@pdb_decorator
def main(override_config: OmegaConf):
    """
    Grammar:
    1. Single  Policy Method
        python humanoidverse/urci.py +simulator=mujoco +checkpoint=/path/to/checkpoint.onnx
        python humanoidverse/urci.py +simulator=real +checkpoint=/path/to/checkpoint.onnx
    2. Multiple Policy Method
        export CKPTS="[/path/to/ckp1.onnx,/path/to/ckp2.onnx,/path/to/ckp3.onnx]"
        python humanoidverse/urci.py +simulator=mujoco +checkpoint=$CKPTS
        
        export CKPTS="[/home/bai/ASAP/logs/G1Loco/20250310_114838-v0CollNoDR-locomotion-g1_23dof_lock_wrist/exported/model_6300.onnx,/home/bai/ASAP/logs/SharedCKPT/20250307_135730-original_collision_add_dr_collision_0.01-motion_tracking-g1_23dof_lock_wrist/exported/model_46300.onnx]"
        HYDRA_FULL_ERROR=1 python humanoidverse/urci.py +simulator=mujoco +checkpoint=$CKPTS
    
    """
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
            from humanoidverse.deploy.mujoco import MujocoRobot
            RobotCls = MujocoRobot
        else:
            raise NotImplementedError(f"Simulator type {simulator_type} not implemented")
        
        from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
        from humanoidverse.utils.helpers import pre_process_config
        import torch
        from humanoidverse.utils.inference_helpers import export_policy_as_jit, export_policy_as_onnx, export_policy_and_estimator_as_onnx

        return RobotCls, pre_process_config, torch
        
    def get_config(override_config: OmegaConf):
        def load_raw_ckpt_config(path: str, override_config: OmegaConf):
            checkpoint = Path(path)
            config_path = checkpoint.parent / "config.yaml"
            if not config_path.exists():
                config_path = checkpoint.parent.parent / "config.yaml"
                if not config_path.exists():
                    logger.error(f"Could not find config path: {config_path}")
                    raise NotImplementedError("Not implemented")

            logger.info(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                train_config = OmegaConf.load(file)

            if train_config.eval_overrides is not None:
                train_config = OmegaConf.merge(
                    train_config, train_config.eval_overrides
                )

            config = OmegaConf.merge(train_config, override_config)
                    
            ckpt_num = path.split('/')[-1].split('_')[-1].split('.')[0]
            config.checkpoint = checkpoint
            config.env.config.save_rendering_dir = str(checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}")
            config.env.config.ckpt_dir = str(checkpoint.parent) # commented out for now, might need it back to save motion
            OmegaConf.set_struct(config, False)
            return config
        
            ...
        
        def check_compatibility_config_robot(main_cfg: OmegaConf, sub_cfg: OmegaConf):
            """
            递归检查 robot 配置的兼容性（跳过 robot.motion.motion_file）
            """
            def _check_compatibility(main_node, sub_node, path="robot"):
                # 跳过 robot.motion.motion_file
                if path == "robot.motion.motion_file":
                    return

                # 基本类型直接比较
                if not (isinstance(main_node, (dict, list, DictConfig, ListConfig))):
                    if main_node != sub_node:
                        raise ValueError(f"Mismatch at {path}: {main_node} != {sub_node}")
                    return

                # 字典类型递归检查
                if isinstance(main_node, (dict, DictConfig)):
                    if isinstance(main_node, DictConfig):
                        main_node = OmegaConf.to_container(main_node, resolve=True)  # 转换为普通字典
                        sub_node = OmegaConf.to_container(sub_node, resolve=True) if sub_node is not None else {}
                    
                    for key in main_node:
                        if key not in sub_node:
                            raise ValueError(f"Missing key at {path}.{key}")
                        _check_compatibility(main_node[key], sub_node[key], f"{path}.{key}")

                # 列表类型逐元素检查
                elif isinstance(main_node, (list, ListConfig)):
                    if isinstance(main_node, ListConfig):
                        main_node = OmegaConf.to_container(main_node, resolve=True)
                        sub_node = OmegaConf.to_container(sub_node, resolve=True) if sub_node is not None else []
                    
                    if len(main_node) != len(sub_node):
                        raise ValueError(f"List length mismatch at {path}: {len(main_node)} vs {len(sub_node)}")
                    
                    for i, (m_item, s_item) in enumerate(zip(main_node, sub_node)):
                        _check_compatibility(m_item, s_item, f"{path}[{i}]")

            # 从 robot 根节点开始检查
            _check_compatibility(
                main_cfg.robot,
                sub_cfg.robot,
                path="robot"
            )
                    
        def check_compatibility_config_obs(main_cfg: DictConfig, sub_cfg: DictConfig):
            # Helper function to convert list of dicts to a single dict
            def list_of_dicts_to_dict(list_of_dicts):
                result = {}
                for d in list_of_dicts:
                    if len(d) != 1:
                        raise ValueError("Each item in obs_dims should be a dict with exactly one key-value pair")
                    key = list(d.keys())[0]
                    value = d[key]
                    result[key] = value
                return result

            # Convert list of dicts to a single dict for both main and sub configs
            main_obs_dims = list_of_dicts_to_dict(main_cfg.obs.obs_dims)
            sub_obs_dims = list_of_dicts_to_dict(sub_cfg.obs.obs_dims)

            # Check obs_auxiliary: template must fully contain sub-config's obs_auxiliary
            for key in sub_cfg.obs.obs_auxiliary:
                if key not in main_cfg.obs.obs_auxiliary:
                    raise ValueError(f"obs_auxiliary key {key} in sub-config is missing in main config")
                
                # Recursive check for nested dictionaries in obs_auxiliary
                def recursive_check(main_aux, sub_aux, path):
                    for sub_key in sub_aux:
                        if sub_key not in main_aux:
                            raise ValueError(f"obs_auxiliary key {path}{sub_key} in sub-config is missing in main config")
                        if isinstance(sub_aux[sub_key], dict):
                            recursive_check(main_aux[sub_key], sub_aux[sub_key], f"{path}{sub_key}.")
                        else:
                            if main_aux[sub_key] != sub_aux[sub_key]:
                                raise ValueError(f"Mismatch in obs_auxiliary for key {path}{sub_key}: {main_aux[sub_key]} != {sub_aux[sub_key]}")
                
                recursive_check(main_cfg.obs.obs_auxiliary[key], sub_cfg.obs.obs_auxiliary[key], f"{key}.")

            # Check obs_scales: same name, same scale
            for key, value in sub_cfg.obs.obs_scales.items():
                if key not in main_cfg.obs.obs_scales:
                    raise ValueError(f"obs_scales key {key} in sub-config is missing in main config")
                if main_cfg.obs.obs_scales[key] != value:
                    raise ValueError(f"Mismatch in obs_scales for key {key}: {main_cfg.obs.obs_scales[key]} != {value}")

            # Check noise_scales: template must contain all sub-config's noise_scales and set them to 0
            for key in sub_cfg.obs.noise_scales:
                if key not in main_cfg.obs.noise_scales:
                    raise ValueError(f"noise_scales key {key} in sub-config is missing in main config")
                main_cfg.obs.noise_scales[key] = 0  # Set noise scale to 0

            # Check obs_dims: same name, same dimension
            for key, value in sub_obs_dims.items():
                if key not in main_obs_dims:
                    raise ValueError(f"obs_dims key {key} in sub-config is missing in main config")
                if main_obs_dims[key] != value:
                    raise ValueError(f"Mismatch in obs_dims for key {key}: {main_obs_dims[key]} != {value}")
        
        single_policy_config = OmegaConf.load("humanoidverse/config/deploy/single.yaml")
        multiple_policy_config = OmegaConf.merge(
            single_policy_config,
            OmegaConf.load("humanoidverse/config/deploy/multiple.yaml"),
        )
        
        is_single_policy = isinstance(override_config.checkpoint, str) and override_config.checkpoint.endswith(".onnx") 
        is_multiple_policy = isinstance(override_config.checkpoint, ListConfig)
        
        logger.debug(f"{override_config.checkpoint=}")
        logger.debug(f"{is_single_policy=}")
        logger.debug(f"{is_multiple_policy=}")
        
        if is_single_policy:
            config = load_raw_ckpt_config(override_config.checkpoint, override_config)
            config = OmegaConf.merge(config, single_policy_config)
            return config.deploy.deploy_mode, (config, config.checkpoint)
        
        elif is_multiple_policy:
            # 1. load multiple configs
            # 2. check the compatibility of different configs
                # * check terms: 
                #  
                # cfg.robot: Must SAME (all terms except cfg.robot.motion.motion_file)
                # cfg.obs:
                    # Idea: one template, comparing with multiple sub-configs
                    # obs_auxiliary: template要完全包含sub的
                    # obs_scales: same name, same scale
                    # noise_scales: template 包含所有sub里出现的名字, 并且设置为0
                    # obs_dims: template 包含所有sub里出现的名字, same name same dim
                    # Others: don't care

            # 3. return one merged config with multiple sub-configs
            
            sub_configs = [load_raw_ckpt_config(path, override_config) for path in (override_config.checkpoint)]
            
            main_cfg = OmegaConf.merge(sub_configs[0], multiple_policy_config)
            
            for sub_cfg in sub_configs:
                # TODO: check more cfg, not only robot and obs
                check_compatibility_config_robot(main_cfg, sub_cfg)
                check_compatibility_config_obs(main_cfg, sub_cfg)
            
            return main_cfg.deploy.deploy_mode, {
                'main': main_cfg,
                'sub': sub_configs
            }
        else:
            raise ValueError(f"Invalid checkpoint type: {type(override_config.checkpoint)}")
        
    
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
    
    deploy_mode, ret_cfg_ckpt = get_config(override_config)
    config: OmegaConf
    checkpoint: Path
    if deploy_mode == 'single':
        config, checkpoint = ret_cfg_ckpt # type: ignore
    elif deploy_mode == 'multiple':
        main_config, sub_configs = ret_cfg_ckpt['main'], ret_cfg_ckpt['sub'] # type: ignore
        config = main_config
        checkpoint = main_config.checkpoint
    
    setup_logging2(config)
    
    RobotCls, pre_process_config, torch= setup_simulator(config)
    
    pre_process_config(config)
    # device = config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    
    assert RobotCls.REAL==REAL, f" {RobotCls.REAL=} is not equal to {REAL=}!!! Please manually set the REAL flag in the header."
    robot:URCIRobot = RobotCls(config)
    
    if deploy_mode == 'single':
        policy_fn = load_policy(config, checkpoint)
        robot.looping(policy_fn)
    elif deploy_mode == 'multiple':
        [pre_process_config(cfg) for cfg in sub_configs]
        cfg_policies = [(cfg.obs, load_policy(cfg, cfg.checkpoint)) for cfg in sub_configs]
        robot.routing(cfg_policies)
    else:
        raise ValueError(f"Invalid deploy mode: {deploy_mode}")
    
    
if __name__ == "__main__":
    main()
