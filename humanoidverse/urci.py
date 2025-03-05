
import os
import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from humanoidverse.utils.logging import HydraLoggerBridge
import logging
from utils.config_utils import *  # noqa: E402, F403
# add argparse arguments

from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from loguru import logger


import onnxruntime as ort
import numpy as np


@hydra.main(config_path="config", config_name="base_eval")
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
        simulator_type = config.simulator['_target_'].split('.')[-1]
        if simulator_type == 'IsaacSim':
            from omni.isaac.lab.app import AppLauncher
            import argparse
            parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL.")
            AppLauncher.add_app_launcher_args(parser)
            
            args_cli, hydra_args = parser.parse_known_args()
            sys.argv = [sys.argv[0]] + hydra_args
            args_cli.num_envs = config.num_envs
            args_cli.seed = config.seed
            args_cli.env_spacing = config.env.config.env_spacing
            args_cli.output_dir = config.output_dir
            args_cli.headless = config.headless

            
            app_launcher = AppLauncher(args_cli)
            simulation_app = app_launcher.app
        if simulator_type == 'IsaacGym':
            import isaacgym
            
            
        from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
        from humanoidverse.utils.helpers import pre_process_config
        import torch
        from humanoidverse.utils.inference_helpers import export_policy_as_jit, export_policy_as_onnx, export_policy_and_estimator_as_onnx

        return BaseAlgo, pre_process_config, torch, export_policy_as_jit, export_policy_as_onnx, export_policy_and_estimator_as_onnx
        
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
        
        def policy_fn(obs: np.ndarray) -> np.ndarray:
            assert obs.shape == (1, actor_dim), f"Observation shape {obs.shape} does not match expected shape (1, {actor_dim})."
            result = session.run([output_name], {input_name: obs})
            return result[0]
        
        return policy_fn
        
    
    setup_logging()
    
    config, checkpoint = get_config(override_config)
    
    setup_logging2(config)
    
    BaseAlgo, pre_process_config, torch, \
        export_policy_as_jit, export_policy_as_onnx, export_policy_and_estimator_as_onnx = \
                                setup_simulator(config)
    
    pre_process_config(config)
    # device = config.device
    
    if config.get("device", None):
        device = config.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    # env = instantiate(config.env, device=device)
    
    policy_fn = load_policy(config, checkpoint)
    
    # algo.evaluate_policy()
    
    breakpoint()
    
if __name__ == "__main__":
    main()
