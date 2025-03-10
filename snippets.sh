# Eval Agent:
HYDRA_FULL_ERROR=1 python humanoidverse/eval_agent.py +checkpoint=xxx/xxxx/model_78300.pt

# Locomotion

python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
        \
        \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_g1_locomotion \
+obs=loco/leggedloco_obs_history_wjx \
        \
        \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
project_name=G1Loco \
experiment_name=v0CollNoDR \
headless=True



python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
        \
        \
+domain_rand=dr_wjx \
+rewards=loco/reward_g1_locomotion \
+obs=loco/leggedloco_obs_history_wjx \
        \
        \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
project_name=G1Loco \
experiment_name=v0Coll_uja0.2 \
headless=True



# Motion Tracking


python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=motion_tracking \
+domain_rand=NO_domain_rand \
+rewards=motion_tracking/reward_motion_tracking_dm_2real \
+robot=g1/g1_23dof_lock_wrist \
+terrain=terrain_locomotion_plane \
+obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
num_envs=4096 \
project_name=MotionTracking \
experiment_name=DevGuitar \
robot.motion.motion_file="/home/bai/ASAP/SharedMotions/0-KIT_572_guitar_right11_poses.pkl" \
rewards.reward_penalty_curriculum=True \
rewards.reward_penalty_degree=0.00001 \
env.config.resample_motion_when_training=False \
env.config.termination.terminate_when_motion_far=True \
env.config.termination_curriculum.terminate_when_motion_far_curriculum=True \
env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.3 \
env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025 \
robot.asset.self_collisions=0
