# Eval Agent:
HYDRA_FULL_ERROR=1 python humanoidverse/eval_agent.py +checkpoint=xxx/xxxx/model_78300.pt

# Data packing
# In cloud:
python playground/extract_logs.py --n_days 3 --max_id 2000 &&  curl -u xieweiji180:aef417d0b26566c15598c4237cc00e64 -T ./output.tar.gz "https://gz01-srdart.srdcloud.cn/generic/p24hqasyf0004/p24hqasyf0004-embodiedai-release-generic-local//wjx/asap.tar.gz"
# In local:
wget -O asap.tar.gz --user=xieweiji180 --password=aef417d0b26566c15598c4237cc00e64 "https://gz01-srdart.srdcloud.cn/generic/p24hqasyf0004/p24hqasyf0004-embodiedai-release-generic-local//wjx/asap.tar.gz" && python playground/unpack_logs.py --input asap.tar.gz && rm asap.tar.gz

# Locomotion




# Debug Run
python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
        \
+domain_rand=dr_wjx_nil \
+rewards=loco/reward_g1_locomotion \
+obs=loco/wjx_hist_dr \
        \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=128 \
+device=cuda:0 \
project_name=DebugLocomotion \
experiment_name=Debug \
headless=True



# No DR
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


# 25.03.10
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


# 25.03.11
# waist joint freeze, 
python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
        \
        \
+domain_rand=dr_wjx_s \
+rewards=loco/reward_g1_locomotion \
+obs=loco/leggedloco_obs_history_wjx \
        \
        \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
project_name=G1Loco \
experiment_name=v0Coll_wf_drs \
headless=True


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
experiment_name=v0Coll_wf_NoDR \
headless=True


python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
        \
        \
+domain_rand=dr_wjx_s \
+rewards=loco/reward_g1_locomotion \
+obs=loco/wjx_hist_dr \
        \
        \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
project_name=G1Loco \
experiment_name=v0Coll_wf_PrivDr \
headless=True


python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
        \
        \
+domain_rand=dr_wjx_nil \
+rewards=loco/reward_g1_locomotion \
+obs=loco/wjx_hist_dr \
        \
        \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
project_name=G1Loco \
experiment_name=v0Nil_wf_PrivDr \
headless=True


# 25.03.12

python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
        \
        \
+domain_rand=dr_wjx_ss \
+rewards=loco/reward_g1_locomotion \
+obs=loco/wjx_hist_dr \
        \
        \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
project_name=G1Loco \
experiment_name=v0drss_wf_PrivDr \
headless=True


python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
        \
        \
+domain_rand=dr_wjx_ss2 \
+rewards=loco/reward_g1_locomotion \
+obs=loco/wjx_hist_dr \
        \
        \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
+device=cuda:0 \
project_name=G1Loco \
experiment_name=v0drss2_wf_PrivDr \
headless=True


python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
        \
+domain_rand=dr_wjx_ss21 \
+rewards=loco/reward_g1_locomotion \
+obs=loco/wjx_hist_dr \
        \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
+device=cuda:0 \
project_name=G1Loco \
experiment_name=v0drss21_wf_PrivDr \
headless=True


python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
        \
+domain_rand=dr_wjx_ss22 \
+rewards=loco/reward_g1_locomotion \
+obs=loco/wjx_hist_dr \
        \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
+device=cuda:1 \
project_name=G1Loco \
experiment_name=v0drss22_wf_PrivDr \
+checkpoint=\
headless=True



python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
        \
+domain_rand=dr_wjx_ss23 \
+rewards=loco/reward_g1_locomotion \
+obs=loco/wjx_hist_dr \
        \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
+device=cuda:0 \
project_name=G1Loco \
experiment_name=v0drss23_wf_PrivDr \
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
