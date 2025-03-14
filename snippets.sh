# Eval Agent:
HYDRA_FULL_ERROR=1 python humanoidverse/eval_agent.py +checkpoint=xxx/xxxx/model_78300.pt

# Data packing
# In cloud:
python playground/extract_logs.py --n_days 3 --max_id 400000 --min_id 1000 &&  curl -u xieweiji180:aef417d0b26566c15598c4237cc00e64 -T ./output.tar.gz "https://gz01-srdart.srdcloud.cn/generic/p24hqasyf0004/p24hqasyf0004-embodiedai-release-generic-local//wjx/asap.tar.gz"
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
+rewards=loco/g1 \
+obs=loco/wjx_hist_dr \
        \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=128 \
+device=cuda:0 \
project_name=DebugLocomotion \
experiment_name=Debug \
headless=True


# 25.03.14

python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=dr_wjx_s \
+rewards=loco/urgg1 \
+obs=loco/wjx_hist_dr \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
project_name=G1Locov2 \
experiment_name=URGG1_drs \
+device=cuda:0 \
headless=True

#----------------------------------------------------------

# 25.03.13

rewards.only_positive_rewards=True

python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=dr_wjx_nil \
+rewards=loco/g1 \
+obs=loco/wjx_hist_dr \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
+device=cuda:0 \
project_name=G1Loco \
experiment_name=v1Nil_OPal3_nostandstill \
rewards.reward_scales.standstill=0.0 \
rewards.reward_scales.alive=3.0 \
rewards.only_positive_rewards=True \
headless=True


python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=dr_wjx_ss \
+rewards=loco/g1 \
+obs=loco/wjx_hist_dr \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
+device=cuda:1 \
project_name=G1Loco \
experiment_name=v1drss_OPal3_nostandstill \
rewards.reward_scales.standstill=0.0 \
rewards.reward_scales.alive=3.0 \
rewards.only_positive_rewards=True \
headless=True


python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=dr_wjx_ss2 \
+rewards=loco/g1 \
+obs=loco/wjx_hist_dr \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
+device=cuda:2 \
project_name=G1Loco \
experiment_name=v1drss2_OPal3_nostandstill \
rewards.reward_scales.standstill=0.0 \
rewards.reward_scales.alive=3.0 \
rewards.only_positive_rewards=True \
headless=True

python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=dr_wjx_s \
+rewards=loco/g1 \
+obs=loco/wjx_hist_dr \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
+device=cuda:2 \
project_name=G1Loco \
experiment_name=v1drs_OPal5_nostandstill \
rewards.reward_scales.standstill=0.0 \
rewards.reward_scales.alive=5.0 \
rewards.only_positive_rewards=True \
headless=True

###

python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=dr_wjx_s \
+rewards=loco/g1 \
+obs=loco/wjx_hist_dr \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
+device=cuda:2 \
project_name=G1Loco \
experiment_name=v1drs_OPal1_nostandstill \
rewards.reward_scales.standstill=0.0 \
rewards.reward_scales.alive=1.0 \
rewards.only_positive_rewards=True \
headless=True


python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=dr_wjx_s \
+rewards=loco/g1 \
+obs=loco/wjx_hist_dr \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
+device=cuda:0 \
project_name=G1Loco \
experiment_name=v1drs_OPal5_nostandstill \
rewards.reward_scales.standstill=0.0 \
rewards.reward_scales.alive=5.0 \
rewards.only_positive_rewards=True \
headless=True

python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=dr_wjx_s \
+rewards=loco/g1 \
+obs=loco/wjx_hist_dr \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
+device=cuda:1 \
project_name=G1Loco \
experiment_name=v1drs_OPal10_nostandstill \
rewards.reward_scales.standstill=0.0 \
rewards.reward_scales.alive=10.0 \
rewards.only_positive_rewards=True \
headless=True

python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=dr_wjx_s \
+rewards=loco/g1 \
+obs=loco/wjx_hist_dr \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
+device=cuda:3 \
project_name=G1Loco \
experiment_name=v1drs_OPal3_nostandstill \
rewards.reward_scales.standstill=0.0 \
rewards.reward_scales.alive=3.0 \
rewards.only_positive_rewards=True \
headless=True


python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=dr_wjx_s \
+rewards=loco/g1 \
+obs=loco/wjx_hist_dr \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
+device=cuda:4 \
project_name=G1Loco \
experiment_name=v1drs_OPal2_nostandstill \
rewards.reward_scales.standstill=0.0 \
rewards.reward_scales.alive=2.0 \
rewards.only_positive_rewards=True \
headless=True


###

python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lkwr_ankiner \
+domain_rand=dr_wjx_s \
+rewards=loco/g1 \
+obs=loco/wjx_hist_dr \
robot.asset.urdf_file="g1/g1_23dof_lkwr_ankiner.urdf" \
num_envs=4096 \
project_name=G1Loco \
experiment_name=v1drs_OPal1_nostandstill \
rewards.reward_scales.standstill=0.0 \
rewards.reward_scales.alive=1.0 \
rewards.only_positive_rewards=True \
+device=cuda:5 \
headless=True


python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lkwr_ankiner \
+domain_rand=dr_wjx_s \
+rewards=loco/g1 \
+obs=loco/wjx_hist_dr \
robot.asset.urdf_file="g1/g1_23dof_lkwr_ankiner.urdf" \
num_envs=4096 \
+device=cuda:6 \
project_name=G1Loco \
experiment_name=v1drs_OPal2_nostandstill \
rewards.reward_scales.standstill=0.0 \
rewards.reward_scales.alive=2.0 \
rewards.only_positive_rewards=True \
headless=True


python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lkwr_ankiner \
+domain_rand=dr_wjx_s \
+rewards=loco/g1 \
+obs=loco/wjx_hist_dr \
robot.asset.urdf_file="g1/g1_23dof_lkwr_ankiner.urdf" \
num_envs=4096 \
+device=cuda:7 \
project_name=G1Loco \
experiment_name=v1drs_OPal3_nostandstill \
rewards.reward_scales.standstill=0.0 \
rewards.reward_scales.alive=3.0 \
rewards.only_positive_rewards=True \
headless=True


python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lkwr_ankiner \
+domain_rand=dr_wjx_s \
+rewards=loco/g1 \
+obs=loco/wjx_hist_dr \
robot.asset.urdf_file="g1/g1_23dof_lkwr_ankiner.urdf" \
num_envs=4096 \
+device=cuda:0 \
project_name=G1Loco \
experiment_name=v1drs_OPal5_nostandstill \
rewards.reward_scales.standstill=0.0 \
rewards.reward_scales.alive=5.0 \
rewards.only_positive_rewards=True \
headless=True


python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lkwr_ankiner \
+domain_rand=dr_wjx_s \
+rewards=loco/g1 \
+obs=loco/wjx_hist_dr \
robot.asset.urdf_file="g1/g1_23dof_lkwr_ankiner.urdf" \
num_envs=4096 \
+device=cuda:1 \
project_name=G1Loco \
experiment_name=v1drs_OPal10_nostandstill \
rewards.reward_scales.standstill=0.0 \
rewards.reward_scales.alive=10.0 \
rewards.only_positive_rewards=True \
headless=True

####



python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lkwr_ankiner \
+domain_rand=dr_wjx_s \
+rewards=loco/g1 \
+obs=loco/wjx_hist_dr \
robot.asset.urdf_file="g1/g1_23dof_lkwr_ankiner.urdf" \
num_envs=4096 \
+device=cuda:3 \
project_name=G1Loco \
experiment_name=v1drs_OPal3_sd5 \
rewards.reward_scales.standstill=5.0 \
rewards.reward_scales.alive=3.0 \
rewards.only_positive_rewards=True \
headless=True



python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lkwr_ankiner \
+domain_rand=dr_wjx_s \
+rewards=loco/g1 \
+obs=loco/wjx_hist_dr \
robot.asset.urdf_file="g1/g1_23dof_lkwr_ankiner.urdf" \
num_envs=4096 \
+device=cuda:3 \
project_name=G1Loco \
experiment_name=v1drs_OPal3_sd10 \
rewards.reward_scales.standstill=10.0 \
rewards.reward_scales.alive=3.0 \
rewards.only_positive_rewards=True \
headless=True




python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=dr_wjx_s \
+rewards=loco/g1 \
+obs=loco/wjx_hist_dr \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
+device=cuda:0 \
project_name=G1Loco \
experiment_name=v1drs_OPal3_sd10 \
rewards.reward_scales.standstill=10.0 \
rewards.reward_scales.alive=3.0 \
rewards.only_positive_rewards=True \
headless=True



python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=dr_wjx_s \
+rewards=loco/g1 \
+obs=loco/wjx_hist_dr \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
+device=cuda:1 \
project_name=G1Loco \
experiment_name=v1drs_OPal3_sd5 \
rewards.reward_scales.standstill=5.0 \
rewards.reward_scales.alive=3.0 \
rewards.only_positive_rewards=True \
headless=True

# ----------------------------------
# 25.03.12

python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=dr_wjx_nil \
+rewards=loco/g1 \
+obs=loco/wjx_hist_dr \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
+device=cuda:4 \
project_name=G1Loco \
experiment_name=v1Nil_nostandstill \
rewards.reward_scales.standstill=0.0 \
headless=True


python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=dr_wjx_ss \
+rewards=loco/g1 \
+obs=loco/wjx_hist_dr \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
+device=cuda:5 \
project_name=G1Loco \
experiment_name=v1drss_nostandstill \
rewards.reward_scales.standstill=0.0 \
headless=True


python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+terrain=terrain_locomotion_plane \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=dr_wjx_ss2 \
+rewards=loco/g1 \
+obs=loco/wjx_hist_dr \
robot.asset.urdf_file="g1/g1_23dof_lock_wrist.urdf" \
num_envs=4096 \
+device=cuda:6 \
project_name=G1Loco \
experiment_name=v1drss2_nostandstill \
rewards.reward_scales.standstill=0.0 \
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
