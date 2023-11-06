import json
import sys
import time
import numpy as np
import torch
import random
sys.path.append("../utils")
sys.path.insert(0, '../envs/air_hockey_challenge')
sys.path.insert(0, '../envs')
sys.path.insert(0, "../")
from trainer.particle_trainer import ParticleTrainer
from trainer.gaussian_trainer import GaussianTrainer
from trainer.gaussian_trainer_soft import GaussianTrainerSoft
from trainer.trainer import SACTrainer
from utils.variant_util import env_producer, get_policy_producer, get_q_producer
from utils.pythonplusplus import load_gzip_pickle


ts = '1584884279.5007188'
ts = '1589352957.4422379'
iter = 190
path = '../data/point/sac_/' + ts
ts = '1590677750.0582957'
path = '../data/point/mean_update_counts/p-oac_/' + ts
ts = '1595343877.9346888'
path = '../data-plot/air_hockey/wac/joints_included/s419248'
path = '../data-plot/air_hockey/wac/clipped_penalty_longer/s8672'
path = '../data-plot/air_hockey/wac/opt/shaped_reward/s905297'
path = '../data-plot/air_hockey/wac/has_hit/s107658'
path = "../data-plot/air_hockey/wac/opt/atacom/s8208"
path = "../data-plot/air_hockey/wac/opt/has_hit_2/s135729"
path = "../data-plot/air_hockey/wac/opt/atacom_new_r/s15984"
# path = "../data-plot/air_hockey/wac/opt/has_hit_stop/1"
path = "../data-plot/air_hockey/wac/opt/atacom_new_r_stop/s836241"
# path = "../data-plot/air_hockey/wac/opt/has_hit_stop/s60902"
path = "../data-plot/air_hockey/wac/opt/7dof_stop_high_level/s563654"
path = "../data-plot/air_hockey/wac/opt/7dof_stop_mod_has_hit/s184942"
# path = "../data-plot/air_hockey/wac/opt/7dof_stop_low_level_smaller_delta/s66979"
path = "../data-plot/air_hockey/wac/opt/7dof_stop_mod_has_hit_int_o_3/s200648"
path = "../data-plot/air_hockey/oac/7dof_io3/s742385"
path = "../data-plot/air_hockey/oac/7dof_0.5_decay/s273489"
path = "../data-plot/air_hockey/oac/7dof_ar_50_delta_03/s700023"
path = "../data-plot/air_hockey/tournament/oac_/whole_game_test/s108963"
path = "../data-plot/air_hockey/tournament/oac_/high_plus_delta/s142432"

path = "../data-plot/air_hockey/tournament/oac_/high_plus_delta_alphar4/s139898"

path = "../data-plot/air_hockey/tournament/oac_/atacom_maxpath800s400/s400"
# path = "../data-plot/air_hockey/tournament/oac_/action_persistence/s326652"
path = "../data-plot/air_hockey/tournament/oac_/atacom_plus_action_persistence/s66193"
path = "../data-plot/air_hockey/tournament/oac_/atacom_shaped_reward/s184414"
path = "../data-plot/air_hockey/tournament/oac_/high_delta_faults_ok/s804680"
path = "../data-plot/air_hockey/tournament/oac_/curriculum/high_delta_mixed_atacom_action_pers10/s985581"
path = "../data-plot/air_hockey/tournament/oac_/curriculum/high_delta_mixed_atacom/s125841"
path = "../data-plot/air_hockey/tournament/oac_/curriculum/large_pen/atacom_shaped_large_pen_done_for_vel/s174108"
# path = "../data-plot/air_hockey/tournament/oac_/curriculum/large_pen/high_delta_mixed_atacom_large_pen_done_vel/s38675"
# path = "../data-plot/air_hockey/tournament/oac_/curriculum/large_pen/high_delta_shaped_large_pen_done_for_vel/s560254"
path = "../data-plot/air_hockey/tournament/oac_/curriculum/large_pen/atacom_shaped_APers_large_bugs_fixed/s290656"

# path = "../data-plot/air_hockey/tournament/oac_/curriculum/large_pen/defend_first/atacom_shaped_APers_bug_fixed/s187147"

path = "../data-plot/air_hockey/tournament/oac_/curriculum/large_pen/defend_first/atacom_shaped_APers_bug_fixed/s187147"

path = "../data-plot/air_hockey/tournament/oac_/curriculum/large_pen/hit_first/atacom_main_bugs_fixed/s564021"

# path = "../data-plot/air_hockey/tournament/oac_/curriculum/large_pen/defend_first/atacom_large_R5k_not_joints/s655422"
#
# path = "../data-plot/air_hockey/tournament/oac_/curriculum/large_pen/best_defend/atacom_edame_bug_fixed/s187147"
#
# path = "../data-plot/air_hockey/tournament/oac_/curriculum/large_pen/best_defend/atacom_edame_bug_fixed/s11111111"
path = "../data-plot/air_hockey/tournament/oac_/curriculum/large_pen/best_hit/high_atacom_hit/s764878"

path = "../data-plot/air_hockey/tournament/oac_/curriculum/large_pen/best_defend/atacom_new_reward_from_scratch/s314382"

path = "../data-plot/air_hockey/7dof-defend/defend_curriculum/atacom_from_scratch/s749420"

path = "../data-plot/air_hockey/7dof-defend/defend_curriculum/defend_and_hit/s348769"

path = "../data-plot/air_hockey/7dof-defend/defend_curriculum/load_from_atacom_scratch_step1/s749420"

path = "../data-plot/air_hockey/7dof-defend/defend_curriculum/load_from_counter_direct_new_Reward/s348769"

path = "../data-plot/air_hockey/7dof-defend/defend_curriculum/load_from_counter_direct_new_Reward_bugs_fixed/s348769"
path = "../data-plot/air_hockey/7dof-defend/defend_curriculum/load_from_counter_direct_new_Reward_bugs_fixed_step1/s348769"

def load_agent(path, baseline_agent=False, env_label=None, random_agent=False):
    variant = json.load(open(path + '/variant.json', 'r'))
    seed = variant['seed']
    domain = variant['domain']
    env_args = {}
    if "hockey" in domain:
        env_args['env'] = variant['hockey_env']
        env_args['simple_reward'] = variant['simple_reward']
        env_args['shaped_reward'] = variant['shaped_reward']
        env_args['large_reward'] = variant['large_reward']
        env_args['large_penalty'] = variant['large_penalty']
        env_args['min_jerk'] = variant['min_jerk']
        env_args['max_jerk'] = variant['max_jerk']
        env_args['history'] = variant['history']
        env_args['use_atacom'] = variant['use_atacom']
        env_args['large_penalty'] = variant['large_penalty']
        env_args['alpha_r'] = variant['alpha_r']
        env_args['c_r'] = variant['c_r']
        env_args['high_level_action'] = variant['high_level_action']
        env_args['clipped_penalty'] = variant['clipped_penalty']
        env_args['include_joints'] = variant['include_joints']
        env_args['jerk_only'] = variant['jerk_only']
        env_args['delta_action'] = variant['delta_action']
        env_args['delta_ratio'] = variant['delta_ratio']
        # env_args['punish_jerk'] = variant['punish_jerk']
        env_args['stop_after_hit'] = False
        env_args['gamma'] = variant['trainer_kwargs']['discount']
        env_args['horizon'] = variant['algorithm_kwargs']['max_path_length'] + 1
        env_args["acceleration"] = variant["acceleration"]
        env_args['max_accel'] = variant['max_accel']
        env_args['interpolation_order'] = variant['interpolation_order']
        env_args['include_old_action'] = variant['include_old_action']
        env_args['use_aqp'] = variant['use_aqp']
        env_args['speed_decay'] = variant['speed_decay']
        env_args['whole_game_reward'] = variant['whole_game_reward']
        env_args['score_reward'] = variant['score_reward']
        env_args['fault_penalty'] = variant['fault_penalty']
        env_args['load_second_agent'] = variant['load_second_agent']
        env_args['dont_include_timer_in_states'] = variant['dont_include_timer_in_states']
        env_args['action_persistence'] = 0#variant['action_persistence']
        env_args['stop_when_puck_otherside'] = variant['stop_when_puck_otherside']

        env_args['curriculum_learning_step1'] = variant['curriculum_learning_step1']
        env_args['curriculum_learning_step2'] = variant['curriculum_learning_step2']
        env_args['curriculum_learning_step3'] = variant['curriculum_learning_step3']

        env_args['start_from_defend'] = variant['start_from_defend']

        env_args['original_env'] = True

        try:
            env_args['aqp_terminates'] = variant['aqp_terminates']

        except:
            env_args['aqp_terminates'] = False

    if baseline_agent:
        env_args['high_level_action'] = False
        env_args['include_joints'] = True
        env_args['delta_action'] = False
        env_args['use_atacom'] = False

    if env_label is not None:
        env_args['env'] = env_label

    env = env_producer(domain, seed, **env_args)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    # Get producer function for policy and value functions
    M = variant['layer_size']
    N = variant['num_layers']
    if not baseline_agent and not random_agent:
        alg = variant['alg']
        output_size = 1
        q_producer = get_q_producer(obs_dim, action_dim, hidden_sizes=[M] * N, output_size=output_size)
        policy_producer = get_policy_producer(
            obs_dim, action_dim, hidden_sizes=[M] * N)
        q_min = variant['r_min'] / (1 - variant['trainer_kwargs']['discount'])
        q_max = variant['r_max'] / (1 - variant['trainer_kwargs']['discount'])
        alg_to_trainer = {
            'sac': SACTrainer,
            'oac': SACTrainer,
            'p-oac': ParticleTrainer,
            'g-oac': GaussianTrainer,
            'gs-oac': GaussianTrainerSoft
        }
        trainer = alg_to_trainer[variant['alg']]

        kwargs = {}
        if alg in ['p-oac', 'g-oac', 'g-tsac', 'p-tsac', 'gs-oac']:
            kwargs = dict(
                delta=variant['delta'],
                q_min=q_min,
                q_max=q_max,
            )
        kwargs.update(dict(
            policy_producer=policy_producer,
            q_producer=q_producer,
            action_space=env.action_space,
        ))
        kwargs.update(variant['trainer_kwargs'])
        trainer = trainer(**kwargs)

        experiment = path + '/params.zip_pkl'
        exp = load_gzip_pickle(experiment)
        trainer.restore_from_snapshot(exp['trainer'])
    elif baseline_agent:
        from envs.air_hockey_challenge.examples.control.hitting_agent import build_agent
        trainer = build_agent(env.wrapped_env._env.env_info)
        trainer.reset()
        env = env.wrapped_env
    else:
        trainer = None
    return trainer, env


if __name__ == "__main__":
    #
    # seed = 10
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    baseline_agent = False
    target_policy = True
    render = True
    random_agent = False
    stop_after_hit = False

    # env_label = "tournament"
    env_label = "7dof-defend"
    n = 1000

    trainer, env = load_agent(path, baseline_agent, env_label=env_label, random_agent=random_agent)
    ob = env.reset()
    start_time = time.time()
    total_steps = 0
    rets = []


    # seed = 10
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)


    for i in range(n):
        s = env.reset()
        # if hasattr(trainer, 'reset'):
        #     trainer.reset()
        done = False
        ret = 0
        t = 0
        while not done and t < 400:
        # while not done and t <= env._env._mdp_info.horizon:

            if render:
                env.render()
            if stop_after_hit and env.has_hit:
                if env.delta_action or env.use_atacom or env.acceleration:
                    a *= 0.5 #** (max(np.linalg.norm(env.ee_vel), 0))
            elif random_agent:
                a = env.action_space.sample()
            elif hasattr(trainer, 'target_policy') and target_policy:
                a, agent_info = trainer.target_policy.get_action(s, deterministic=True)
            elif hasattr(trainer, 'policy') and not baseline_agent:
                a, agent_info = trainer.policy.get_action(s, deterministic=True)
            elif hasattr(trainer, 'draw_action'):
                # a = trainer.draw_action(expl_env._state)
                a = trainer.draw_action(env._env._state)
            else:
                a, agent_info = trainer.policy.get_action(s, deterministic=True)

            s, r, done, info = env.step(a)

            t += 1
            ret += r
        if render:
            env.render()
        total_steps += t
        # print("Success:", info["success"])
        print("Return:", ret)
        print("T:", t)
        rets.append(ret)
        if trainer is not None and hasattr(trainer, 'reset'):
            trainer.reset()
    end_time = time.time()
    print(" %d steps ---- %d episodes --- %s seconds ---" % (total_steps, n, end_time - start_time))
    print("Return: ", np.mean(rets), " +/- " + str(np.std(rets) / np.sqrt(n)))