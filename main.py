
import argparse
from numpy import e
import torch
import time
import utils.pytorch_util as ptu
from utils.env_utils import domain_to_epoch 
from utils.variant_util import build_variant
from utils.rng import set_global_pkg_rng_state
import sys
from scripts import run
import json

sys.path.insert(0,'envs/air_hockey_challenge')

def experiment(variant, prev_exp_state=None):
    env_args, alg_args, learn_args, log_args = build_variant(variant)


    if variant['load_from'] is not None:
        should_write_header = True if variant['load_from'] != variant['log_dir'] else False
    else:
        should_write_header = False

    print("slm")
    # algorithm = BatchRLAlgorithm(
    #     trainer=trainer,
    #     exploration_data_collector=expl_path_collector,
    #     remote_eval_data_collector=remote_eval_path_collector,
    #     replay_buffer=replay_buffer,
    #     optimistic_exp_hp=variant['optimistic_exp'],
    #     deterministic=variant['alg'] == 'p-oac',
    #     should_write_header=should_write_header,
    #     **variant['algorithm_kwargs']
    # )

    # algorithm.to(ptu.device)

    start_epoch = prev_exp_state['epoch'] + \
                  1 if prev_exp_state is not None else 0

    run.main(env_args, alg_args, learn_args, log_args, variant)
def get_cmd_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="algorithm", required=True)

    # for stable baseline3
    # parser.add_argument("--policy", type=str, default="MlpPolicy")
    # parser.add_argument("--verbose", type=int, default=1)

    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--domain', type=str, default='mountain')
    parser.add_argument('--dim', type=int, default=25)
    parser.add_argument('--pac', action="store_true")
    parser.add_argument('--ensemble', action="store_true")
    parser.add_argument('--n_policies', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alg', type=str, default='oac', choices=[
        'oac', 'p-oac', 'sac', 'g-oac', 'g-tsac', 'p-tsac', 'ddpg', 'oac-w', 'gs-oac'
    ])
    parser.add_argument('--hockey_env', type=str, default='3dof-hit', choices=[
        '3dof-hit', '3dof-defend', "7dof-hit", "7dof-defend", "7dof-prepare", "tournament"
    ])
    parser.add_argument('--no_gpu', default=False, action='store_true')
    parser.add_argument('--base_log_dir', type=str, default='./data')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--save_heatmap', action="store_true")
    parser.add_argument('--comp_MADE', action="store_true")
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--layer_size', type=int, default=256)
    parser.add_argument('--fake_policy', action="store_true")
    parser.add_argument('--random_policy', action="store_true")
    parser.add_argument('--expl_policy_std', type=float, default=0)
    parser.add_argument('--target_paths_qty', type=float, default=0)
    parser.add_argument('--dont_use_target_std', action="store_true")
    parser.add_argument('--n_estimators', type=int, default=2)
    parser.add_argument('--share_layers', action="store_true")
    parser.add_argument('--mean_update', action="store_true")
    parser.add_argument('--counts', action="store_true", help="count the samples in replay buffer")
    parser.add_argument('--std_inc_prob', type=float, default=0.)
    parser.add_argument('--prv_std_qty', type=float, default=0.)
    parser.add_argument('--prv_std_weight', type=float, default=1.)
    parser.add_argument('--std_inc_init', action="store_true")
    parser.add_argument('--log_dir', type=str, default='./data')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--max_path_length', type=int, default=1000) # SAC: 1000
    parser.add_argument('--replay_buffer_size', type=float, default=1e6)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256) # SAC: 256
    parser.add_argument('--r_min', type=float, default=0.)
    parser.add_argument('--r_max', type=float, default=1.)
    parser.add_argument('--r_mellow_max', type=float, default=1.)
    parser.add_argument('--mellow_max', action="store_true")
    parser.add_argument('--priority_sample', action="store_true")
    parser.add_argument('--global_opt', action="store_true")
    parser.add_argument('--save_sampled_data', default=False, action='store_true')
    parser.add_argument('--n_components', type=int, default=1)
    parser.add_argument('--snapshot_gap', type=int, default=10)
    parser.add_argument('--keep_first', type=int, default=-1)
    parser.add_argument('--save_fig', action='store_true')
    parser.add_argument('--simple_reward', action='store_true')
    parser.add_argument('--shaped_reward', action='store_true')
    parser.add_argument('--jerk_only', action='store_true')
    parser.add_argument('--high_level_action', action='store_true')
    parser.add_argument('--delta_action', action='store_true')
    parser.add_argument('--acceleration', action='store_true')
    parser.add_argument('--include_joints', action='store_true')
    parser.add_argument('--delta_ratio', type=float, default=0.1)
    parser.add_argument('--max_accel', type=float, default=0.2)
    parser.add_argument('--large_reward', type=float, default=1000)
    parser.add_argument('--large_penalty', type=float, default=100)
    parser.add_argument('--alpha_r', type=float, default=1.)
    parser.add_argument('--c_r', type=float, default=0.)
    parser.add_argument('--min_jerk', type=float, default=10000)
    parser.add_argument('--max_jerk', type=float, default=100000)
    parser.add_argument('--history', type=int, default=0)
    parser.add_argument('--use_atacom', action='store_true')
    parser.add_argument('--stop_after_hit', action='store_true')
    parser.add_argument('--punish_jerk', action='store_true')
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--interpolation_order', type=int, default=-1, choices=[1, 2, 3, 5, -1])
    parser.add_argument('--include_old_action', action='store_true')
    parser.add_argument('--use_aqp', action='store_true')
    parser.add_argument('--n_threads', type=int, default=25)
    parser.add_argument('--restore_only_policy', action='store_true')
    parser.add_argument('--no_buffer_restore', action='store_true')
    parser.add_argument('--speed_decay', type=float, default=0.5)
    parser.add_argument('--aqp_terminates', action='store_true')
    parser.add_argument('--whole_game_reward', action='store_true')
    parser.add_argument('--score_reward', type=float, default=10)
    parser.add_argument('--fault_penalty', type=float, default=-5)
    parser.add_argument('--load_second_agent', action='store_true')
    parser.add_argument('--dont_include_timer_in_states', action='store_true', default=False)
    parser.add_argument('--action_persistence', type=int, default=0)

    parser.add_argument('--stop_when_puck_otherside', action='store_true')

    parser.add_argument('--curriculum_learning_step1', action='store_true')
    parser.add_argument('--curriculum_learning_step2', action='store_true')
    parser.add_argument('--curriculum_learning_step3', action='store_true')

    parser.add_argument('--start_from_defend', action='store_true')
    parser.add_argument('--curriculum_transition', action='store_true')




    parser.add_argument(
        '--snapshot_mode',
        type=str,
        default='last_every_gap',
        choices=['last_every_gap', 'all', 'last', 'gap', 'gap_and_last', 'none']
    )
    parser.add_argument(
        '--difficulty', 
        type=str, 
        default='hard', 
        choices=[
            'empty', 'easy', 'medium', 'hard', 'harder', 'maze',
            'maze_easy', 'maze_med', 'maze_simple', 'double_L', 'double_I', 'para', 'maze_hard'
        ],
        help='only for point environment'
    )
    parser.add_argument('--policy_lr', type=float, default=3E-4)
    parser.add_argument('--qf_lr', type=float, default=3E-4)
    parser.add_argument('--std_lr', type=float, default=3E-4)
    parser.add_argument('--target_policy_lr', type=float, default=0)
    parser.add_argument('--sigma_noise', type=float, default=0.0)
    parser.add_argument('--deterministic_rs', action="store_true", help="make riverswim deterministic")
    parser.add_argument('--policy_grad_steps', type=int, default=1)
    parser.add_argument('--fixed_alpha', type=float, default=0)
    parser.add_argument('--stable_critic', action='store_true')

    parser.add_argument('--std_soft_update', action="store_true")
    parser.add_argument('--clip_state', action="store_true", help='only for point environment')
    parser.add_argument('--terminal', action="store_true", help='only for point environment')
    parser.add_argument('--max_state', type=float, default=500., help='only for point environment')
    parser.add_argument('--sparse_reward', action="store_true", help='only for point environment')
    parser.add_argument('--clipped_penalty', type=float, default=0.5, help='only for point environment')
    # optimistic_exp_hyper_param
    parser.add_argument('--beta_UB', type=float, default=0.0) # humanoid: 4.66
    parser.add_argument('--trainer_UB', action='store_true')
    parser.add_argument('--delta', type=float, default=0.0) # humanoid: 23.53
    parser.add_argument('--delta_oac', type=float, default=20.53)
    parser.add_argument('--deterministic_optimistic_exp', action='store_true')
    parser.add_argument('--no_resampling', action="store_true",
                        help="Samples are removed from replay buffer after being used once")

    # Training param
    parser.add_argument('--num_expl_steps_per_train_loop',
                        type=int, default=1000) # OAC default 1000
    parser.add_argument('--num_trains_per_train_loop', type=int, default=1000)
    parser.add_argument('--num_train_loops_per_epoch', type=int, default=1)
    parser.add_argument('--num_eval_steps_per_epoch', type=int, default=5000)
    parser.add_argument('--min_num_steps_before_training', type=int, default=1000)
    parser.add_argument('--clip_action', dest='clip_action', action='store_true')
    parser.add_argument('--no_clip_action', dest='clip_action', action='store_false')
    parser.set_defaults(clip_action=True)
    parser.add_argument('--policy_activation', type=str, default='ReLU')
    parser.add_argument('--policy_output', type=str, default='TanhGaussian')
    parser.add_argument('--policy_weight_decay', type=float, default=0)
    parser.add_argument('--reward_scale', type=float, default=1.0)
    parser.add_argument('--entropy_tuning', dest='entropy_tuning', action='store_true')
    parser.add_argument('--no_entropy_tuning', dest='entropy_tuning', action='store_false')
    parser.set_defaults(entropy_tuning=True)
    parser.add_argument('--load_from', type=str, default='')
    parser.add_argument('--load_itr', type=str, default='params.zip_pkl')
    parser.add_argument('--train_bias', dest='train_bias', action='store_true')
    parser.add_argument('--no_train_bias', dest='train_bias', action='store_false')
    parser.add_argument('--should_use',  action='store_true')
    parser.add_argument('--stochastic',  action='store_true')
    parser.set_defaults(train_bias=True)
    parser.add_argument('--soft_target_tau', type=float, default=5E-3)
    parser.add_argument('--ddpg', action='store_true', help='use a ddpg version of the algorithms')
    parser.add_argument('--ddpg_noisy', action='store_true', help='use noisy exploration policy')
    parser.add_argument('--std', type=float, default=0.1, help='use noisy exploration policy for ddpg')
    parser.add_argument('--use_target_policy', action='store_true', help='use a target policy in ddpg')
    parser.add_argument('--rescale_targets_around_mean', action='store_true', help='use a target policy in ddpg')
    parser.add_argument('--restore_path_collectors', action='store_true')
    parser.add_argument('--restore_only_buffer', action='store_true')
    parser.add_argument('--clip_vel', action='store_true')


    # learning args

    parser.add_argument("--total_timesteps", type=int, required=True)
    # parser.add_argument("--tb_log_name", type=str)

    # ppo arguments
    parser_ppo = subparsers.add_parser("ppo")
    parser_ppo.add_argument("--policy", type=str, default="MlpPolicy")
    parser_ppo.add_argument("--lr", type=float, default=3e-4)
    parser_ppo.add_argument("--steps_per_update", type=int, default=32)
    parser_ppo.add_argument("--batch_size", type=int, default=64)
    parser_ppo.add_argument("--n_epochs", type=int, default=10)
    parser_ppo.add_argument("--gamma", type=float, default=0.999)
    parser_ppo.add_argument("--gae_lambda", type=float, default=0.95)
    parser_ppo.add_argument("--clip_range", type=float, default=0.2)
    parser_ppo.add_argument("--clip_range_vf", type=float)
    parser_ppo.add_argument("--normalize_advantage", action="store_true", default=True)
    parser_ppo.add_argument("--ent_coef", type=float, default=0.0)
    parser_ppo.add_argument("--vf_coef", type=float, default=0.5)
    parser_ppo.add_argument("--max_grad_norm", type=float, default=0.5)
    parser_ppo.add_argument("--use_sde", action="store_true", default=False)
    parser_ppo.add_argument("--sde_sample_freq", type=int, default=-1)
    parser_ppo.add_argument("--target_kl", type=float)
    parser_ppo.add_argument("--stats_window_size", type=int, default=100)
    parser_ppo.add_argument("--tensorboard_log", type=str)
    parser_ppo.add_argument("--verbose", type=int, default=1)
    parser_ppo.add_argument("--seed", type=int)
    parser_ppo.set_defaults(alg='ppo')

    #sac arguments
    parser_sac = subparsers.add_parser("sac")
    parser_sac.add_argument("--policy", type=str, default="MlpPolicy")
    parser_sac.add_argument("--learning_rate", type=float, default=3e-4)
    parser_sac.add_argument("--buffer_size", type=int, default=1000000)
    parser_sac.add_argument("--learning_starts", type=int, default=1000)
    parser_sac.add_argument("--batch_size", type=int, default=256)
    parser_sac.add_argument("--tau", type=float, default=0.005)
    parser_sac.add_argument("--gamma", type=float, default=0.99)
    parser_sac.add_argument("--train_freq", type=int, default=1)
    parser_sac.add_argument("--gradient_steps", type=int, default=1)
    parser_sac.add_argument("--action_noise", type=str, default=None)
    parser_sac.add_argument("--replay_buffer_class", type=str, default=None)
    parser_sac.add_argument("--replay_buffer_kwargs", type=json.loads, default=None)
    parser_sac.add_argument("--optimize_memory_usage", action="store_true")
    parser_sac.add_argument("--ent_coef", type=str, default='auto')
    parser_sac.add_argument("--target_update_interval", type=int, default=1)
    parser_sac.add_argument("--target_entropy", type=str, default=None)
    parser_sac.add_argument("--use_sde", action="store_true")
    parser_sac.add_argument("--sde_sample_freq", type=int, default=-1)
    parser_sac.add_argument("--use_sde_at_warmup", action="store_true")
    parser_sac.add_argument("--stats_window_size", type=int, default=100)
    parser_sac.add_argument("--tensorboard_log", type=str, default=None)
    parser_sac.add_argument("--policy_kwargs", type=json.loads, default=None)
    parser_sac.add_argument("--verbose", type=int, default=1)
    parser_sac.add_argument("--seed", type=int, default=None)
    parser_sac.add_argument("--device", type=str, default="auto")
    parser_sac.add_argument("--_init_setup_model", action="store_true")
    parser_sac.set_defaults(alg='sac')

    #dqn arguments
    parser_dqn = subparsers.add_parser("dqn")
    parser_dqn.add_argument("--policy", type=str, default="MlpPolicy")
    parser_dqn.add_argument("--learning_rate", type=float, default=0.001)
    parser_dqn.add_argument("--buffer_size", type=int, default=10000)
    parser_dqn.add_argument("--learning_starts", type=int, default=1000)
    parser_dqn.add_argument("--batch_size", type=int, default=32)
    parser_dqn.add_argument("--tau", type=float, default=1.0)
    parser_dqn.add_argument("--gamma", type=float, default=0.99)
    parser_dqn.add_argument("--train_freq", type=int, default=1)
    parser_dqn.add_argument("--gradient_steps", type=int, default=1)
    parser_dqn.add_argument("--replay_buffer_class", type=str, default=None)
    parser_dqn.add_argument("--replay_buffer_kwargs", type=json.loads, default=None)
    parser_dqn.add_argument("--optimize_memory_usage", action="store_true")
    parser_dqn.add_argument("--target_update_interval", type=int, default=100)
    parser_dqn.add_argument("--exploration_fraction", type=float, default=0.1)
    parser_dqn.add_argument("--exploration_initial_eps", type=float, default=1.0)
    parser_dqn.add_argument("--exploration_final_eps", type=float, default=0.02)
    parser_dqn.add_argument("--max_grad_norm", type=float, default=10.0)
    parser_dqn.add_argument("--stats_window_size", type=int, default=100)
    parser_dqn.add_argument("--tensorboard_log", type=str, default=None)
    parser_dqn.add_argument("--policy_kwargs", type=json.loads, default=None)
    parser_dqn.add_argument("--verbose", type=int, default=1)
    parser_dqn.add_argument("--seed", type=int, default=None)
    parser_dqn.add_argument("--device", type=str, default="auto")
    parser_dqn.add_argument("--_init_setup_model", action="store_true", default=True)
    parser_dqn.set_defaults(alg='dqn')


    args = parser.parse_args()

    return args


def get_log_dir(args, should_include_base_log_dir=True, should_include_seed=True, should_include_domain=True):
    start_time = time.time()
    if args.load_dir != '':
        log_dir = args.load_dir
    else:
        if args.n_policies > 1:
            el = str(args.n_policies)
        elif args.n_components > 1:
            el = str(args.n_components)
        else:
            el = ''
        log_dir = args.log_dir + '/' + args.domain + '/' +  \
                  (args.difficulty + '/' if args.domain == 'point' else '') + \
                  (args.hockey_env + '/' if args.domain == 'air_hockey' else '') + \
                  ('terminal' + '/' if args.terminal and  args.domain == 'point' else '') + \
                  (str(args.dim) + '/' if args.domain == 'riverswim' else '') + \
                  ('global/' if args.global_opt else '') + \
                  ('ddpg/' if args.ddpg else '') + \
                  ('mean_update_' if args.mean_update else '') + \
                  ('_priority_' if args.priority_sample else '') + \
                  ('counts/' if args.counts else '') + \
                  ('/' if args.mean_update and not args.counts else '') + \
                   args.alg + ('_std' if args.std_soft_update else '') + '_' + el + '/' +\
                   args.suffix + '/' # + str(int(start_time))
        if args.log_dir == './data/debug':
            log_dir = log_dir + str(int(start_time))


    return log_dir


if __name__ == "__main__":
    # Parameters for the experiment are either listed in variant below
    # or can be set through cmdline args and will be added or overrided
    # the corresponding attributein variant

    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1e6), # default 1e6
        algorithm_kwargs=dict(
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=None,
            num_expl_steps_per_train_loop=None,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=32,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            update_mode="step",
            action_noise=None,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
        ),
        optimistic_exp={}
    )

    args = get_cmd_args()
    print(args.load_from)

    variant['log_dir'] = get_log_dir(args)
    if args.load_from != '':
        variant['load_from'] = args.load_from + '/'
        if args.load_itr != 'params.zip_pkl':
            variant['params'] = "itr_" + args.load_itr + ".zip_pkl"
        else:
            variant['params'] = args.load_itr
    else:
        variant['load_from'] = None
        variant['params'] = args.load_itr

    variant['seed'] = args.seed
    variant['policy'] = args.policy
    variant['verbose'] = args.verbose
    variant['total_timesteps'] = args.total_timesteps
    #variant['git_hash'] = git.Repo().head.object.hexsha
    variant['domain'] = args.domain
    variant['num_layers'] = args.num_layers
    variant['layer_size'] = args.layer_size
    variant['share_layers'] = args.share_layers
    variant['n_estimators'] = args.n_estimators if args.alg in ['p-oac', 'p-tsac'] else 2
    variant['replay_buffer_size'] = int(args.replay_buffer_size)
    variant['algorithm_kwargs']['num_epochs'] = domain_to_epoch(args.domain) if args.epochs <= 0 else args.epochs
    variant['algorithm_kwargs']['num_trains_per_train_loop'] = args.num_trains_per_train_loop
    variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = args.num_expl_steps_per_train_loop
    variant['algorithm_kwargs']['max_path_length'] = args.max_path_length
    variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = args.num_eval_steps_per_epoch
    variant['algorithm_kwargs']['min_num_steps_before_training'] = args.min_num_steps_before_training
    variant['algorithm_kwargs']['batch_size'] = args.batch_size
    variant['algorithm_kwargs']['save_sampled_data'] = args.save_sampled_data
    variant['algorithm_kwargs']['num_train_loops_per_epoch'] = args.num_train_loops_per_epoch
    variant['algorithm_kwargs']['trainer_UB'] = args.trainer_UB
    variant['algorithm_kwargs']['fake_policy'] = args.fake_policy
    variant['algorithm_kwargs']['random_policy'] = args.random_policy
    variant['algorithm_kwargs']['domain'] = args.domain
    variant['algorithm_kwargs']['target_paths_qty'] = args.target_paths_qty
    # variant['algorithm_kwargs']['log_dir'] = args.log_dir

    variant['delta'] = args.delta
    variant['std'] = args.std
    variant['simple_reward'] = args.simple_reward
    variant['shaped_reward'] = args.shaped_reward
    variant['clipped_penalty'] = args.clipped_penalty
    variant['jerk_only'] = args.jerk_only
    variant['high_level_action'] = args.high_level_action
    variant['delta_action'] = args.delta_action
    variant['max_accel'] = args.max_accel
    variant['acceleration'] = args.acceleration
    variant['delta_ratio'] = args.delta_ratio
    variant['include_joints'] = args.include_joints
    variant['large_reward'] = args.large_reward
    variant['large_penalty'] = args.large_penalty
    variant['min_jerk'] = args.min_jerk
    variant['max_jerk'] = args.max_jerk
    variant['alpha_r'] = args.alpha_r
    variant['c_r'] = args.c_r
    variant['history'] = args.history
    variant['use_atacom'] = args.use_atacom
    variant['stop_after_hit'] = args.stop_after_hit
    variant['punish_jerk'] = args.punish_jerk
    variant['parallel'] = args.parallel
    variant['interpolation_order'] = args.interpolation_order
    variant['restore_path_collectors'] = args.restore_path_collectors
    variant['restore_only_buffer'] = args.restore_only_buffer
    variant['include_old_action'] = args.include_old_action
    variant['use_aqp'] = args.use_aqp
    variant['n_threads'] = args.n_threads
    variant['restore_only_policy'] = args.restore_only_policy
    variant['no_buffer_restore'] = args.no_buffer_restore
    variant['speed_decay'] = args.speed_decay
    variant['clip_vel'] = args.clip_vel
    variant['aqp_terminates'] = args.aqp_terminates
    variant['whole_game_reward'] = args.whole_game_reward
    variant['score_reward'] = args.score_reward
    variant['fault_penalty'] = args.fault_penalty
    variant['load_second_agent'] = args.load_second_agent
    variant['dont_include_timer_in_states'] = args.dont_include_timer_in_states
    variant['action_persistence'] = args.action_persistence
    variant['stop_when_puck_otherside'] = args.stop_when_puck_otherside

    if args.curriculum_learning_step3:
        variant['curriculum_learning_step1'] = True
        variant['curriculum_learning_step2'] = True
        variant['curriculum_learning_step3'] = True
    elif args.curriculum_learning_step2:
        variant['curriculum_learning_step1'] = True
        variant['curriculum_learning_step2'] = True
        variant['curriculum_learning_step3'] = False
    elif args.curriculum_learning_step1:
        variant['curriculum_learning_step1'] = True
        variant['curriculum_learning_step2'] = False
        variant['curriculum_learning_step3'] = False
    else:
        variant['curriculum_learning_step1'] = False
        variant['curriculum_learning_step2'] = False
        variant['curriculum_learning_step3'] = False

    variant['start_from_defend'] = args.start_from_defend
    variant['curriculum_transition'] = args.curriculum_transition


    variant['optimistic_exp']['should_use'] = args.beta_UB > 0 or args.delta > 0 and not args.alg in [
        'p-oac', 'sac','g-oac', 'g-tsac','p-tsac', 'gs-oac', 'ddpg'
    ]
    if not variant['optimistic_exp']['should_use']:
        variant['optimistic_exp']['should_use'] = args.should_use
    variant['optimistic_exp']['beta_UB'] = args.beta_UB if args.alg in ['oac', 'oac-w'] else 0
    variant['optimistic_exp']['delta'] = args.delta if args.alg in ['p-oac', 'oac', 'g-oac', 'oac-w', 'gs-oac'] else 0
    variant['optimistic_exp']['share_layers'] = False
    if args.alg in ['p-oac']:
        variant['optimistic_exp']['share_layers'] = args.share_layers
    if args.should_use and args.alg in ['p-oac']:
        variant['optimistic_exp']['delta'] = args.delta_oac
    variant['optimistic_exp']['deterministic'] = args.deterministic_optimistic_exp
    if args.alg not in ['ddpg']:
        variant['trainer_kwargs']['use_automatic_entropy_tuning'] = args.entropy_tuning
    variant['trainer_kwargs']['discount'] = args.gamma
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['qf_lr'] = args.qf_lr
    if not args.target_policy_lr == 0:
        variant['trainer_kwargs']['target_policy_lr'] = args.target_policy_lr
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['ensemble'] = args.ensemble
    variant['n_policies'] = args.n_policies if args.ensemble else 1
    variant['n_components'] = args.n_components
    variant['priority_sample'] = False
    variant['clip_action'] = args.clip_action
    variant['policy_activation'] = args.policy_activation
    variant['policy_output'] = args.policy_output
    variant['stochastic'] = args.stochastic
    if args.domain == 'lqg':
        variant['clip_action'] = True
    if not args.fixed_alpha == 0:
        variant['trainer_kwargs']['fixed_alpha'] = args.fixed_alpha
    # if args.alg in ['g-oac']:
    #     variant['trainer_kwargs']['expl_policy_lr'] = args.expl_policy_lr
    if args.alg in ['p-oac', 'g-oac', 'g-tsac', 'p-tsac', 'oac-w', 'gs-oac']:
        variant['trainer_kwargs']['std_soft_update'] = args.std_soft_update
        variant['trainer_kwargs']['counts'] = args.counts
        variant['trainer_kwargs']['prv_std_qty'] = args.prv_std_qty
        variant['trainer_kwargs']['prv_std_weight'] = args.prv_std_weight
        variant['trainer_kwargs']['dont_use_target_std'] = args.dont_use_target_std
    if args.alg in ['gs-oac', 'oac-w']:
        variant['trainer_kwargs']['train_bias'] = args.train_bias # duplicate
    if args.alg in ['gs-oac']:
        variant['trainer_kwargs']['mean_update'] = args.mean_update # duplicate
        variant['trainer_kwargs']['stable_critic'] = args.stable_critic # 
    if args.alg in ['p-oac', 'g-oac', 'g-tsac', 'p-tsac']:
        variant['trainer_kwargs']['share_layers'] = args.share_layers
        variant['trainer_kwargs']['mean_update'] = args.mean_update 
        variant['trainer_kwargs']['std_inc_prob'] = args.std_inc_prob
        variant['trainer_kwargs']['std_inc_init'] = args.std_inc_init
        variant['trainer_kwargs']['fake_policy'] = args.fake_policy
        variant['priority_sample'] = args.priority_sample
        variant['trainer_kwargs']['global_opt'] = args.global_opt
        variant['trainer_kwargs']['policy_grad_steps'] = args.policy_grad_steps
        if args.alg in ['p-oac', 'g-oac']:
            variant['trainer_kwargs']['r_mellow_max'] = args.r_mellow_max
            variant['trainer_kwargs']['mellow_max'] = args.mellow_max
            variant['algorithm_kwargs']['global_opt'] = args.global_opt
            variant['algorithm_kwargs']['save_fig'] = args.save_fig
            variant['algorithm_kwargs']['expl_policy_std'] = args.expl_policy_std
            variant['trainer_kwargs']['train_bias'] = args.train_bias
            variant['trainer_kwargs']['rescale_targets_around_mean'] = args.rescale_targets_around_mean
            variant['trainer_kwargs']['use_target_policy'] = args.use_target_policy
        if args.alg in ['g-oac', 'oac-w', 'gs-oac']:
            variant['trainer_kwargs']['std_lr'] = args.std_lr
    variant['algorithm_kwargs']['save_heatmap'] = args.save_heatmap
    variant['algorithm_kwargs']['comp_MADE'] = args.comp_MADE
    variant['trainer_kwargs']['policy_weight_decay'] = args.policy_weight_decay
    variant['trainer_kwargs']['reward_scale'] = args.reward_scale
    variant['hockey_env'] = args.hockey_env
    variant['alg'] = args.alg
    variant['dim'] = args.dim
    variant['difficulty'] = args.difficulty
    variant['max_state'] = args.max_state
    variant['clip_state'] = args.clip_state
    variant['terminal'] = args.terminal
    variant['sparse_reward'] = args.sparse_reward
    variant['pac'] = args.pac
    variant['no_resampling'] = args.no_resampling
    variant['r_min'] = args.r_min
    variant['r_max'] = args.r_max
    variant['sigma_noise'] = args.sigma_noise
    variant['deterministic_rs'] = args.deterministic_rs # added


    variant['trainer_kwargs']['soft_target_tau'] = args.soft_target_tau
    variant['algorithm_kwargs']['ddpg_noisy'] = args.ddpg_noisy
    if args.alg in ['p-oac', 'g-oac', 'g-tsac', 'p-tsac']: # default False
        N_expl = variant['algorithm_kwargs']['num_expl_steps_per_train_loop']
        N_train = variant['algorithm_kwargs']['num_trains_per_train_loop']
        B = variant['algorithm_kwargs']['batch_size']
        N_updates = (N_train * B) / N_expl
        std_soft_update_prob = 2 / (N_updates * (N_updates + 1))
        variant['trainer_kwargs']['std_soft_update_prob'] = std_soft_update_prob
    if args.ddpg or args.alg == 'ddpg':
        variant['algorithm_kwargs']['num_trains_per_train_loop'] = 1
        variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = 4
        variant['algorithm_kwargs']['num_train_loops_per_epoch'] = args.num_expl_steps_per_train_loop // 4
        variant['trainer_kwargs']['use_target_policy'] = args.use_target_policy
        variant['algorithm_kwargs']['ddpg'] = args.ddpg
        if args.alg == 'ddpg':
            variant['algorithm_kwargs']['ddpg_noisy'] = True
        else:
            variant['algorithm_kwargs']['ddpg_noisy'] = args.ddpg_noisy

    #print("Prob %s" % variant['trainer_kwargs']['std_soft_update_prob'])

    if args.no_resampling:
        variant['algorithm_kwargs']['num_trains_per_train_loop'] = 500
        variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = 500 * args.batch_size
        variant['algorithm_kwargs']['min_num_steps_before_training'] = 1000 
        # 4 * 500 * args.batch_size # SAC: 10000
        variant['algorithm_kwargs']['batch_size'] = args.batch_size
        variant['replay_buffer_size'] = 5 * 500 * args.batch_size
    if torch.cuda.is_available():
        gpu_id = int(args.seed % torch.cuda.device_count())
    else:
        gpu_id = None
    if not args.no_gpu:
        try:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        except:
            pass
    experiment(variant)
    # run_experiment_here(
    #     experiment,
    #     variant,
    #     seed=args.seed,
    #     use_gpu=not args.no_gpu and torch.cuda.is_available(),
    #     gpu_id=gpu_id,
    #
    #     # Save the params every snapshot_gap and override previously saved result
    #     snapshot_gap=args.snapshot_gap,
    #     snapshot_mode=args.snapshot_mode,
    #     keep_first=args.keep_first,
    #
    #     log_dir=variant['log_dir']
    # )
