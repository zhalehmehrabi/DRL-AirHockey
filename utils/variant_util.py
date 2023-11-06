from torch.nn import functional as F
import random
def build_variant(variant, return_replay_buffer=True, return_collectors=True):
    env_args = {}
    alg_args = {}
    learn_args = {}
    log_args = {}

    log_args['log_dir'] = variant['log_dir']
    log_args['alg'] = variant['alg']
    learn_args['tb_log_name'] = variant['log_dir']

    res = {}

    domain = variant['domain']
    seed = variant['seed']
    env_args = {}
    if domain in ['riverswim']:
        env_args['dim'] = variant['dim']
        env_args['deterministic'] = False
        if 'deterministic_rs' in variant:
            env_args['deterministic'] = variant['deterministic_rs']
    if domain in ['lqg']:
        env_args['sigma_noise'] = variant['sigma_noise']
    if domain in ['point']:
        env_args['difficulty'] = variant['difficulty']
        env_args['clip_state'] = variant['clip_state']
        env_args['terminal'] = variant['terminal']
        env_args['sparse_reward'] = variant['sparse_reward']
        env_args['max_state'] = variant['max_state']
    if domain in ['ant_maze']:
        env_args['difficulty'] = variant['difficulty']
        env_args['clip_state'] = variant['clip_state']
        env_args['terminal'] = variant['terminal']
    if 'cliff' in domain:
        env_args['sigma_noise'] = variant['sigma_noise']
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
        env_args['acceleration'] = variant['acceleration']
        env_args['delta_ratio'] = variant['delta_ratio']
        env_args['max_accel'] = variant['max_accel']
        env_args['gamma'] = variant['trainer_kwargs']['discount']
        env_args['horizon'] = variant['algorithm_kwargs']['max_path_length'] + 1
        env_args['stop_after_hit'] = variant['stop_after_hit']
        env_args['punish_jerk'] = variant['punish_jerk']
        env_args['interpolation_order'] = variant['interpolation_order']
        env_args['include_old_action'] = variant['include_old_action']
        env_args['use_aqp'] = variant['use_aqp']
        env_args['speed_decay'] = variant['speed_decay']
        env_args['clip_vel'] = variant['clip_vel']
        env_args['aqp_terminates'] = variant['aqp_terminates']
        env_args['whole_game_reward'] = variant['whole_game_reward']
        env_args['score_reward'] = variant['score_reward']
        env_args['fault_penalty'] = variant['fault_penalty']
        env_args['load_second_agent'] = variant['load_second_agent']
        env_args['dont_include_timer_in_states'] = variant['dont_include_timer_in_states']
        env_args['action_persistence'] = variant['action_persistence']
        env_args['stop_when_puck_otherside'] = variant['stop_when_puck_otherside']

        env_args['curriculum_learning_step1'] = variant['curriculum_learning_step1']
        env_args['curriculum_learning_step2'] = variant['curriculum_learning_step2']
        env_args['curriculum_learning_step3'] = variant['curriculum_learning_step3']

        env_args['start_from_defend'] = variant['start_from_defend']
        env_args['curriculum_transition'] = variant['curriculum_transition']

        env_args['original_env'] = True

    if variant['alg'] == 'ppo':
        alg_args['learning_rate'] = variant.lr
        try:
            alg_args['policy'] = variant.policy
        except Exception:
            alg_args['policy'] = 'MlpPolicy'
        alg_args['n_steps'] = variant.steps_per_update
        alg_args['batch_size'] = variant.batch_size
        alg_args['n_epochs'] = variant.n_epochs
        alg_args['gamma'] = variant.gamma
        alg_args['gae_lambda'] = variant.gae_lambda
        alg_args['clip_range'] = variant.clip_range
        alg_args['clip_range_vf'] = variant.clip_range_vf
        alg_args['normalize_advantage'] = variant.normalize_advantage
        alg_args['ent_coef'] = variant.ent_coef
        alg_args['vf_coef'] = variant.vf_coef
        alg_args['max_grad_norm'] = variant.max_grad_norm
        alg_args['use_sde'] = variant.use_sde
        alg_args['sde_sample_freq'] = variant.sde_sample_freq
        alg_args['target_kl'] = variant.target_kl
        alg_args['stats_window_size'] = variant.stats_window_size
        alg_args['tensorboard_log'] = variant.tensorboard_log
        alg_args['verbose'] = variant.verbose
        alg_args['seed'] = variant.seed
    elif variant['alg'] == 'sac':
        alg_args['policy'] = variant['policy']
        alg_args['learning_rate'] = variant['trainer_kwargs']['policy_lr']  # the same will be used for all networks
        alg_args['buffer_size'] = variant['replay_buffer_size']
        alg_args['learning_starts'] = variant['algorithm_kwargs']['min_num_steps_before_training']
        alg_args['batch_size'] = variant['algorithm_kwargs']['batch_size']
        alg_args['tau'] = variant['trainer_kwargs']['soft_target_tau']
        alg_args['gamma'] = variant['trainer_kwargs']['discount']
        alg_args['train_freq'] = variant['algorithm_kwargs']['num_expl_steps_per_train_loop']
        alg_args['gradient_steps'] = variant['algorithm_kwargs']['num_trains_per_train_loop']
        alg_args['action_noise'] = variant['trainer_kwargs']['action_noise']
        alg_args['replay_buffer_class'] = None
        alg_args['replay_buffer_kwargs'] = None
        alg_args['optimize_memory_usage'] = False
        alg_args['ent_coef'] = "auto"
        alg_args['target_update_interval'] = variant['trainer_kwargs']['target_update_period']
        alg_args['target_entropy'] = "auto"
        alg_args['use_sde'] = False
        alg_args['sde_sample_freq'] = -1
        alg_args['use_sde_at_warmup'] = True
        alg_args['stats_window_size'] = 100
        alg_args['tensorboard_log'] = variant['log_dir']
        alg_args['policy_kwargs'] = None
        alg_args['verbose'] = variant['verbose']
        if variant['seed'] == 0:
            variant['seed'] = random.randint(0, 999999)
        alg_args['seed'] = variant['seed']
        alg_args['device'] = "auto"
        alg_args['_init_setup_model'] = True
    elif variant['alg'] == 'dqn':
        alg_args['policy'] = variant.policy
        alg_args['learning_rate'] = variant.learning_rate
        alg_args['buffer_size'] = variant.buffer_size
        alg_args['learning_starts'] = variant.learning_starts
        alg_args['batch_size'] = variant.batch_size
        alg_args['tau'] = variant.tau
        alg_args['gamma'] = variant.gamma
        alg_args['train_freq'] = variant.train_freq
        alg_args['gradient_steps'] = variant.gradient_steps
        alg_args['replay_buffer_class'] = variant.replay_buffer_class
        alg_args['replay_buffer_kwargs'] = variant.replay_buffer_kwargs
        alg_args['optimize_memory_usage'] = variant.optimize_memory_usage
        alg_args['target_update_interval'] = variant.target_update_interval
        alg_args['exploration_fraction'] = variant.exploration_fraction
        alg_args['exploration_initial_eps'] = variant.exploration_initial_eps
        alg_args['exploration_final_eps'] = variant.exploration_final_eps
        alg_args['max_grad_norm'] = variant.max_grad_norm
        alg_args['stats_window_size'] = variant.stats_window_size
        alg_args['tensorboard_log'] = variant.tensorboard_log
        alg_args['policy_kwargs'] = variant.policy_kwargs
        alg_args['verbose'] = variant.verbose
        alg_args['seed'] = variant.seed
        alg_args['device'] = variant.device
        alg_args['_init_setup_model'] = variant._init_setup_model

    learn_args['total_timesteps'] = variant['total_timesteps']

    return env_args, alg_args, learn_args, log_args
