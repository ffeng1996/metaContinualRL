import os

os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import numpy as np

import tensorflow as tf

from ple.games.catcher import Catcher
from ple import PLE
import gym

from dqn import DQN


def process_state(state):
    return np.array([state.values()])


# Normalise states to roughly 0 mean, 1 std
def normalise_state(state, means, stds, task_id=False, num_tasks=2, curr_task=0):
    normed_state = np.divide(np.array(state) - np.array(means), np.array(stds))
    if curr_task == 'cartpole':
        curr_task = 0
    elif curr_task == 'catcher':
        curr_task = 1
    if task_id:
        temp_state = np.zeros((1, num_tasks + len(normed_state[0])))
        temp_state[0][curr_task + len(normed_state[0])] = 1
        temp_state[0][:len(normed_state[0])] = normed_state[0]
        normed_state = temp_state
    return normed_state


# Simulation params
sim_params = {}
sim_params['num_epochs'] = 20
sim_params['start_epoch'] = 0
sim_params['num_eps_per_epoch'] = 20000
sim_params['episode_length'] = 500
sim_params['num_tasks'] = 2
sim_params['num_inputs'] = 4
sim_params['num_actions'] = 2
sim_params['first_task'] = "cartpole"
sim_params['update_summary'] = False

# Training params
train_params = {}
# For multitasking (i.e., training both tasks at the same time), set num_eps_per_epoch to 1
train_params['multitask'] = False
train_params['lrate'] = 0.001
train_params['hetero_lr'] = False
train_params['hetero_dist'] = None
train_params['epsilon_decay'] = 0.9995
train_params['eps_start'] = 1.0
train_params['epsilon_min'] = 0.0
# How often to test network with epsilon=0 (continual: episodes, multitask: epochs)
train_params['test_freq'] = 10
# Reward scaling (default cartpole)
train_params['r_scale'] = 1
# True if scale catcher reward rather than cartpole
train_params['catcherscale'] = False
train_params['replay_sizes'] = {'cartpole': 2000, 'catcher': 2000}
train_params['loss_type'] = 'swc'
# Set value for soft q temperature (i.e., \alpha in Eq.12), set as None for normal q learning
train_params['soft_tau'] = 0.01
# Set value for boltzmann policy temp, set as None for greedy policy
train_params['smax_tau'] = None
# Set true for double Q learning
train_params['double'] = False
# Set true for online learning
train_params['online'] = False
train_params['update_freq'] = 20

# Architecture params
arch_params = {}
arch_params['num_hidden1'] = 400
arch_params['num_hidden2'] = 200
arch_params['final_bias'] = True
arch_params['target_net'] = True
arch_params['filtered_target_update'] = False
arch_params['target_taus'] = {'cartpole': 0.01, 'catcher': 0.01}
# epochs if multitask, otherwise episodes
arch_params['target_update_freq'] = 10
# Append one hot vector of task-id to state vector
arch_params['task_id'] = False
# Set true for task_specific biases and gains
arch_params['task_spec_vars'] = True
# Set true for ts vars for all layers (not just output)
arch_params['mod_all_layers'] = True

# Cartpole setup and params
cartpole_dict = {}
cartpole_dict['gamma'] = 0.95
cartpole_dict['learning_rate'] = train_params['lrate']
cartpole_dict['batch_size'] = 64
cartpole_dict['memory_size'] = train_params['replay_sizes']['cartpole']
cartpole_dict['pygame'] = False
cartpole_dict['state_means'] = [0, 0, 0, 0]
cartpole_dict['state_stds'] = [0.75, 0.75, 0.07, 0.3]

catcher_dict = {}
catcher_dict['gamma'] = 0.99
catcher_dict['learning_rate'] = train_params['lrate']
catcher_dict['batch_size'] = 64
catcher_dict['memory_size'] = train_params['replay_sizes']['catcher']
catcher_dict['pygame'] = True
catcher_dict['state_means'] = [29.88745927, 0.15930137, 22.5392288, 24.73781436]
catcher_dict['state_stds'] = [13.89457683, 2.04087944, 17.41686248, 23.38546788]

game_params = {'cartpole': cartpole_dict, 'catcher': catcher_dict}

if __name__ == "__main__":
    # Initiate cartpole envs
    cartpole_env = gym.make('CartPole-v1')
    # Initiate catcher envs
    catcher_env = PLE(Catcher(init_lives=1), state_preprocessor=process_state, display_screen=False)
    catcher_env.init()

    game_params['catcher']['actions'] = catcher_env.getActionSet()

    envs = {'cartpole': cartpole_env, 'catcher': catcher_env}

    # Initialise the first task: cartpole
    curr_task = sim_params['first_task']

    env = envs[curr_task]

    # Multiple replay databases maintained if multitasking
    if train_params['multitask']:
        mem_length = train_params['replay_sizes']
    else:
        mem_length = game_params[curr_task]['memory_size']

    # Create agent
    agent = DQN(sim_params['num_inputs'], sim_params['num_actions'], env, scope="policy",
                multitask=train_params['multitask'],
                task_specific=arch_params['task_spec_vars'], gamma=game_params[curr_task]['gamma'],
                metaplastic=False,
                mem_length=mem_length, epsilon=train_params['eps_start'], epsilon_decay=train_params['epsilon_decay'],
                epsilon_min=train_params['epsilon_min'], learning_rate=game_params[curr_task]['learning_rate'],
                pygame=game_params[curr_task]['pygame'],
                num_hidden=arch_params['num_hidden1'], num_hidden_2=arch_params['num_hidden2'],
                task_id=arch_params['task_id'], num_tasks=sim_params['num_tasks'],
                mod_all_layers=arch_params['mod_all_layers'], double=train_params['double'],
                loss_type=train_params['loss_type'],
                soft_tau=train_params['soft_tau'], smax_tau=train_params['smax_tau'],
                update_summary=sim_params['update_summary'], final_bias=arch_params['final_bias'])

    # Create target agent
    if arch_params['target_net']:
        target_agent = DQN(sim_params['num_inputs'], sim_params['num_actions'], env, scope="target",
                           multitask=train_params['multitask'],
                           task_specific=arch_params['task_spec_vars'], gamma=game_params[curr_task]['gamma'],
                           metaplastic=False,
                           mem_length=mem_length, epsilon=train_params['eps_start'],
                           epsilon_decay=train_params['epsilon_decay'],
                           epsilon_min=train_params['epsilon_min'],
                           learning_rate=game_params[curr_task]['learning_rate'],
                           pygame=game_params[curr_task]['pygame'],
                           num_hidden=arch_params['num_hidden1'], num_hidden_2=arch_params['num_hidden2'],
                           task_id=arch_params['task_id'], num_tasks=sim_params['num_tasks'],
                           mod_all_layers=arch_params['mod_all_layers'], double=train_params['double'],
                           loss_type=train_params['loss_type'],
                           soft_tau=train_params['soft_tau'], smax_tau=train_params['smax_tau'],
                           final_bias=arch_params['final_bias'])

        if arch_params['filtered_target_update']:
            target_agent.update_target_graph("policy")
            target_agent.filtered_update_target_graph("policy", arch_params['target_taus'][curr_task])
        else:
            target_agent.update_target_graph("policy")
    else:
        target_agent = None

    init = tf.global_variables_initializer()

    # Monitors
    # training scores
    scores = []
    mean_losses = []
    mean_qs = []
    num_active_flows = []

    # configure computation way of tf.Session, i.e., CPU / GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Simulation
    with tf.Session(config=config) as sess:
        sess.run(init)
        update_count = 0

        for epoch in range(sim_params['start_epoch'], sim_params['start_epoch'] + sim_params['num_epochs']):
            ep_lengths = []
            test_scores = []

            mean_loss_ma = 0
            ep_length_ma = 0
            meanq_ma = 0
            score_ma = 0
            if train_params['multitask']:
                if epoch % train_params['test_freq'] < 2:
                    test = True
                else:
                    test = False
            if epoch > 0:
                # Switch current task and reset hyper params
                if curr_task == 'cartpole':
                    curr_task = 'catcher'
                elif curr_task == 'catcher':
                    curr_task = 'cartpole'

                env = envs[curr_task]
                agent.set_gamma(game_params[curr_task]['gamma'])
                agent.set_is_pygame(game_params[curr_task]['pygame'])
                agent.set_learning_rate(game_params[curr_task]['learning_rate'])
                agent.set_env(env)
                if not train_params['multitask']:
                    agent.set_epsilon(train_params['eps_start'])
                    agent.reset_memory(mem_length=game_params[curr_task]['memory_size'])

            if arch_params['target_net']:
                if arch_params['filtered_target_update']:
                    target_agent.filtered_update_target_graph("policy", arch_params['target_taus'][curr_task])
                    target_agent.update_target(sess)
                elif (epoch % arch_params['target_update_freq'] == 0):
                    target_agent.update_target(sess)

            for i in range(sim_params['num_eps_per_epoch']):
                totalq = 0
                if not train_params['multitask']:
                    if i % train_params['test_freq'] == 0:
                        test = True
                    else:
                        test = False
                    if arch_params['target_net']:
                        if arch_params['filtered_target_update']:
                            target_agent.filtered_update_target(sess)
                        elif (i % arch_params['target_update_freq'] == 0):
                            target_agent.update_target(sess)

                if game_params[curr_task]['pygame']:
                    env.reset_game()
                    state = env.getGameState()
                else:
                    state = env.reset()
                    state = np.reshape(state, [1, sim_params['num_inputs']])
                state = normalise_state(state, game_params[curr_task]['state_means'],
                                        game_params[curr_task]['state_stds'],
                                        task_id=arch_params['task_id'], num_tasks=sim_params['num_tasks'],
                                        curr_task=curr_task)

                total_r = 0
                done = False

                for t in range(sim_params['episode_length']):
                    # Choose action
                    action, maxQ = agent.act(state, curr_task, test=test)
                    totalq += maxQ
                    if game_params[curr_task]['pygame']:
                        reward = env.act(game_params[curr_task]['actions'][action])
                        if reward < 0:
                            reward = -1
                        next_state = env.getGameState()
                        done = env.game_over()
                        if t > (sim_params['episode_length'] - 2):
                            done = True
                        if train_params['catcherscale']:
                            reward = reward * train_params['r_scale']
                    else:
                        next_state, reward, done, _ = env.step(action)
                        next_state = np.reshape(next_state, [1, sim_params['num_inputs']])
                        if not train_params['catcherscale']:
                            reward = train_params['r_scale'] * reward

                    next_state = normalise_state(next_state, game_params[curr_task]['state_means'],
                                                 game_params[curr_task]['state_stds'],
                                                 task_id=arch_params['task_id'], num_tasks=sim_params['num_tasks'],
                                                 curr_task=curr_task)
                    agent.remember(state, action, reward, next_state, done, curr_task=curr_task)
                    total_r += reward
                    state = next_state

                    if (train_params['online']) and (t % train_params['update_freq'] == 0) and (
                            len(agent.memory) >= game_params[curr_task]['batch_size']):
                        mean_loss, summ = agent.replay_meta(game_params[curr_task]['batch_size'],
                                                            target_agent=target_agent,
                                                            curr_task=curr_task)

                    if done:
                        if agent.epsilon > train_params['epsilon_min']:
                            agent.set_epsilon(agent.epsilon * train_params['epsilon_decay'])
                        if not test:
                            ep_lengths.append(t)

                            scores.append(total_r)
                            ep_length_ma = 0.99 * ep_length_ma + 0.01 * t
                            score_ma = 0.999 * score_ma + 0.001 * total_r
                        else:
                            test_scores.append(total_r)

                        '''
                        print("episode: {}/{}, score: {}, epsilon: {}".format(i, sim_params['num_eps_per_epoch'], total_r,
                                                                            agent.epsilon))
                        '''
                        # print ep_length_ma
                        break

                if not test and not train_params['online']:
                    if ((not train_params['multitask']) and (
                            len(agent.memory) >= game_params[curr_task]['batch_size'])) or (
                            train_params['multitask'] and all(
                        len(agent.memory[task]) >= game_params[task]['batch_size'] for task in agent.memory)):

                        mean_loss, summ = agent.replay_meta(game_params[curr_task]['batch_size'],
                                                            target_agent=target_agent, curr_task=curr_task)
                        update_count += 1

                        if train_params['multitask']:
                            mean_losses.append(mean_loss)
                        else:
                            mean_losses.append(mean_loss)
                        mean_loss_ma = 0.99 * mean_loss_ma + 0.01 * mean_loss
                        # print "Mean loss: ", mean_loss_ma

                if train_params['multitask']:
                    mean_qs.append(totalq / t)
                else:
                    mean_qs.append(totalq / t)
                meanq_ma = 0.99 * meanq_ma + 0.01 * totalq / t
                # print "Q:", meanq_ma

            print ' ==> finish epoch#' + str(epoch) + '... <=='
            with open('ep_len_MER.txt', 'a+') as f:
                f.write(str(ep_lengths) + '\n')

            with open('test_score_MER.txt', 'a+') as f:
                f.write(str(test_scores) + '\n')
