# DQN implementation for Continual Reinforcement Learning with Complex Synapses
import numpy as np
import random
from collections import deque

import tensorflow as tf

TINY_CONSTANT = 0.0000000000001

class DQN:
    def __init__(self, num_states, num_actions, env, scope, task_specific=False, num_hidden=24, num_hidden_2=12,
                 mem_length=2000, pygame=False, optimiser='adam',
                 gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999, learning_rate=0.001, task_id=False,
                 num_tasks=2, multitask=False, metaplastic=False, meta_ts=True,
                 mod_all_layers=False, double=False, smax_tau=None, soft_tau=None, loss_type='mse', mod_flow=False,
                 flow_factor=1.0, ts_flow_pause=False, update_summary=False, final_bias=True):

        # Append task_id to state in the form of a 1-hot vector
        if task_id:
            self.num_states = num_states + num_tasks
        else:
            self.num_states = num_states
        self.num_actions = num_actions
        self.num_tasks = num_tasks
        self.task_specific = task_specific
        self.mod_all_layers = mod_all_layers
        self.multitask = multitask
        # If multitasking, maintain replay databases for each task
        if self.multitask:
            assert len(mem_length) == num_tasks
            self.memory = {}
            for task in mem_length:
                self.memory[task] = deque(maxlen=mem_length[task])
        # Otherwise only have one replay database, which is cleared at switch time
        else:
            self.mem_length = mem_length
            self.memory = deque(maxlen=mem_length)
        self.pygame = pygame
        self.env = env
        self.scope = scope
        self.smax_tau = smax_tau
        self.soft_tau =soft_tau
        self.double = double
        # cannot combine double and soft q learning
        assert not (self.soft_tau and self.double)
        # cannot have both soft Q learning and different softmax policy
        assert ((self.soft_tau == None) or (self.smax_tau == None))
        # Training parameters
        # discount factor
        self.gamma = gamma
        # initial exploration rate
        self.epsilon = epsilon
        # final epsilon
        self.epsilon_min = epsilon_min
        # exponential decay of epsilon
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate 
        self.lr_placeholder = tf.placeholder(tf.float32, shape=[])
        # number of hidden units in first layer
        self.num_hidden = num_hidden
        self.num_hidden_2 = num_hidden_2
        self.final_bias = final_bias
        # 'mse' or 'huber'
        self.loss_type = loss_type
        self.metaplastic = metaplastic
        if self.metaplastic and self.task_specific:
            # True if task-specific variables are metaplastic
            self.meta_ts = meta_ts
        else:
            self.meta_ts = False
        # make true to modulate benna fusi flow by derivative of policy wrt weights
        self.mod_flow = mod_flow
        if self.mod_flow:
            # Multiplicative factor for beaker flow
            self.flow_factor = flow_factor
        # Boolean variable for whether metaplasticity is paused in task_specific vars corresponding to task
        # that is not being trained on
        self.ts_flow_pause = ts_flow_pause
        self.optimiser = optimiser
        self.update_summary = update_summary
        self.create_model()
        
    def create_model(self):
        # Build graph
        with tf.variable_scope(self.scope):
            # placeholder for storing task number
            self.task_number = tf.placeholder(tf.int32, shape=(1,), name="task_num")
            if self.task_specific:
                # one-hot encoder for task number
                self.task_num_oh = tf.one_hot(self.task_number, self.num_tasks)

            # Input states
            self.inputs = tf.placeholder(shape=[1, self.num_states], dtype=tf.float32, name="inputs")

            # First hidden layer
            self.W_1 = tf.Variable(tf.random_normal([self.num_states, self.num_hidden], stddev=1.0/np.sqrt(self.num_states)),name="W_1")
            if self.task_specific and self.mod_all_layers:
                self.b_1 = tf.Variable(tf.constant(0.0, shape=[self.num_tasks, self.num_hidden]), name="b_1")
                self.g_1 = tf.Variable(tf.constant(1.0, shape=[self.num_tasks, self.num_hidden]), name="g_1")
            else:
                self.b_1 = tf.Variable(tf.constant(0.1, shape=[self.num_hidden]), name="b_1")
            if self.task_specific and self.mod_all_layers:
                self.h_1 = tf.multiply(tf.nn.relu(tf.matmul(self.inputs, self.W_1)+tf.matmul(self.task_num_oh, self.b_1)), tf.matmul(self.task_num_oh, self.g_1), name="h_1")
            else:
                self.h_1 = tf.nn.relu(tf.matmul(self.inputs, self.W_1)+self.b_1, name="h_1")

            # Second hidden layer
            self.W_2 = tf.Variable(tf.random_normal([self.num_hidden, self.num_hidden_2],stddev=1.0/np.sqrt(self.num_hidden)), name="W_2")
            if self.task_specific and self.mod_all_layers:
                self.b_2 = tf.Variable(tf.constant(0.0, shape=[self.num_tasks, self.num_hidden_2]), name="b_2")
                self.g_2 = tf.Variable(tf.constant(1.0, shape=[self.num_tasks, self.num_hidden_2]), name="g_2")
            else:
                self.b_2 = tf.Variable(tf.constant(0.0, shape=[self.num_hidden_2]), name="b_2")
            if self.task_specific and self.mod_all_layers:
                self.h_2 = tf.multiply(tf.nn.relu(tf.matmul(self.h_1, self.W_2)+tf.matmul(self.task_num_oh, self.b_2)), tf.matmul(self.task_num_oh, self.g_2), name="h_2")
            else:
                self.h_2 = tf.nn.relu(tf.matmul(self.h_1, self.W_2)+self.b_2, name="h_2")

            # Output layer
            self.W_3 = tf.Variable(tf.random_normal([self.num_hidden_2, self.num_actions], stddev=1.0/np.sqrt(self.num_hidden_2)), name="W_3")
            if self.task_specific:
                self.b_3 = tf.Variable(tf.constant(0.0,shape=[self.num_tasks,self.num_actions]),name="b_3")
                self.g_3 = tf.Variable(tf.constant(1.0,shape=[self.num_tasks,self.num_actions]),name="g_3")
                self.Qout = tf.multiply((tf.matmul(self.h_2,self.W_3)+tf.matmul(self.task_num_oh,self.b_3)),tf.matmul(self.task_num_oh,self.g_3))
            else:
                if self.final_bias:
                    self.b_3 = tf.Variable(tf.constant(0.0,shape=[1,self.num_actions]),name="b_3")
                    self.Qout = tf.matmul(self.h_2,self.W_3)+self.b_3
                else:
                    self.Qout = tf.matmul(self.h_2, self.W_3)
                    
            self.predict = tf.argmax(self.Qout, 1) # for greedy action selection

            if self.soft_tau is not None:
                # for softmax action selection
                self.action_probs = tf.nn.softmax((1.0/self.soft_tau)*self.Qout)
                # value function for soft q learning
                self.Vout = self.soft_tau * tf.reduce_logsumexp((1.0 / self.soft_tau) * self.Qout)
            else:
                self.Vout = tf.reduce_max(self.Qout)
                
            if self.smax_tau is not None:
                self.action_probs = tf.nn.softmax((1.0/self.smax_tau)*self.Qout) # for softmax action selection

            # Metaplastic vars
            if self.metaplastic:
                tf.add_to_collection("metaplastic", self.W_1)
                tf.add_to_collection("metaplastic", self.W_2)
                tf.add_to_collection("metaplastic", self.W_3)
                if not self.task_specific:
                    tf.add_to_collection("metaplastic", self.b_1)
                    tf.add_to_collection("metaplastic", self.b_2)
                    if self.final_bias:
                        tf.add_to_collection("metaplastic", self.b_3)
                elif self.meta_ts:
                    tf.add_to_collection("metaplastic", self.b_1)
                    tf.add_to_collection("metaplastic", self.g_1)
                    tf.add_to_collection("metaplastic", self.b_2)
                    tf.add_to_collection("metaplastic", self.g_2)
                    tf.add_to_collection("metaplastic", self.b_3)
                    tf.add_to_collection("metaplastic", self.g_3)
                    tf.add_to_collection("task_specific", self.b_1)
                    tf.add_to_collection("task_specific", self.g_1)
                    tf.add_to_collection("task_specific", self.b_2)
                    tf.add_to_collection("task_specific", self.g_2)
                    tf.add_to_collection("task_specific", self.b_3)
                    tf.add_to_collection("task_specific", self.g_3)
                    
            # Gradient updates
            self.nextQ = tf.placeholder(shape=[1, self.num_actions], dtype=tf.float32)
            if self.loss_type == 'mse':
                self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
            elif self.loss_type =='huber':
                self.loss=tf.losses.huber_loss(self.nextQ,self.Qout)

            if self.optimiser == 'adam':
                self.trainer = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder)
            self.updateModel = self.trainer.minimize(self.loss)

            # For recording magnitudes of updates
            if self.update_summary:
                self.prev_weights = [tf.placeholder(shape=var.get_shape(),dtype=var.dtype) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)]
                self.diff_mags = [tf.log(tf.abs(curr-prev)+TINY_CONSTANT,name=curr.op.name+'/diffmag') for curr,prev in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope),self.prev_weights)]
                for mag in self.diff_mags:
                    tf.summary.histogram(mag.name,mag)
                self.summary_op = tf.summary.merge_all()
                
            # Mean of absolute values of derivative of action probabilities (ie policy) wrt the trainable parameters
            # (for modulating beaker flow)
            if self.smax_tau or self.soft_tau:
                self.action_grads = [tf.squeeze(tf.reduce_mean(abs_grad, axis=-1)) for abs_grad in [tf.stack([tf.abs(tf.gradients(a_prob,var)) for a_prob in tf.unstack(self.action_probs,axis=1)],axis=-1) for var in tf.get_collection("metaplastic",scope=self.scope)]]

        # Create flow_mods if metaplasticity is paused on task_specific variables while other task being trained
        if self.task_specific and self.ts_flow_pause and self.metaplastic:
            ts_var_names = [ts_var.name for ts_var in tf.get_collection("task_specific", scope=self.scope)]
            self.pause_flow_mods = []
            for var in tf.get_collection("metaplastic",scope=self.scope):
                if var.name in ts_var_names:
                    pause_flow_mod = tf.multiply(tf.reshape(self.task_num_oh,(self.num_tasks,1)),tf.ones(var.get_shape()))
                else:
                    pause_flow_mod = tf.ones(var.get_shape())
                self.pause_flow_mods.append(pause_flow_mod)
                    
    def remember(self, state, action, reward, next_state, done, curr_task=None):
        if curr_task == 'cartpole':
            task_num = 0
        elif curr_task == 'catcher':
            task_num = 1
        elif curr_task == 'cartpole_long':
            task_num = 2
            
        if self.multitask:
            self.memory[curr_task].append((state, action, reward, next_state, done,task_num))
        else:
            self.memory.append((state, action, reward, next_state, done, task_num))

    def act(self, state, curr_task, test=False):
        sess = tf.get_default_session()
        a = [None]
        if curr_task == 'cartpole':
            task_num = 0
        elif curr_task == 'catcher':
            task_num = 1
        elif curr_task == 'cartpole_long':
            task_num = 2
        if not test and (np.random.rand(1) < self.epsilon):
            if self.pygame:
                a[0] = random.randint(0, self.num_actions-1)
            else:
                a[0] = self.env.action_space.sample()
        else:
            if self.soft_tau or self.smax_tau:
                action_probs, self.value = sess.run([self.action_probs, self.Vout], feed_dict={self.inputs: state, self.task_number: [task_num]})
                action_probs = action_probs[0]
                action_probs /= action_probs.sum()
                assert ~np.any(np.isnan(action_probs))
                a[0] = np.random.choice(range(self.num_actions), p=action_probs)
            else:
                a, self.allQ = sess.run([self.predict, self.Qout], feed_dict={self.inputs: state})
                self.value = np.max(self.allQ[0])    
               
        return a[0], self.value

    def replay(self, batch_size, bf=None, flow_switches=None, target_agent=None, curr_task=None, bf_manual=False):
        sess = tf.get_default_session()
        if self.multitask:
            if curr_task == -1:
                minibatch = []
                microbatch_size = batch_size/self.num_tasks
                for task in self.memory:
                    minibatch.extend(random.sample(self.memory[task], microbatch_size))
                random.shuffle(minibatch)
            else:
                minibatch = random.sample(self.memory[curr_task], batch_size)
        else:
            minibatch = random.sample(self.memory, batch_size)
        losses = []
        if curr_task == 'cartpole':
            curr_task = 0
        elif curr_task == 'catcher':
            curr_task = 1
        elif curr_task == 'cartpole_long':
            curr_task = 2
        # List to hold means of derivatives of policy wrt variables to weight modulation of flow in Benna Fusi beakers
        if self.mod_flow and self.metaplastic:
            flow_mods = [np.zeros(var.get_shape()) for var in tf.get_collection("metaplastic", scope=self.scope)]
        elif self.task_specific and self.ts_flow_pause and self.metaplastic:
            assert not self.multitask
            flow_mods = sess.run(self.pause_flow_mods, feed_dict={self.task_number: [curr_task]})
        else:
            flow_mods = None

        # Record magnitudes of weight updates
        if self.update_summary:
            prev_weights = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope))
        
        # Update weights with minibatch
        for state, action, reward, next_state, done, task_num in minibatch:
            target = reward
            if not done:
                if target_agent == None:
                    V1 = sess.run(self.Vout, feed_dict={self.inputs: next_state, self.task_number: [task_num]})
                else:
                    if self.soft_tau:
                        V1 = sess.run(target_agent.Vout, feed_dict={target_agent.inputs: next_state, target_agent.task_number: [task_num]})
                    else:
                        Q1 = sess.run(target_agent.Qout, feed_dict={target_agent.inputs:next_state,target_agent.task_number:[task_num]})
                        if self.double:
                            best_a = sess.run(self.predict, feed_dict={self.inputs:next_state,self.task_number:[task_num]})
                            V1 = Q1[0][best_a]
                        else:
                            V1 = np.max(Q1)
                target = reward + self.gamma * V1

            if target_agent == None:
                targetQ = sess.run(self.Qout, feed_dict={self.inputs: state, self.task_number: [task_num]})
            else:
                targetQ = sess.run(target_agent.Qout, feed_dict={target_agent.inputs: state, target_agent.task_number:[task_num]})

            targetQ[0][action] = target
            if self.mod_flow:
                loss, flow_mod, _ = sess.run([self.loss, self.action_grads,self.updateModel],feed_dict={self.inputs:state,self.task_number:[task_num],self.nextQ:targetQ,self.lr_placeholder:self.learning_rate})
            else:
                loss, _ = sess.run([self.loss, self.updateModel], feed_dict={self.inputs: state, self.task_number:[task_num], self.nextQ:targetQ,self.lr_placeholder:self.learning_rate})
            losses.append(loss)
            # Accumulate flow modulation factors
            if self.mod_flow and self.metaplastic:
                flow_mods = map(np.add,flow_mods, flow_mod)
                
            if (not bf_manual) and (bf is not None) and (bf.slow_update==False):
                assert not self.mod_flow
                bf.apply_flow(sess=sess, flow_switches=flow_switches,flow_mods=flow_mods)

        if self.mod_flow and self.metaplastic:
            flow_mods=[self.flow_factor*np.divide(el,batch_size) for el in flow_mods]
        
        if (not bf_manual) and (bf is not None) and bf.slow_update:
            bf.apply_flow(sess=sess, flow_switches=flow_switches,flow_mods=flow_mods)

        if self.update_summary:
            summ_feed_dict = {i: j for i, j in zip(self.prev_weights, prev_weights)}
            summ = sess.run(self.summary_op, feed_dict=summ_feed_dict)
        else:
            summ = None
            
        return np.mean(losses), summ

    def update_target_graph(self, policy_scope):
        policy_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=policy_scope)
        target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        ops = []
        for i in range(len(target_vars)):
            ops.append(target_vars[i].assign(policy_vars[i]))
        self.update_target_ops = ops

        return ops

    def filtered_update_target_graph(self, policy_scope, tau):
        policy_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=policy_scope)
        target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        ops = []
        for i in range(len(target_vars)):
            ops.append(target_vars[i].assign(tau * policy_vars[i] + (1-tau) * target_vars[i]))
        self.filtered_update_target_ops = ops

        return ops

    def update_target(self, sess):
        for op in self.update_target_ops:
            sess.run(op)

    def filtered_update_target(self, sess):
        for op in self.filtered_update_target_ops:
            sess.run(op)

    def set_epsilon(self, new_eps=1.0):
        self.epsilon = new_eps

    def set_gamma(self,gamma):
        self.gamma = gamma

    def set_is_pygame(self,pygame):
        self.pygame = pygame

    def reset_memory(self, mem_length):
        self.mem_length = mem_length
        self.memory = deque(maxlen=mem_length)

    def set_learning_rate(self,new_lr):
        self.learning_rate = new_lr

    def set_env(self, env):
        self.env = env

    
