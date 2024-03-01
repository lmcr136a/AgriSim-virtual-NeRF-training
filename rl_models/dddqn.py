import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np


class DuelingDeepQNetwork(keras.Model):
    def __init__(self, n_actions=5, fc1_dims=512, fc2_dims=512):
        super(DuelingDeepQNetwork, self).__init__()
        
        self.checkpoint_file = os.path.join(
                    '_ddpg')

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_action = n_actions

        self.fc1 = Dense(self.fc1_dims, input_dim=1, activation='relu')
        self.fc2 = Dense(self.fc2_dims, input_dim=fc1_dims, activation='relu')
        self.V = Dense(1, activation=None)
        self.A = Dense(n_actions, activation=None)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        V = self.V(x)
        A = self.A(x)

        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))

        return Q

    def advantage(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        A = self.A(x)

        return A

class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                    dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                        dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class DDDQNAgent():
    def __init__(self, input_dims, epsilon=1, lr=1e-3, gamma=0.99, n_actions=2, batch_size=64,
                 epsilon_dec=1e-3, eps_end=0.01, 
                 mem_size=100000, fc1_dims=128,
                 fc2_dims=128, replace=100):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = eps_end
        self.replace = replace
        self.batch_size = batch_size

        self.learn_step_counter = 1
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)
        self.q_next = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)
        self.q_eval.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')
        self.q_next.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')
        self.action_space = [i for i in range(n_actions)]

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation, evaluate=False):
        if np.random.random() < self.epsilon and evaluate is False:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.advantage(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]
            print("Action shape:", action.shape)
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        if self.learn_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

        states, actions, rewards, states_, dones = \
                                    self.memory.sample_buffer(self.batch_size)

        q_pred = self.q_eval(states) 
        q_next = self.q_next(states_) 
        q_target = q_pred.numpy()
        max_actions = tf.math.argmax(self.q_eval(states_), axis=1)
        
        for idx, terminal in enumerate(dones):
            q_target[idx, actions[idx]] = rewards[idx] * 10 + \
                    self.gamma*q_next[idx, max_actions[idx]]*(1-int(dones[idx]))
        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                        self.eps_min else self.eps_min

        self.learn_step_counter += 1

    def save_models(self, i):
        print('... saving models ...')
        self.q_eval.save_weights("./models/" + str(i) + "_eval.h5")
        self.q_next.save_weights("./models/" + str(i) + "_target.h5")

    def load_models(self, i):
        print('... loading models ...')
        self.q_eval.load_weights("./models/" + str(i) + "_eval.h5")
        self.q_next.load_weights("./models/" + str(i) + "_target.h5")