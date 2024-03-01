import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gym
from gym import spaces
from ingp import INGPTrainer

class CustomEnv(gym.Env):
    def __init__(self, c):
        super(CustomEnv, self).__init__()
        self.trainer = INGPTrainer(c)
        self.baseline_performance = 0.8423  # ssim
        self.per_iter = self.trainer.args.n_steps / c.K      # how many steps for one episode 
                                                            # if 35000 iteration for the baseline and K is 10, 
                                                            # then it's 3500.
        self.per_images = int(c.total_images / c.K)

        low_range = [[-c.pose_max, c.pose_min, -c.pose_max]]*self.per_images
        high_range = [[c.pose_max, c.pose_max, c.pose_max]]*self.per_images
        self.action_space = spaces.Box(low=np.array(low_range), 
                                       high=np.array(high_range), 
                                       shape=(self.per_images, 3), dtype=np.float32)  # xyz 
        # self.observation_space = spaces.Box(
        #                             low=np.array([0, 0, 0]), 
        #                             high=np.array([self.MAX_X, self.MAX_Y, self.MAX_Z]),
        #                             dtype=np.float32
        #                             )
        self.state = []

        self.trainer.args.n_steps = self.per_iter


        self.seed()
        self.reset()

    def seed(self, seed=447):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, new_poses, epo):
        
        # self.trainer.train(new_poses)
        reward = self.trainer.val(epo)    # ssim
        
        self.state.append(new_poses.tolist())   # state = list of selected poses
        done = len(self.state) >= 100 or reward > self.baseline_performance
        # print(self.state)
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = []  # initial selected pose
        return np.array(self.state)

class DDQN:
    def __init__(self, state_size, action_size, env):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        act_values = self.model.predict(state)
        return act_values[0]  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def run_rl(c):
    env = CustomEnv(c)
    state_size = len(env.reset())
    action_size = env.action_space.shape[0]
    agent = DDQN(state_size, action_size, env)
    batch_size = 32

    epo = 0
    for e in range(c.episodes):
        state = env.reset()
        # state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action, epo)
            epo += 1
            reward = reward if not done else -1
            # next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            # if done:
            print("episode: {}/{}, reward: {}, e: {:.2}"
                    .format(e, c.episodes, reward, agent.epsilon))
                # break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 10 == 0:
            agent.save("./save/cartpole-ddqn.h5")

    

if __name__ == "__main__":
    from utils import get_configs
    np.random.seed(447)
    c = get_configs()
    run_rl(c)