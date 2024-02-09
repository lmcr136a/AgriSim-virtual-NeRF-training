import os
import time
import numpy as np
from rl_models.dddqn import DDDQNAgent
from rl_env import NeRFENV


"""

selec_idx = RL_prediction(images, positions, position_candidates, new_pos_num)


"""

class RL():
    def __init__(self, args, position_candidates) -> None:
    
        self.env = NeRFENV(args, position_candidates)
        self.agent = DDDQNAgent(env=self.env, lr=1e-3, gamma=0.99, n_actions=5, epsilon=1.0,
                batch_size=64, input_dims=[1])
        self.n_games = args.rl_n_games
        self.scores = []
        self.eps_history = []
        self.eval = args.rl_eval
        self.log_dir = os.path.join(args.basedir, args.expname)
        self.checkpoint = os.path.join(self.log_dir, "RLcheckpoint")
        if os.path.exists(self.checkpoint):
            print("Load RL checkpoint from ", self.checkpoint)
            n_steps = 0
            while n_steps <= self.agent.batch_size:
                observation = self.env.reset()
                action = np.random.choice(self.env.action_space)
                observation_, reward, done, info = self.env.step(action)
                self.agent.store_transition(observation, action, reward, observation_, done)
                n_steps += 1
            self.agent.learn()
            model_name = input("Enter model name under models dir (ex. dueling): ")
            self.agent.load_models(model_name)

        if self.eval is True: 
            print("RL is in evaluation mode")
            self.n_games = 1

    def train(self, ):
        for i in range(self.n_games):
            done = False
            score = 0
            observation = self.env.reset()
            print("initial state: ", observation, end='\t')
            while not done:
                # if you train model, change choose_action's evaluate to False
                action = self.agent.choose_action(observation, evaluate=True)
                observation_, reward, done, info = self.env.step(action)
                print(f"- action: {action} | state: {observation_} | reward: {reward}")
                score += reward
                self.agent.store_transition(observation, action, reward, observation_, done)
                if not self.eval:
                    self.agent.learn()
                observation = observation_
            self.eps_history.append(self.agent.epsilon)

            self.scores.append(score)

            avg_score = np.mean(self.scores[-100:])
            print('| episode: ', i,'\t| score: %.2f' % score, '\t| average score %.2f' % avg_score)
            print(" - last state: ", observation, "\t| reward: ", reward, "\t| action: ", action, "\t| epsilon: ", self.agent.epsilon)
            
            if (i % 20 == 0 and i != 0) or i == self.n_games - 1:
                if not self.checkpoint:
                    self.agent.save_models(os.path.join(self.log_dir,'h5_duel' + str(i)))