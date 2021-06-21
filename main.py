import gym
import numpy as np
from utils import fix_rendering_on_windows
from train import choose_action
import pickle


LENGTH = 200
def visualize(q_table, env, length):
    state = env.reset()
    total_reward = 0
    for i in range(length):
        env.render()
        
        action =  np.argmax(q_table[state, :])
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            print(f"Done after {i} steps. Total reward:", total_reward)
            break


def main():
    env = gym.make("Taxi-v3").env
    fix_rendering_on_windows()
    q_table = np.load("model.npy", allow_pickle = True)
    visualize(q_table, env, LENGTH)
    

if __name__ == "__main__":
    main()