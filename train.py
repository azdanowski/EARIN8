import numpy as np
import gym
from matplotlib import pyplot as plt

EPSILON = 0.5
EPSILON_MIN = 0.01
EPSILON_MAX = 1
DECAY = 0.05
EPISODES = 5000
ALPHA = 0.1
GAMMA = 0.6


def train(q_table, env):
    too_long_episodes = 0
    rewards_plt = []
    for i in range(EPISODES):
        state = env.reset()
        episode_length, reward, penalties, reward_penalty_sum, = 0, 0, 0, 0
        done = False
        #episode ends with dropoff + 20 reward
        while ( not done ): #and episode_length < 200 is done automatically by gym
            state, reward, done = _step(q_table, env, state)
            reward_penalty_sum += reward
            
            if reward == -10: 
                penalties += 1
            
            episode_length += 1
            if episode_length == 200:
                too_long_episodes += 1
                #print("Episode LENGTH EXCEEDED minimum!")
        global EPSILON
        EPSILON = (
            EPSILON_MIN + (EPSILON_MAX - EPSILON_MIN) * np.exp(-DECAY * EPISODES)
        )
        if i % 100 == 0:      
            print(f"Episode: {i+1}")
            print(f"\tLength: {episode_length}, Penalties: "
            +f"{penalties}, Reward: {reward_penalty_sum}")
        rewards_plt.append(reward_penalty_sum)
        
    print(f"Training finished. Episodes which were cut at 200 steps:"
          +f" {too_long_episodes} ({too_long_episodes/EPISODES*100}%).")
    plt.plot(rewards_plt)
    plt.savefig("rewards_vs_episode")
    return q_table
                                                                
def _step(q_table, env, state):
    action = choose_action(q_table, env, state)
    
    next_state, reward, done, _, = env.step(action)
    prev_q = q_table[state][action]
    
    
    next_max_val = np.max(q_table[next_state])
    
    new_q = (1 - ALPHA) * prev_q + ALPHA * (reward + GAMMA * next_max_val)
    q_table[state][action] = new_q
    
    return next_state, reward, done

def choose_action(q_table, env, state, eps=EPSILON):
    if np.random.uniform(0, 1) < EPSILON:
        action = env.action_space.sample()
    else:
        action =  np.argmax(q_table[state, :])
    return action
    
def main():
    env = gym.make("Taxi-v3")
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    q_table = train(q_table, env)
    np.save("model.npy", q_table)
    
if __name__ == "__main__":
    main()
    