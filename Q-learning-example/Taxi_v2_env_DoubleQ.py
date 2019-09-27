'''
code from https://github.com/simoninithomas/Deep_reinforcement_learning_Course/
'''

import numpy as np
import gym
import random


env = gym.make("Taxi-v2")
env.render()

action_size = env.action_space.n
print("Action size ", action_size)

state_size = env.observation_space.n
print("State size ", state_size)

qtable = np.zeros((state_size, action_size))
qtable1 = np.zeros((state_size, action_size))
qtable2 = np.zeros((state_size, action_size))
print(qtable)

total_episodes = 50000        # Total episodes
total_test_episodes = 100     # Total test episodes
max_steps = 99                # Max steps per episode

learning_rate = 0.7           # Learning rate
gamma = 0.618                 # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.01             # Exponential decay rate for exploration prob

# 2 For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False

    ## First we randomize a number
    exp_exp_tradeoff = random.uniform(0, 1)
    ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
    if exp_exp_tradeoff > epsilon:
        action = np.argmax(qtable[state, :])
    # Else doing a random choice --> exploration
    else:
        action = env.action_space.sample()

    # for step in range(max_steps):
    while True:
        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        choose_rate = random.uniform(0, 1)
        if choose_rate>0.5:
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
            q1_max_a = np.argmax(qtable1[new_state, :])
            if q1_max_a != 0:
                q1_max_a
                pass
            qtable1[state, action] = qtable1[state, action] + learning_rate * (
                    reward + gamma * (qtable2[new_state, q1_max_a]) - qtable1[state, action])
        else:
            q2_max_a = np.argmax(qtable2[new_state, :])
            if q2_max_a != 0:
                pass
            qtable2[state, action] = qtable2[state, action] + learning_rate * (
                    reward + gamma * (qtable1[new_state, q2_max_a]) - qtable2[state, action])
        qtable = qtable1 + qtable2

        # Our new state is state
        state = new_state
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)
        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])
        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()
        # If done : finish episode
        if done == True:
            break

    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

env.reset()
rewards = []

for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    # print("****************************************************")
    # print("EPISODE ", episode)

    for step in range(max_steps):
        # UNCOMMENT IT IF YOU WANT TO SEE OUR AGENT PLAYING
        # env.render()
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state, :])

        new_state, reward, done, info = env.step(action)

        total_rewards += reward

        if done:
            rewards.append(total_rewards)
            # print ("Score", total_rewards)
            break
        state = new_state
env.close()
print("Score over time: " + str(sum(rewards) / total_test_episodes))