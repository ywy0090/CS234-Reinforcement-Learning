'''
code from https://github.com/simoninithomas/Deep_reinforcement_learning_Course/
modify by ywy
'''
import numpy as np
import gym
import random

env = gym.make("FrozenLake-v0")
action_size = env.action_space.n
state_size = env.observation_space.n

qtable = np.zeros((state_size, action_size))

qtable1 = np.zeros((state_size, action_size))
qtable2 = np.zeros((state_size, action_size))
print(qtable)
# hyperparameters
total_episodes = 55000        # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 99                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 0.05                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.005             # Exponential decay rate for exploration prob

#
# List of rewards
rewards = []
q1_update_cnt = 0
q2_update_cnt = 0
# 2 For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    action = np.argmax(qtable[state, :])
    while  True:
        # 3. Choose an action a in the current world state (s)
        exp_exp_tradeoff = random.uniform(0, 1)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])
        else:
            action = env.action_space.sample()
        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)
        choose_rate = random.uniform(0, 1)
        # if reward > 0.000001:
            # print(reward)
        if choose_rate>0.5:
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
            max_a = np.argmax(qtable1[new_state, :])
            qtable1[state, action] = qtable1[state, action] + learning_rate * (
                    reward + gamma * (qtable2[new_state, max_a]) - qtable1[state, action])
            q1_update_cnt += 1
        else:
            max_a = np.argmax(qtable2[new_state, :])
            qtable2[state, action] = qtable2[state, action] + learning_rate * (
                    reward + gamma * (qtable1[new_state, max_a]) - qtable2[state, action])
            q2_update_cnt += 1
        qtable += qtable1 + qtable2
        total_rewards += reward
        # Our new state is state
        state = new_state
        # If done (if we're dead) : finish episode
        if done == True:
            break

    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)
print("Score over time: " + str(sum(rewards) / total_episodes))
print("q1 update cnt:{}, q2 update cnt:{}".format(q1_update_cnt, q2_update_cnt))
print(qtable)

env.reset()

for episode in range(5):
    state = env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):

        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state, :])

        new_state, reward, done, info = env.step(action)

        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
            env.render()

            # We print the number of step it took.
            print("Number of steps", step)
            break
        state = new_state
env.close()