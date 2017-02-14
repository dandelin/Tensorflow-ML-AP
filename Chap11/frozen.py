import gym
from gym.envs.registration import register
from msvcrt import getch
import numpy as np
from matplotlib import pyplot as plt

# arrow_keys = {
#     72: 3, # UP
#     80: 1, # DOWN
#     75: 0, # LEFT
#     77: 2 # RIGHT
# }

# register(
#     id = 'FrozenLake-v3',
#     entry_point='gym.envs.toy_text:FrozenLakeEnv',
#     kwargs={'map_name': '4x4', 'is_slippery': True}
# )

env = gym.make('FrozenLake-v0')
env.render()

Q = np.zeros([env.observation_space.n, env.action_space.n])

lr, dis, num_episodes = 0.85, 0.99, 2000

rList = []

for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))
        new_state, reward, done, info = env.step(action)
        Q[state, action] = (1 - lr) * Q[state, action] + lr * (reward + dis * np.max(Q[new_state, :]))
        state = new_state
        rAll += reward
    
    rList.append(rAll)

print('score over time: ' + str(sum(rList)/num_episodes))
print('final Q-table values')
print(Q)

plt.bar(range(len(rList)), rList, color='blue')
plt.show()


# while True:
#     key = ord(getch())
#     if key == 224:
#         key = ord(getch())
#         action = arrow_keys[key]
#         state, reward, done, info = env.step(action)
#         env.render()
#         print('State: ', state, 'Action: ', action, 'Reward: ', reward, 'Info: ', info)

#         if done:
#             print('Finished with reward: ', reward)
#             break
#     else:
#         print('Game aborted!')
#         break
