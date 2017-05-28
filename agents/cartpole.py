
# Cartpole implementation.
# Just tries random models, and picks the best.
# By Tom Jacobs

import gym
import numpy as np
import matplotlib.pyplot as plt

def run_episode(env, parameters):
    observation = env.reset()

    # Run 200 steps and see what our total reward is
    total_reward = 0
    for t in range(200):

        # Show us what's going on. Remove this line to run super fast. 
        # The monitor will still render some random ones though for video recording, even if render is off.
#        env.render()

        # Pick action
        action = 0 if np.matmul(parameters, observation) < 0 else 1

        # Step
        observation, reward, done, info = env.step(action)
        total_reward += reward

        # Done?
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    return total_reward

def train(submit):
    env = gym.make('CartPole-v0')
    if submit:
        env = gym.wrappers.Monitor(env, 'cartpole', force=True)

    # Run lots of episodes with random params, and find the best_parameters
    results = []
    counter = 0
    best_parameters = None
    best_reward = 0
    for t in range(100):

        # Pick random parameters and run
        parameters = np.random.rand(4) * 2 - 1
        reward = run_episode(env, parameters)
        results.append(reward)
        counter += 1

        # Did this one do better?
        if reward > best_reward:
            best_reward = reward
            best_parameters = parameters
            print("Better parameters found.")

            # And did we win the world?
            if reward == 200:
                print("Win! Episode {}".format(t))
                break # Can't do better than 200 reward, so quit trying

    # Run 100 runs with the best found params
    print("Found best_parameters, running 100 more episodes with them.")
    for t in range(100):
        reward = run_episode(env, best_parameters)
        results.append(reward)
        print( "Episode " + str(t) )

    return results

# Submit it?
submit = True

# Run
results = train(submit=submit)
if submit:
    # Submit to OpenAI Gym
    print("Uploading to gym")
    gym.scoreboard.api_key = '' # Your key
    gym.upload('cartpole')

else:
    # Graph
    plt.plot(results)
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Rewards over time')
    plt.show()


