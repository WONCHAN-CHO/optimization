# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:38:53 2024

@author: WONCHAN
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from stable_baselines3 import PPO

class RiemannianGradientDescentEnv(gym.Env):
    def __init__(self, initial_point, x_star, max_iter=100):
        super(RiemannianGradientDescentEnv, self).__init__()
        self.initial_point = initial_point
        self.x_star = x_star
        self.max_iter = max_iter
        self.current_point = self.initial_point
        self.iteration = 0

        # Action space: stepsize (alpha)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Observation space: current distance to x_star
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

    def reset(self):
        self.current_point = self.initial_point
        self.iteration = 0
        return np.array([self._compute_distance(self.current_point, self.x_star)], dtype=np.float32)

    def step(self, action):
        stepsize = action[0]
        self.current_point = self._update_point(self.current_point, stepsize)
        self.iteration += 1

        done = self.iteration >= self.max_iter or self._compute_distance(self.current_point, self.x_star) < 1e-3
        reward = -self._compute_distance(self.current_point, self.x_star)
        observation = np.array([self._compute_distance(self.current_point, self.x_star)], dtype=np.float32)

        return observation, reward, done, {}

    def _compute_distance(self, x, y):
        # Compute the geodesic distance between x and y in the Riemannian manifold
        return np.linalg.norm(x - y)

    def _update_point(self, x, stepsize):
        # Update the point x using Riemannian gradient descent with the given stepsize
        gradient = self._compute_gradient(x)
        return x - stepsize * gradient

    def _compute_gradient(self, x):
        # Placeholder for the gradient computation in the Riemannian manifold
        return 2 * (x - self.x_star)

# Define initial point and target point
initial_point = np.array([10.0, 10.0])
x_star = np.array([0.0, 0.0])

# Create the environment
env = RiemannianGradientDescentEnv(initial_point, x_star)

# Train the PPO agent
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

# Test the trained agent and visualize the path
obs = env.reset()
path = [env.current_point.copy()]
distances = [env._compute_distance(env.current_point, x_star)]
stepsizes = []

for _ in range(100):
    action, _states = model.predict(obs)
    stepsizes.append(action[0])
    obs, rewards, done, info = env.step(action)
    path.append(env.current_point.copy())
    distances.append(env._compute_distance(env.current_point, x_star))
    if done:
        break

# Calculate the mean stepsize
mean_stepsize = np.mean(stepsizes)
print("Mean stepsize:", mean_stepsize)

# Plot the distance to x_star over time
plt.figure()
plt.plot(distances)
plt.xlabel('Iteration')
plt.ylabel('Distance to x*')
plt.title('Distance to x* over iterations')
plt.grid()
plt.show()

# Plot the path taken by the agent
path = np.array(path)
plt.figure()
plt.plot(path[:, 0], path[:, 1], marker='o')
plt.scatter([initial_point[0]], [initial_point[1]], color='red', label='Initial Point')
plt.scatter([x_star[0]], [x_star[1]], color='green', label='Target Point')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Path taken by the agent')
plt.legend()
plt.grid()
plt.show()

