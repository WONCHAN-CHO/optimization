# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:39:05 2024

@author: WONCHAN
"""

import numpy as np
import gym
from gym import spaces
from stable_baselines3 import DQN
import matplotlib.pyplot as plt

# 구면체상에 정의된 목표 함수 및 그라디언트
def f(x):
    return 0.5 * (np.linalg.norm(x - np.array([0, 0, 1]))**2)

def grad_f(x):
    return x - np.array([0, 0, 1])

class GradientDescentSphereEnv(gym.Env):
    def __init__(self, f, grad_f, x_init, eta_values):
        super(GradientDescentSphereEnv, self).__init__()
        self.f = f
        self.grad_f = grad_f
        self.x = x_init / np.linalg.norm(x_init)  # 구면체 위로 정규화
        self.eta_values = eta_values
        self.action_space = spaces.Discrete(len(eta_values))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=x_init.shape, dtype=np.float32)
        self.history = []  # 경로를 기록하기 위한 리스트
    
    def reset(self):
        self.x = np.random.randn(*self.x.shape)
        self.x = self.x / np.linalg.norm(self.x)  # 구면체 위로 정규화
        self.history = [self.x.copy()]  # 초기값을 기록
        return self.x

    def step(self, action):
        eta = self.eta_values[action]
        grad = self.grad_f(self.x)
        grad = grad / np.linalg.norm(grad)  # 탄젠트 벡터 정규화
        self.x = self.x - eta * grad
        self.x = self.x / np.linalg.norm(self.x)  # 구면체 위로 정규화
        reward = -self.f(self.x)  # 최소화를 위해 함수 값을 음수로 사용
        done = np.linalg.norm(grad) < 1e-5  # 수렴 기준
        self.history.append(self.x.copy())  # 매 스텝마다 상태를 기록
        return self.x, reward, done, {}

# 초기값 설정
x_init = np.array([1.0, 0.0, 0.0])
eta_values = np.linspace(0.001, 1.0, 10)  # 학습률 값을 이산적인 값들로 정의

# 환경 생성
env = GradientDescentSphereEnv(f, grad_f, x_init, eta_values)

# 강화학습 모델 생성 및 학습
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# 학습된 모델로 학습률 테스트 및 시각화
obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        break

# 최적화 과정 시각화
history = np.array(env.history)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(history[:, 0], history[:, 1], history[:, 2], marker='o')
ax.set_title('Optimization Path on Sphere')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

print("최적화된 파라미터:", obs)
