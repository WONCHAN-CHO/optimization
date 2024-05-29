# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:48:02 2024

@author: WONCHAN
"""

import numpy as np
import gym
from gym import spaces
from stable_baselines3 import DQN
import shimmy
import matplotlib.pyplot as plt

class GradientDescentEnv(gym.Env):
    def __init__(self, f, grad_f, x_init, alpha, beta, eta_values):
        super(GradientDescentEnv, self).__init__()
        self.f = f
        self.grad_f = grad_f
        self.x = x_init
        self.alpha = alpha
        self.beta = beta
        self.eta_values = eta_values
        self.action_space = spaces.Discrete(len(eta_values))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=x_init.shape, dtype=np.float32)
        self.history = []  # 경로를 기록하기 위한 리스트
    
    def reset(self):
        self.x = np.random.randn(*self.x.shape)
        self.history = [self.x.copy()]  # 초기값을 기록
        return self.x

    def step(self, action):
        eta = self.eta_values[action]
        grad = self.grad_f(self.x)
        self.x = self.x - eta * grad
        reward = -self.f(self.x)  # 최소화를 위해 함수 값을 음수로 사용
        done = np.linalg.norm(grad) < 1e-5  # 수렴 기준
        self.history.append(self.x.copy())  # 매 스텝마다 상태를 기록
        return self.x, reward, done, {}

# 예제 함수 및 그레이디언트 정의
def f(x):
    return 0.5 * np.dot(x, x)

def grad_f(x):
    return x

# 초기값 설정
x_init = np.array([10.0, 10.0])
alpha = 1.0  # 강한 볼록성 상수
beta = 1.0  # 스무스 상수
eta_values = np.linspace(0.001, 1.0, 10)  # 학습률 값을 이산적인 값들로 정의

# 환경 생성
env = GradientDescentEnv(f, grad_f, x_init, alpha, beta, eta_values)

# 강화학습 모델 생성 및 학습
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# 학습된 모델로 학습률 테스트
obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        break

# 최적화 과정 시각화
history = np.array(env.history)

plt.figure()
plt.plot(history[:, 0], history[:, 1], marker='o')
plt.title('Optimization Path')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.show()

print("최적화된 파라미터:", obs)