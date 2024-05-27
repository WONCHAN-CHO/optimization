# -*- coding: utf-8 -*-
"""
Created on Mon May 27 22:06:36 2024

@author: WONCHAN
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class RiemannianOptimizationEnv:
    def __init__(self, model, loss_fn, train_loader):
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.state = None
        self.reset()
    
    def reset(self):
        self.model.apply(self._weight_reset)
        self.state = 0
        return self.state
    
    def step(self, learning_rate):
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        
        total_loss = 0
        self.model.train()
        for inputs, targets in self.train_loader:
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        reward = -total_loss
        self.state = total_loss
        
        return self.state, reward
    
    def _weight_reset(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

class QLearningAgent:
    def __init__(self, action_space, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.action_space = action_space
        self.q_table = np.zeros(action_space)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.q_table)
    
    def learn(self, state, action, reward, next_state):
        predict = self.q_table[action]
        target = reward + self.gamma * np.max(self.q_table)
        self.q_table[action] = self.q_table[action] + self.alpha * (target - predict)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(28*28, 10)  # MNIST 이미지 크기에 맞춘 모델

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.fc(x)

# 데이터셋 및 데이터 로더 설정
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 학습률 액션 공간 정의
learning_rates = [0.001, 0.01, 0.1, 0.2, 0.5]
action_space = len(learning_rates)

# 모델, 손실 함수 및 데이터 로더 설정
model = MyModel()
loss_fn = nn.CrossEntropyLoss()

# 환경 및 에이전트 초기화
env = RiemannianOptimizationEnv(model, loss_fn, train_loader)
agent = QLearningAgent(action_space)

# 학습 루프
num_episodes = 100
rewards = []
learning_rate_history = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        learning_rate = learning_rates[action]
        next_state, reward = env.step(learning_rate)
        agent.learn(state, action, reward, next_state)
        state = next_state
        if state == next_state:  # 손실이 변하지 않으면 종료
            done = True
    
    rewards.append(-reward)
    learning_rate_history.append(learning_rate)
    print(f"Episode {episode + 1}: Total Loss = {-reward}, Learning Rate = {learning_rate}")

# 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Loss')
plt.title('Total Loss per Episode')

plt.subplot(1, 2, 2)
plt.plot(learning_rate_history)
plt.xlabel('Episode')
plt.ylabel('Learning Rate')
plt.title('Learning Rate per Episode')

plt.tight_layout()
plt.show()

