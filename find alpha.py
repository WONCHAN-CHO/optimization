# -*- coding: utf-8 -*-
"""
Created on Mon May 27 22:56:16 2024

@author: WONCHAN
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SphereRiemannianEnv:
    def __init__(self, model, loss_fn, train_loader, retraction, zeta=1):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.retraction = retraction
        self.zeta = zeta
        self.state = None
        self.optimal_weights = None
        self.reset()
    
    def reset(self):
        self.model.apply(self._weight_reset)
        self.optimal_weights = [param.clone().to(device) for param in self.model.parameters()]
        self.state = 0
        return self.state
    
    def compute_distance(self, param1, param2):
        return torch.norm(param1 - param2).item()
    
    def compute_cosine_angle(self, param1, param2):
        dot_product = torch.dot(param1.view(-1), param2.view(-1))
        norm_product = torch.norm(param1) * torch.norm(param2)
        return (dot_product / norm_product).item()

    def compute_alpha(self, param, grad, optimal_param, learning_rate):
        new_param = self.retraction(param, -learning_rate * grad)
        
        a = self.compute_distance(new_param, optimal_param)
        b = self.compute_distance(param, new_param)
        c = self.compute_distance(param, optimal_param)
        cos_A = self.compute_cosine_angle(param - new_param, param - optimal_param)
        
        if a**2 <= self.zeta * b**2 + 2 * b * c * cos_A:
            return learning_rate
        else:
            return learning_rate * 0.5  # 조건을 만족하지 않으면 학습률을 줄임

    def step(self, learning_rate):
        total_loss = 0
        self.model.train()
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            
            with torch.no_grad():
                for param, optimal_param in zip(self.model.parameters(), self.optimal_weights):
                    grad = param.grad
                    alpha = self.compute_alpha(param, grad, optimal_param, learning_rate)
                    new_param = self.retraction(param, -alpha * grad)
                    param.copy_(new_param)
            
            total_loss += loss.item()
        
        reward = -total_loss
        self.state = total_loss
        
        return self.state, reward
    
    def _weight_reset(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(28*28, 10)  # MNIST 이미지 크기에 맞춘 모델

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.fc(x)

# Riemannian retraction function for the sphere
def retraction(x, v):
    return (x + v) / torch.norm(x + v)

# 데이터셋 및 데이터 로더 설정
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 모델, 손실 함수 및 데이터 로더 설정
model = MyModel()
loss_fn = nn.CrossEntropyLoss()

# 환경 초기화
env = SphereRiemannianEnv(model, loss_fn, train_loader, retraction, zeta=1)

# 학습 루프
num_episodes = 50
learning_rates = [0.01, 0.05, 0.1, 0.5, 1.0]
results = {}

for lr in learning_rates:
    rewards = []
    env.reset()
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        learning_rate = lr
        while not done:
            next_state, reward = env.step(learning_rate)
            state = next_state
            if state == next_state:  # 손실이 변하지 않으면 종료
                done = True
        rewards.append(-reward)
    results[lr] = rewards
    print(f"Learning Rate {lr}: Final Loss = {-reward}")

# 시각화
plt.figure(figsize=(12, 8))

for lr in learning_rates:
    plt.plot(results[lr], label=f'LR = {lr}')

plt.xlabel('Episode')
plt.ylabel('Total Loss')
plt.title('Total Loss per Episode for Different Learning Rates')
plt.legend()
plt.show()
