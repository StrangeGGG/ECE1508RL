# ECE1508RL - Using RL for Traffic Signal Control

# 1. Introduction
The problem of traffic congestion has been a major challenge for many cities in modern days. It results in longer travel time, more fuel consumption, and more CO2 emissions. In major cities, intersections signalized by traffic lights often cause long queues and high waiting times as a result of inefficient traffic light schedules, especially in dynamic and unpredictable traffic conditions. Traditional control strategies are usually static and unable to adapt to the real-time situation of the traffic, causing non-optimal performance when the traffic volumes change during the day.

Reinforcement Learning (RL) provides a data-driven framework to develop adaptive traffic signal control systems that learn and evolve optimal signal policies through interactions with the varying environment. By formulating signal control algorithm as a sequential decision-making problem, the RL agent can use current traffic conditions to predict future congestions, then choose signal strategies that minimize waiting time and reduce congestions. Compared to static signal systems, RL has the potential to handle the traffic conditions dynamically and more effectively.

This project uses Deep Reinforcement Learning to solve the traffic signal control system problem with a lightweight simulation environment. The goal is to demonstrate how RL can help improve the traffic management performance while indicating the trade-offs between model complexity, generalization, and applicability in real world.

# 2. Baseline & RL Models

## 2.1 Baseline Models
### Fixed-Timing
We first estimate the traffic flow and predefined each traffic light duration time. Each current phrase (traffic light) will change to the next current phrase at a constant time. The transfer of the current phrase repeats in a cycle, regardless how the environment has changed.
<img width="812" height="182" alt="FixedTimeExample" src="https://github.com/user-attachments/assets/ff451cce-ed88-4619-b2f4-541efac60be1" />

### Round-Robin
We implement a simple round-robin controller as a non-adaptive baseline. The intersection alternates between the East–West and North–South phases in a fixed cyclic order, assigning equal green durations to each phase. This ensures fairness between the two directions but does not respond to varying traffic densities.
<img width="791" height="184" alt="RoundRobinExample" src="https://github.com/user-attachments/assets/544b5671-2c9a-4edb-8836-b7fa91922160" />

## 2.2 RL Models
### Deep-Q Network (DQN)
DQN is a popular off-policy reinforcement learning algorithm which estimates the action-value function (q(s,a)) by neural network and chooses subsequent actions that maximize the return. DQN has two core mechanisms - experience replay, which breaks the potential correlation between samplings by random selection, and target network, which utilizes two different neural networks to prevent overfitting and gradual parameter updates.The architecture of Double DQN and Dueling head will ensure algorithm stability as well. This algorithm is ideal for our traffic signal control problem due to its nature of effectiveness for discrete action choice problems. 
<img width="752" height="333" alt="DQN" src="https://github.com/user-attachments/assets/dc839f10-0c20-4455-82fd-a2024fe3a938" />

### Proximal Policy Optimization (PPO)
PPO is a popular on-policy reinforcement learning algorithm which optimizes the policy by gradient estimates of value function. PPO has a core mechanism of clipping, which limits the step size in policy iteration. By constraining the policy change ratio in each update, PPO ensures steady policy improvement. In addition, Generalized Advantage Estimation (GAE) will also be implemented to calculate the reward. This algorithm is ideal for our traffic signal control problem due to its capability of hybrid/ continuous control and stable delivery of learning curve.

# 3. Usage Example
### Fixed-Timing

For fixed-timing simulation, run `self_designed_env/simulation.py`. <br> You can switch between ideal/realistic environment by choosing simulation function in `run_simulation()`.

### Round-Robin
### DQN
For DQN simulation, run `DQN/train.py`. <br> You can switch between ideal/realistic environment by choosing simulation function in `train()`. <br> You may also choose different reward function in `TrafficEngine_realistic._compute_reward()` in `DQN/traffic_env.py`
### PPO
