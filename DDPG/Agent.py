import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from DDPG.Actor import Actor
from DDPG.Critic import Critic

class DDPGAgent:
    def __init__(self, state_dim, action_dim, device, lr=1e-3, gamma=0.99, tau=0.005):
        self.device = device
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.target_actor = Actor(state_dim, action_dim).to(self.device)
        self.target_critic = Critic(state_dim, action_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Create replay buffer
        self.replay_buffer = []
        self.gamma = gamma
        self.tau = tau

    def act(self, state):
        # move state to gpu
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = self.actor(state).cpu().detach().numpy()  # Move results back to CPU for use by AirSim
        return action

    def store(self, state, action, reward, next_state):
        # save experience
        self.replay_buffer.append((state, action, reward, next_state))

    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        
        # 從重放緩衝區中隨機抽取一個批次的經驗
        batch = np.random.choice(self.replay_buffer, batch_size, replace=False)
        states, actions, rewards, next_states = zip(*batch)

        # 將數據轉換為張量並移到GPU
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)

        # update Critic
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_q_values = self.target_critic(next_states, target_actions)
            target_q_values = rewards + self.gamma * target_q_values

        current_q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target net
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)

    def soft_update(self, target_net, source_net):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)
