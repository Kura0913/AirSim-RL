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
            return None, None  # Return None if there's not enough data in the replay buffer

        # Randomly draw a batch of experiences from the replay buffer
        batch = np.random.choice(self.replay_buffer, batch_size, replace=False)
        states, actions, rewards, next_states = zip(*batch)

        # Convert data to tensors and move to GPU
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)

        # Update Critic
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_q_values = self.target_critic(next_states, target_actions)
            target_q_values = rewards + self.gamma * target_q_values

        current_q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target network
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)

        # Return the losses for logging and visualization
        return critic_loss.item(), actor_loss.item()

    def soft_update(self, target_net, source_net):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)

    def save(self, path):
        """Save the model weights and optimizer state to the specified path."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
        print(f"Model and optimizer states saved to {path}")

    def load(self, path):
        """Load the model weights and optimizer state from the specified path."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load the model weights
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # Load the optimizer state
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        print(f"Model and optimizer states loaded from {path}")

