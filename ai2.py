# -*- coding: utf-8 -*-
# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from PIL import Image





class Actor(nn.Module):
    def __init__ (self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (1, 42, 42)
                nn.Conv2d(1, 16, kernel_size=(3,3), padding = 0),
                nn.BatchNorm2d(16),
                #nn.Dropout(p=0.2),
                #nn.LeakyReLU(0.1),
                nn.ReLU(),  # activation
                nn.Conv2d(16, 32, kernel_size=(3,3), padding = 0),
                nn.BatchNorm2d(32),  
                #nn.Dropout(p=0.2),,
                #nn.LeakyReLU(0.1),
                nn.ReLU(),  # activation
                nn.Conv2d(32, 64, kernel_size=(3,3), padding = 0),
                nn.BatchNorm2d(64), 
                #nn.LeakyReLU(0.1),
                #nn.Dropout(p=0.2),
                nn.ReLU(),

                nn.MaxPool2d(2, 2),

                nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
                nn.ReLU(),

                nn.Conv2d(16, 32, kernel_size=(3,3), padding = 0),
                nn.BatchNorm2d(32),  
                #nn.Dropout(p=0.2),
                #nn.LeakyReLU(0.1),
                nn.ReLU(),  # activation
                nn.Conv2d(32, 64, kernel_size=(3,3), padding = 0),
                nn.BatchNorm2d(64),  
                #nn.Dropout(p=0.2),
                nn.ReLU(),  # activation
                #nn.LeakyReLU(0.1),

                nn.MaxPool2d(2, 2),

                nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
                nn.ReLU(),

                nn.Conv2d(16, 32, kernel_size=(3,3), padding = 0),
                nn.BatchNorm2d(32), 
                #nn.Dropout(p=0.2),
                nn.ReLU(),


                #nn.LeakyReLU(0.1),

                #nn.Conv2d(16, 32, kernel_size=(3,3), padding = 0), 
                

                nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 1), padding=0),
                nn.AvgPool2d(5)                        
        )

        
        # Defining the first Critic neural network
        self.layer_1 = nn.Sequential(nn.Linear(8+2, 200))
        self.layer_2 = nn.Sequential(nn.Linear(200, 100))
        self.layer_3 = nn.Sequential(nn.Linear(100, 1) ) 
        self.max_action = max_action
        self.action_dim = action_dim
      
        #self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            #nn.init.constant_(m.bias, 0.1)

    def forward(self, x, orientation, n_orientation):

        #print("===================Orientation============")
        #print(orientation)
        #print(n_orientation)
        x = self.cnn_base(x)
        
        x = x.view(-1, 8)

        xu1 = torch.cat([x, orientation,n_orientation], 1)
        
        x1 = F.relu(self.layer_1(xu1))
        x1 = F.relu(self.layer_2(x1))
        out = self.layer_3(x1)

        print("Out==>>>>>>>>>>>>>>>>>>>>>>>>>>>>{}".format(out))
        print(torch.tanh(out))
        print("Multiply========================{}".format(self.max_action * torch.tanh(out)))
        
        return self.max_action * torch.tanh(out)

class Critic(nn.Module):
    def __init__(self, state_dim,action_dim):
        super(Critic, self).__init__()
    # Defining the first Critic neural network
        self.cnn_base = nn.Sequential(  # input shape (1, 42, 42)
             nn.Conv2d(1, 16, kernel_size=(3,3), padding = 0),
                nn.BatchNorm2d(16),
                #nn.Dropout(p=0.2),
                #nn.LeakyReLU(0.1),
                nn.ReLU(),  # activation
                nn.Conv2d(16, 32, kernel_size=(3,3), padding = 0),
                nn.BatchNorm2d(32),  
                #nn.Dropout(p=0.2),,
                #nn.LeakyReLU(0.1),
                nn.ReLU(),  # activation
                nn.Conv2d(32, 64, kernel_size=(3,3), padding = 0),
                nn.BatchNorm2d(64), 
                #nn.LeakyReLU(0.1),
                #nn.Dropout(p=0.2),
                nn.ReLU(),

                nn.MaxPool2d(2, 2),
                nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
                nn.ReLU(),

                nn.Conv2d(16, 32, kernel_size=(3,3), padding = 0),
                nn.BatchNorm2d(32),  
                #nn.Dropout(p=0.2),
                #nn.LeakyReLU(0.1),
                nn.ReLU(),  # activation
                nn.Conv2d(32, 64, kernel_size=(3,3), padding = 0),
                nn.BatchNorm2d(64),  
                #nn.Dropout(p=0.2),
                nn.ReLU(),  # activation
                #nn.LeakyReLU(0.1),

                nn.MaxPool2d(2, 2),
                nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
                nn.ReLU(),

                nn.Conv2d(16, 32, kernel_size=(3,3), padding = 0),
                nn.BatchNorm2d(32), 
                #nn.Dropout(p=0.2),
                nn.ReLU(),


                #nn.LeakyReLU(0.1),

                #nn.Conv2d(16, 32, kernel_size=(3,3), padding = 0), 
                

                nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 1), padding=0),
                nn.AvgPool2d(5)                    
            
        )

        self.cnn_base2 = nn.Sequential(  # input shape (1, 42, 42)
              nn.Conv2d(1, 16, kernel_size=(3,3), padding = 0),
                nn.BatchNorm2d(16),
                #nn.Dropout(p=0.2),
                #nn.LeakyReLU(0.1),
                nn.ReLU(),  # activation
                nn.Conv2d(16, 32, kernel_size=(3,3), padding = 0),
                nn.BatchNorm2d(32),  
                #nn.Dropout(p=0.2),,
                #nn.LeakyReLU(0.1),
                nn.ReLU(),  # activation
                nn.Conv2d(32, 64, kernel_size=(3,3), padding = 0),
                nn.BatchNorm2d(64), 
                #nn.LeakyReLU(0.1),
                #nn.Dropout(p=0.2),
                nn.ReLU(),

                nn.MaxPool2d(2, 2),
                nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
                nn.ReLU(),

                nn.Conv2d(16, 32, kernel_size=(3,3), padding = 0),
                nn.BatchNorm2d(32),  
                #nn.Dropout(p=0.2),
                #nn.LeakyReLU(0.1),
                nn.ReLU(),  # activation
                nn.Conv2d(32, 64, kernel_size=(3,3), padding = 0),
                nn.BatchNorm2d(64),  
                #nn.Dropout(p=0.2),
                nn.ReLU(),  # activation
                #nn.LeakyReLU(0.1),

                nn.MaxPool2d(2, 2),
                nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
                nn.ReLU(),

                nn.Conv2d(16, 32, kernel_size=(3,3), padding = 0),
                nn.BatchNorm2d(32), 
                #nn.Dropout(p=0.2),
                nn.ReLU(),


                #nn.LeakyReLU(0.1),

                #nn.Conv2d(16, 32, kernel_size=(3,3), padding = 0), 
                

                nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 1), padding=0),
                nn.AvgPool2d(5)                           
            
        )
        #self.apply(self._weights_init)


        
        self.action_dim = action_dim
        # Defining the first Critic neural network
        self.layer_1 = nn.Linear(8 + self.action_dim +2, 200)
        self.layer_2 = nn.Linear(200, 100)
        self.layer_3 = nn.Linear(100, 1)

        # Defining the Second Critic neural network
        self.layer_4 = nn.Linear(8 + self.action_dim+2, 200)
        self.layer_5 = nn.Linear(200, 100)
        self.layer_6 = nn.Linear(100, 1)
 
    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            #nn.init.constant_(m.bias, 0.1)

    def forward(self, x, u, orientation, n_orientation):


        #print("===================Orientation2============")
        #print(orientation)
        #print(n_orientation)
        xc1 = self.cnn_base(x)
        xc1 = xc1.view(-1, 8)

        xc2 = self.cnn_base2(x)
        xc2 = xc2.view(-1, 8)


        xu1 = torch.cat([xc1, u, orientation, n_orientation], 1)

        xu2 = torch.cat([xc2, u, orientation, n_orientation], 1)

        # Forward-Propagation on the first Critic Neural Network
        x1 = F.relu(self.layer_1(xu1))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        # Forward-Propagation on the second Critic Neural Network
        x2 = F.relu(self.layer_4(xu2))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)

        return x1,x2

    def Q1(self, x, u, orientation, n_orientation):
        #print("===================Orientation3============")
        #print(orientation)
        #print(n_orientation)
        xc1 = self.cnn_base(x)
        xc1 = xc1.view(-1, 8)
        xu1 = torch.cat([xc1, u, orientation, n_orientation], 1)
        x1 = F.relu(self.layer_1(xu1))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1
    
# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0
    self.i =0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
       self.i = self.i+1
       self.storage.append(transition)

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones, batch_ct_orientation, n_batch_ct_orientation , \
    batch_nxt_orientation, n_batch_nxt_orientation = [], [], [], [], [], [], [], [], []
    for i in ind: 
      state, next_state, action, reward, done ,current_orientation, n_current_orientation, next_orientation, n_next_orientation= self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
      batch_ct_orientation.append(np.array(current_orientation, copy=False))
      n_batch_ct_orientation.append(np.array(n_current_orientation, copy=False))
      batch_nxt_orientation.append(np.array(next_orientation, copy=False))
      n_batch_nxt_orientation.append(np.array(n_next_orientation, copy=False))

    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions).reshape(-1, 1), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1), \
    np.array(batch_ct_orientation).reshape(-1, 1), np.array(n_batch_ct_orientation).reshape(-1, 1), \
    np.array(batch_nxt_orientation).reshape(-1, 1), np.array(n_batch_nxt_orientation).reshape(-1, 1)





# Building the whole Training Process into a class

class TD3(object):
  
  def __init__(self, state_dim, action_dim, max_action):
    #print("=========================={},{},{}".format(state_dim,action_dim,max_action))
    self.cnt =0
    self.actor = Actor(state_dim, action_dim, max_action)
    self.actor_target = Actor(state_dim, action_dim, max_action)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0004,weight_decay = 0.0001)
    self.critic = Critic(state_dim, action_dim)
    self.critic_target = Critic(state_dim, action_dim)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.0001, weight_decay = 0.0001)
    self.max_action = max_action

  def select_action(self, state, orientation, n_orientation):
    #result = state[0, :, :]
    #print(result)
    #im = Image.fromarray(result)
    #self.cnt =self.cnt+1
    #im.save("./dump/test{}.png".format(self.cnt))

    orientation = np.array(orientation, dtype='float32').reshape(1)
    n_orientation =np.array(n_orientation, dtype='float32').reshape(1)
    print(orientation.size)
    print(n_orientation.size)

    orientation = torch.from_numpy(orientation).float().cpu().unsqueeze(0)
    n_orientation = torch.from_numpy(n_orientation).float().cpu().unsqueeze(0)

    print("Shape===========================")

    print(orientation.shape)
    print(n_orientation.shape)

    with torch.no_grad():
      state = torch.from_numpy(state).float().cpu().unsqueeze(0)
      print("Select Action state {}==========".format(state.shape))
      self.actor.eval()
      print("Trainin Mode============={}".format(self.actor.training))
      action = self.actor(state, orientation, n_orientation)
      self.actor.train()

    return action.cpu().data.numpy().flatten()

    #Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
  def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    #iterations = 50
    print("Iterations==============={}".format(iterations))
    #iterations = 80
    
    for it in range(iterations):
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones, batch_ct_orientation, n_batch_ct_orientation, batch_nxt_orientation, n_batch_nxt_orientation, \
      = replay_buffer.sample(batch_size)
      #print("Batch rewards============={}".format(batch_rewards))

      state = torch.Tensor(batch_states)   
      next_state = torch.Tensor(batch_next_states)
      action = torch.Tensor(batch_actions)
      reward = torch.Tensor(batch_rewards)
      done = torch.Tensor(batch_dones)
      current_orientation = torch.Tensor(batch_ct_orientation)
      n_current_orientation = torch.Tensor(n_batch_ct_orientation)
      next_orientation = torch.Tensor(batch_nxt_orientation)
      n_next_orientation = torch.Tensor(n_batch_nxt_orientation)

      # Step 5: From the next state s’, the Actor target plays the next action a’
      next_action = self.actor_target(next_state, next_orientation, n_next_orientation)
     
      #Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise)
      noise = noise.clamp(-noise_clip, noise_clip)
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
      
      # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      target_Q1, target_Q2 = self.critic_target(next_state, next_action, next_orientation, n_next_orientation)
      
      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      
      # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(state, action, current_orientation, n_current_orientation)
      
      # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      
      # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(state, self.actor(state, current_orientation, n_current_orientation), current_orientation, n_current_orientation).mean()
        print("Actor Loss==============={}".format(actor_loss))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
  
  # Making a save method to save a trained model
  def save(self, filename, directory):
    print("Save======================={},{}".format(filename, directory))
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))




