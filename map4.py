# Self Driving Car

# Importing the libraries
import os
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

import random as rn
# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture



from ai2 import Actor
from ai2 import Critic
from ai2 import ReplayBuffer
from ai2 import TD3
import torch
from PIL import Image


# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

#points  = [(212,111),(436,312),(799,356),(1116,212),(1019,399),(1056,481),(1179,534),(1404,526),(508,612),(865,375),(948,84),(583,201),(711,218),(883,478),(936,180),(531,290),(1120,568),(1327,446)]
points = [(120,266),(349,300),(492,510),(721,415),(1043,407),(1162,290),(598,93),(1011,135),(153,133),(226,542),(515,588)]

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function

action2rotation = [0,5,-5]
last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")

# textureMask = CoreImage(source="./kivytest/simplemask1.png")
file_name = "%s_%s" % ("TD3", "auto_car")
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")

save_models = True # Boolean checker whether or not to save the pre-trained model

if not os.path.exists("./results"):
  os.makedirs("./results")
if save_models and not os.path.exists("./pytorch_models"):
  os.makedirs("./pytorch_models")

if not os.path.exists("./dump"):
  os.makedirs("./dump")
# Initializing the last distance

arrow = Image.open("./images/arrow4.png")

# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    
    
    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation




class Game(Widget):
 

    car = ObjectProperty(None)

    
    max_timesteps = 500000
    start_timesteps = 2500 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
    eval_freq = 1000 # How often the evaluation step is performed (after how many timesteps)
    max_timesteps = 5e5 # Total number of iterations/timesteps
    
    expl_noise = 2# Exploration noise - STD value of exploration Gaussian noise
    batch_size = 100 # Size of the batch
    discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
    tau = 0.005 # Target network update rate
    policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
    noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
    policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    episode_reward = 0
    episode_timesteps = 0

    replay_buffer = ReplayBuffer()
    """
    Environment wrapper for CarRacing 
    """
    
    max_action = 5
    random_action = np.arange(-max_action, max_action, 0.5).tolist()
    
    longueur = 1464
    largeur =  800
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    img = np.asarray(img)
    sand = img/255
    goal_x = 1385
    goal_y = 668
    first_update = True
    swap = 0
    

    reward = 0
    observation_space = (1,42,42)
    action_space = 1
    length = 200
    history = np.zeros(length)
    count = 0
    done =True
    state_dim = observation_space
    action_dim = action_space 
    last_distance = 0
    policy = TD3(state_dim, action_dim, max_action)

    unwanted_state = False
    num  =0
    obs = np.empty((0))
    current_orientation = 0.
    next_orientation =0. 

    def reward_memory(self,rwd):
        # record reward for last 100 steps

        self.history[self.count] = rwd
        #print(self.history)
        self.count = (self.count + 1) % self.length
        #print("==================Cont{} , reward : {}, mean : {}".format(self.count,rwd,np.mean(self.history)))
        return np.mean(self.history)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)    

    def reset(self):
        
        print("Reset===========")
        self.history = np.zeros(self.length)
        #self.car.x = rn.randrange(100, 1400)
        #self.car.y = rn.randrange(100, 620)

        
        self.car.x,self.car.y = rn.choice(points)

        xx = self.goal_x - self.car.x
        yy = self.goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.


        img_patch = self.img[int(self.car.x)-21:int(self.car.x)+21, int(self.car.y)-21:int(self.car.y)+21]

        background = Image.fromarray(img_patch)


        background.save("./dump/reset.png")


        background = Image.open("./dump/reset.png")

        cimg = Image.open("./gray1.png")
        #print("============Angle{}".format(self.car.angle))
        cimg = cimg.rotate(self.car.angle, expand=True)
        cimg.save("./dump/car23.png")
        cimg = Image.open("./dump/car23.png")

        background.paste(cimg,(11,8),cimg)
        background.save("./dump/reset.png")

        new = PILImage.open("./dump/reset.png")
        new = np.asarray(new)
        new = new/128.-1.

        img_patch = np.expand_dims(new, axis=0)
        
        return img_patch,orientation
        #return np.array(img_patch)

    def step(self, action):

       
        print("step===========")
        print(type(action))

        step_done = False    
        
        self.car.move(action)

        xx = self.goal_x - self.car.x
        yy = self.goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.

        distance = np.sqrt((self.car.x - self.goal_x)**2 + (self.car.y - self.goal_y)**2)

        if self.sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            print(1, self.goal_x, self.goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            last_reward = -2
            #if distance < self.last_distance:
                #last_reward = -1
        else: # otherwise
            self.car.velocity = Vector(3, 0).rotate(self.car.angle)
            last_reward = 0.5
            print(0, self.goal_x, self.goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            if distance < self.last_distance:
                last_reward = 3

        img_patch = self.img[int(self.car.x)-21:int(self.car.x)+21, int(self.car.y)-21:int(self.car.y)+21]

       #img_patch = Image.fromarray(img_patch)

        #big = self.img[int(self.car.x)-50:int(self.car.x)+50, int(self.car.y)-50:int(self.car.y)+50]



        background = Image.fromarray(img_patch)



        #t.paste(arrow,(int(self.car.x),int(self.car.y),arrow)

        background.save("./dump/patch.png")


        background = Image.open("./dump/patch.png")

        cimg = Image.open("./gray1.png")
        #print("============Angle{}".format(self.car.angle))
        cimg = cimg.rotate(self.car.angle, expand=True)
        cimg.save("./dump/car23.png")
        cimg = Image.open("./dump/car23.png")

        background.paste(cimg,(11,8),cimg)
        background.save("./dump/patch.png")

        new = PILImage.open("./dump/patch.png")
        
        #new.save("xyz{}.png".format(self.num))
        new = np.asarray(new)
        #print(new)
        new = new/128. -1.

        #print(new)
        #self.num= self.num+1
        #x = new*255.0
        #x =Image.fromarray(x)
        #x.save('abc{}.png'.format(self.num))


        img_patch = np.expand_dims(new, axis=0)
        #print(type(img_patch))

        

        if self.car.x < 25:
            self.car.x = 25
            last_reward = -5
            self.unwanted_state = True

        if self.car.x > self.width - 25:
            self.car.x = self.width - 25
            last_reward = -10
            self.unwanted_state = True

        if self.car.y < 25:
            self.car.y = 25
            last_reward = -5
            self.unwanted_state = True

        if self.car.y > self.height - 25:
            self.car.y = self.height - 25
            last_reward = -5
            self.unwanted_state = True

        temp = self.reward_memory(last_reward)
        
        if temp <-0.5:       
            step_done = True 
        else:            
            step_done= False

        if distance < 25:
            if self.swap == 1:
                self.goal_x = 1420
                self.goal_y = 622
                self.swap = 2

            else:
                self.goal_x = 9
                self.goal_y = 85
                self.swap = 1
            last_reward = 50
            step_done = True   

        self.last_distance = distance
        if img_patch.shape != (1, 42, 42):
            self.unwanted_state = True
            


        if self.unwanted_state:
            print ("Not equal============================================")
            new = PILImage.open("./default.png".format(action)).convert('L')
            new = np.asarray(new)
            new = new/255
            img_patch = np.expand_dims(new, axis=0)

        print(img_patch.shape)
        print("Laast Reward==============={}".format(last_reward))
        return img_patch, last_reward, step_done, orientation

        
        
    

    def evaluate_policy(self,policy):
        avg_reward = 0.0
        eval_episodes=10

        for _ in range(eval_episodes):
            print("Evaluate Policy--------------------------------------------")
            obs = self.reset()
            done = False
            while not done:
                action = self.policy.select_action(obs)
                print(type(action))
                action = action.item()
                obs, reward, done = self.step(action)
                avg_reward += reward
        avg_reward /= eval_episodes
        print ("---------------------------------------")
        print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
        print ("---------------------------------------")
        return avg_reward

    def update(self, dt):
        

        
        evaluations = []
        cnt = 0

        if self.total_timesteps < self.max_timesteps:
        # If the episode is done

            
            # If the episode is done
            if self.done:
            # If we are not at the very beginning, we start the training process of the model
                
                if self.total_timesteps != 0:
                    print("Total Timesteps: {} Episode Num: {} Reward: {}".format(self.total_timesteps, self.episode_num, self.episode_reward))
                    self.policy.train(self.replay_buffer, self.episode_timesteps, self.batch_size, self.discount, self.tau, self.policy_noise, self.noise_clip, self.policy_freq)

                    # We evaluate the episode and we save the policy
                    #if self.timesteps_since_eval >= self.eval_freq:
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    #self.timesteps_since_eval %= self.eval_freq
                    #evaluations.append(self.evaluate_policy(self.policy))
                    self.policy.save(file_name, directory="./pytorch_models")
                    #np.save("./results/%s" % (file_name), evaluations)
    
                    # When the training step is done, we reset the state of the environment
                    self.obs,self.current_orientation = self.reset()
                    
                    #break
                    # Set the Done to False
                    self.done = False
    
                    # Set rewards and episode timesteps to zero
                    self.episode_reward = 0
                    self.episode_timesteps = 0
                    self.episode_num += 1
  
            # Before 10000 timesteps, we play random actions
            if self.total_timesteps < self.start_timesteps:
                action = rn.choice(self.random_action)
                
            else: # After 10000 timesteps, we switch to the model                      
                action = self.policy.select_action(self.obs, self.current_orientation, -self.current_orientation)
                # If the explore_noise parameter is not 0, we add noise to the action and we clip it
                if self.expl_noise != 0:
                    action = (action + np.random.normal(0, self.expl_noise, size=self.action_space)).clip(-self.max_action, self.max_action)
                    action = action[0]
            #break
            if self.first_update:

                self.obs,self.current_orientation = self.reset()
                self.first_update = False
            # The agent performs the action in the environment, then reaches the next state and receives the reward

            new_obs, reward, self.done, self.next_orientation = self.step(action) 
            
            test_action = self.policy.select_action(new_obs, self.next_orientation, -self.next_orientation)
            print("Test Action++++++++++++++++{}".format(test_action))
            #print("new_obs: {} Reward: {} Done: {} info : {}".format(new_obs, reward, done, _))
  
            # We check if the episode is done
            #done_bool = 0 if episode_timesteps + 1 == env.max_episode_steps else float(done)
  
            # We increase the total reward
            self.episode_reward +=reward

            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            self.replay_buffer.add((self.obs, new_obs, action, reward, float(self.done), self.current_orientation, -self.current_orientation, self.next_orientation, -self.next_orientation))

            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            self.obs = new_obs
            self.current_orientation = self.next_orientation

            self.episode_timesteps += 1
            self.total_timesteps += 1
            self.timesteps_since_eval += 1


        



    

class CarApp(App):

    def build(self):
        
        parent = Game()
        parent.serve_car()

        Clock.schedule_interval(parent.update, 1.0/60.0)

        return parent



# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
