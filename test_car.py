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

from PIL import Image

from ai2 import Actor
from ai2 import Critic
from ai2 import ReplayBuffer
from ai2 import TD3
import torch

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

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
# Initializing the last distance

if not os.path.exists("./dump"):
  os.makedirs("./dump")
last_distance = 0

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

    random_action = np.arange(-5, 5, 0.1).tolist()
    max_timesteps = 500000
    start_timesteps = 1e4 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
    eval_freq = 5e3 # How often the evaluation step is performed (after how many timesteps)
    max_timesteps = 5e5 # Total number of iterations/timesteps
    
    expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
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
    
    longueur = 1464
    largeur =  800
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    img = np.asarray(img)
    sand = img/255
    goal_x = 1042
    goal_y = 583
    first_update = True
    swap = 0
    

    reward = 0
    observation_space = (1,42,42)
    action_space = 1
    length = 15
    history = np.zeros(length)
    count = 0
    done =True
    state_dim = observation_space
    action_dim = action_space 

    policy = TD3(state_dim, action_dim, max_action)
    policy.load(file_name, './pytorch_models/')
    obs = np.empty((0))
    orientation =0

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
        self.car.x = 356
        self.car.y = 481

        xx = self.goal_x - self.car.x
        yy = self.goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        
        img_patch = self.img[int(self.car.x)-21:int(self.car.x)+21, int(self.car.y)-21:int(self.car.y)+21]
        background = Image.fromarray(img_patch)


        background.save("./dump/reset{},{}.png".format(self.car.x,self.car.y))


        background = Image.open("./dump/reset{},{}.png".format(self.car.x,self.car.y))

        cimg = Image.open("./gray1.png")
        #print("============Angle{}".format(self.car.angle))
        cimg = cimg.rotate(self.car.angle, expand=True)
        cimg.save("./dump/car23.png")
        cimg = Image.open("./dump/car23.png")

        background.paste(cimg,(11,8),cimg)
        background.save("./dump/reset{},{}.png".format(self.car.x,self.car.y))

        new = PILImage.open("./dump/reset{},{}.png".format(self.car.x,self.car.y)).convert('L')
        new = np.asarray(new)
        new = new/128. -1.

        img_patch = np.expand_dims(new, axis=0)
        return img_patch, orientation
        #return np.array(img_patch)

    def step(self, action):

       
        print("step===========")
        print(action)
        step_done = False    
        
        self.car.move(action)

        xx = self.goal_x - self.car.x
        yy = self.goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.


        distance = np.sqrt((self.car.x - self.goal_x)**2 + (self.car.y - self.goal_y)**2)

        self.car.velocity = Vector(1, 0).rotate(self.car.angle)
        print(1, self.goal_x, self.goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
        """if self.sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            print(1, self.goal_x, self.goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            last_reward = -1

        else: # otherwise
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.car.velocity = Vector(3, 0).rotate(self.car.angle)
            last_reward = -0.2
            print(0, self.goal_x, self.goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            if distance < last_distance:
                last_reward = 0.1"""

        img_patch = self.img[int(self.car.x)-21:int(self.car.x)+21, int(self.car.y)-21:int(self.car.y)+21]
        background = Image.fromarray(img_patch)

        last_reward =1

        #t.paste(arrow,(int(self.car.x),int(self.car.y),arrow)

        background.save("./dump/car_bak.png".format(action))


        background = Image.open("./dump/car_bak.png".format(action))

        cimg = Image.open("./gray1.png")
        #print("============Angle{}".format(self.car.angle))
        cimg = cimg.rotate(self.car.angle, expand=True)
        cimg.save("./dump/car23.png")
        cimg = Image.open("./dump/car23.png")

        background.paste(cimg,(11,8),cimg)
        background.save("./dump/car_bak.png".format(action))

        new = PILImage.open("./dump/car_bak.png".format(action))
        new = np.asarray(new)
        new = new/128.-1.

        img_patch = np.expand_dims(new, axis=0)
        
        if self.car.x < 50:
            self.car.x = 50
            self.car.angle = -self.car.angle
            last_reward = -5
        if self.car.x > self.width - 50:
            print("Here1==========================================")
            self.car.x = self.width - 50
            self.car.angle = -self.car.angle
            last_reward = -5
        if self.car.y < 50:
            self.car.y = 50
            self.car.angle = -self.car.angle
            last_reward = -5
        if self.car.y > self.height - 50:
            print("Here2==========================================")
            self.car.y = self.height - 50
            self.car.angle = -self.car.angle
            last_reward = -5

        
        
       
        if distance < 25:
            if self.swap == 1:
                self.goal_x = 770
                self.goal_y = 357
                self.swap = 2

            else:
                self.goal_x = 384
                self.goal_y = 309
                self.swap = 1
            step_done = True    
                

        return img_patch, last_reward, step_done, orientation

        
        
    

    def evaluate_policy(policy, eval_episodes=10):
        avg_reward = 0.0
        for _ in range(eval_episodes):
            obs = self.reset()
            done = False
            while not done:
                action = self.policy.select_action(obs)
                obs, reward, done = self.step(action)
                avg_reward += reward
        avg_reward /= eval_episodes
        print ("---------------------------------------")
        print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
        print ("---------------------------------------")
        return avg_reward

    def update(self, dt):
        
        if self.first_update:
            self.obs,self.orientation =self.reset()
            self.first_update =False

            
        action = self.policy.select_action(self.obs,self.orientation,-self.orientation)
        action = action.item()

        self.obs, reward, done, self.orientation = self.step(action)

        


        



    

class CarApp(App):

    def build(self):
        
        parent = Game()
        parent.serve_car()

        Clock.schedule_interval(parent.update, 1.0/60.0)

        return parent



# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
