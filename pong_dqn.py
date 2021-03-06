# -*- coding: utf-8 -*-
"""Pong DQN.ipynb

NOTE: This is the training script for the basic dense FFNN that trained successfully and delivered preliminary results.
These results were used at the poster session on Dec 3, 2019.

Based in large extent on the implementation by Andrei Karpathy:
https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/121aLSu52u-_bq6qwEPQEt4HOfebJ8zS4
"""

# Commented out IPython magic to ensure Python compatibility.
import sys
root_path = 'gdrive/My Drive/Colab Notebooks'  #change dir to your project folder

sys.path.append('/content/drive/My Drive/Colab Notebooks/DQN project')
# %cd "drive/My Drive/Colab Notebooks/DQN project"

# !apt-get install -y xvfb python-opengl > /dev/null 2>&1
# !pip install gym pyvirtualdisplay > /dev/null 2>&1
# !apt-get install x11-utils
# #!pip install piglet

#!pip install --upgrade tensorflow

# import necessary modules from keras
import numpy as np
import keras
import tensorflow as tf
#tf.debugging.set_log_device_placement(True)
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

# creates a generic neural network architecture
model = Sequential()

# hidden layer takes a pre-processed frame as input, and has 200 units
model.add(Dense(units=200,input_dim=80*80, activation='relu', kernel_initializer='glorot_uniform'))

# output layer
model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))

# compile the model using traditional Machine Learning losses and optimizers
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

dims = 1

# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 1) , kernel_initializer='RandomNormal'))
# model.add(MaxPooling2D(pool_size=(3, 3)))
# # model.add(Conv2D(64, (2, 2), activation='relu'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(units=200, activation='relu', kernel_initializer="glorot_uniform"))
# model.add(Dense(units=1, activation='sigmoid', kernel_initializer="RandomNormal"))
# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'binary_accuracy'])

import numpy as np
import gym
frames = []

# gym initialization
env = gym.make("Pong-v0")
observation = env.reset()
prev_input = None

# Macros
UP_ACTION = 2
DOWN_ACTION = 3


# Hyperparameters
gamma = 0.99

# initialization of variables used in the main loop
x_train, y_train, rewards = [],[],[]
reward_sum = 0
episode_nb = 0

from datetime import datetime
from keras import callbacks
import os

# initialize variables
resume = True
resume_version = 3
running_reward = None
epochs_before_saving = 100
# set_log_dir = "Logfolder"
# os.path.join("Logfolder" + datetime.now().strftime("%Y%m%d-%H%M%S"))
# log_dir = '.\\Logfolder' + datetime.now().strftime("%Y%m%d-%H%M%S") + "\\"

# load pre-trained model if exist
if (resume and os.path.isfile('my_model_weights.h5')):
    print("loading previous weights")
    model.load_weights('my_model_weights{}.h5'.format(resume_version))

# # add a callback tensorboard object to visualize learning
# tbCallBack = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
#           write_graph=True, write_images=True)


# preprocessing used by Karpathy (cf. https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
def prepro(I, ravel=True):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    I = I.astype(np.float)
    if ravel:
      return I.ravel()
    return I

# reward discount used by Karpathy (cf. https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    r = np.array(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    # we go from last reward to first one so we don't have to do exponentiations
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # if the game ended (in Pong), reset the reward sum
        running_add = running_add * gamma + r[t] # the point here is to use Horner's method to compute those rewards efficiently
        discounted_r[t] = running_add
    discounted_r -= np.mean(discounted_r) #normalizing the result
    discounted_r /= np.std(discounted_r) #idem
    return discounted_r

# from https://www.floydhub.com/explore/templates/reinforcement-learning/gym-retro

import matplotlib.pyplot as plt
from matplotlib import animation

def save_frames_as_gif(frames, filename=None):
    """
    Save a list of frames as a gif
    """
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    if filename:
        anim.save(filename, dpi=72, writer='imagemagick')

### more HYPERPARAMETERS
TOTAL_EPISODES = 4000
GIF_EVERY = TOTAL_EPISODES * 2 + 1
episode_nb = 0

# A list of frames N long where N is how many gifs we want to make
frames_list = [ [] for i in range(int(TOTAL_EPISODES / GIF_EVERY + 1))] # our previous "frames" will be frames_list[0].
# Note that it will probably also save on episode_nb==0
frames_ctr = 0

### main loop
prev_input = None
dims = 1
with open('old_net_train.txt', 'w') as f:

    while episode_nb < TOTAL_EPISODES:

        # preprocess the observation, set input as difference between images
        if dims == 2:
          cur_input = prepro(observation, ravel=False)
          x = cur_input - prev_input if prev_input is not None else np.zeros((80, 80))
          prev_input = cur_input
          proba = model.predict(x.reshape(1, x.shape[0], x.shape[1], 1).T)
        else:
          cur_input = prepro(observation, ravel=True)
          x = cur_input - prev_input if prev_input is not None else np.zeros(80 * 80)
          prev_input = cur_input
          proba = model.predict(np.expand_dims(x, axis=1).T)

        # forward the policy network and sample action according to the proba distribution


        action = UP_ACTION if np.random.uniform() < proba else DOWN_ACTION
        y = 1 if action == 2 else 0 # 0 and 1 are our labels

        # log the input and label to train later
        x_train.append(x)
        y_train.append(y)
        # do one step in our environment
        observation, reward, done, info = env.step(action)
        if episode_nb % GIF_EVERY == 0:                   ## ADDED STUFF HERE
          frames_list[frames_ctr].append(observation) # collecting observation

        rewards.append(reward)
        reward_sum += reward

        # end of an episode
        if done:
            f.write('At the end of episode ' + str(episode_nb) + ' the total reward was ' + str(reward_sum))
            f.write('\n')
            if dims == 2:
              x_train = np.array(x_train)
            # increment episode number
            episode_nb += 1
            if episode_nb % GIF_EVERY == 0:
              save_frames_as_gif(frames_list[frames_ctr], filename='pong-random-steps{}.gif'.format(frames_ctr))   ## ADDED STUFF HERE
              frames_ctr += 1

            # training
            #model.fit(x=np.vstack(x_train), y=np.vstack(y_train), verbose=1, callbacks=[tbCallBack], sample_weight=discount_rewards(rewards, gamma))
            with tf.device('/GPU:0'):
              if dims == 1:
                model.fit(x=np.vstack(x_train), y=np.vstack(y_train), verbose=1, sample_weight=discount_rewards(rewards, gamma))
              else:
                model.fit(x=x_train.reshape(x_train.shape[0], 80, 80, 1), y=np.vstack(y_train), verbose=1, sample_weight=discount_rewards(rewards, gamma))

            # Saving the weights used by our model
            if episode_nb % epochs_before_saving == 0:
                model.save_weights('my_model_weights' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5')

            # Log the reward
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            #tflog('running_reward', running_reward, custom_dir=log_dir)


            # Reinitialization
            x_train, y_train, rewards = [],[],[]
            observation = env.reset()
            reward_sum = 0
            prev_input = None
f.close()
model.save_weights('my_model_weights' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5')

# Frame list collector
frames = []
STEPS = 1200

# initializing our environment
env = gym.make("Pong-v0")
# beginning of an episode
observation = env.reset()

# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib import animation
# # set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display as ipythondisplay

def sample_action(model, x, UP_ACTION=2, DOWN_ACTION=3, dim=2):
    """Return an action by executing a forward pass of a (trained) model"""
    if dim == 1:
      proba = model.predict(np.expand_dims(x, axis=1).T)
    elif dim == 2:
      proba = model.predict(x.reshape(1, x.shape[0], x.shape[1], 1).T)
    action = UP_ACTION if np.random.uniform() < proba else DOWN_ACTION
    return action

dims = 1
prev_input = None
# main loop
for i in range(STEPS):
    if dims == 2:
      cur_input = prepro(observation, ravel=False)
      x = cur_input - prev_input if prev_input is not None else np.zeros((80, 80))
      prev_input = cur_input
      proba = model.predict(x.reshape(1, x.shape[0], x.shape[1], 1).T)
    else:
      cur_input = prepro(observation, ravel=True)
      x = cur_input - prev_input if prev_input is not None else np.zeros(80 * 80)
      prev_input = cur_input
      proba = model.predict(np.expand_dims(x, axis=1).T)

    # sample action
    action = sample_action(model, x, dim=dims)

    #run one step
    observation, reward, done, info = env.step(action)

    screen = env.render(mode='rgb_array')
    plt.imshow(screen)
    #ipythondisplay.clear_output(wait=True)
    #ipythondisplay.display(plt.gcf())

    frames.append(observation) # collecting observation

    # if episode is over, reset to beginning
    if done:
        observation = env.reset()
        frames.append(observation) # collecting observation
        #ipythondisplay.clear_output(wait=True)
        #ipythondisplay.display(plt.gcf())

save_frames_as_gif(frames, filename='pong-random-{}-steps_{}.gif'.format(
     STEPS, "trained_test"
))

# save_frames_as_gif(frames, filename='pong-random-{}-steps_{}.gif'.format(
#      STEPS, datetime.now().strftime("%Y%m%d-%H%M%S")
# ))
