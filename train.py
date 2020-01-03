"""Training script for Pong DQN.

It should take hyperparameters, optionally some trained weights,
and return a network object, along with dumping weights
to cold-start the trained network."""

import numpy as np
import keras
import tensorflow as tf
#tf.debugging.set_log_device_placement(True)
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

def get_devices():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

def setup_dist_sess():
    mirrored_strategy = tf.distribute.MirroredStrategy()
    return tf.Session(config=tf.ConfigProto(log_device_placement=False))

def get_stats(arr, rtn=False):
    fls = {"mean": np.mean,
        "std": np.std,
        "min": np.min,
        "max": np.max}
    out = {n: f(arr) for n, f in fls.items()}
    if rtn:
        from pandas import Series
        return Series(out)
    else:
        print(out)

def huber_loss(y_true, y_pred, clip_delta=1.0):
    """https://srome.github.io/A-Tour-Of-Gotchas-When-Implementing-Deep-Q-Networks-With-Keras-And-OpenAi-Gym/"""
    # error = np.abs(x)
    error = tf.keras.backend.abs(y_true - y_pred)
    square = tf.keras.backend.minimum(error, clip_delta)
    return 0.5 * tf.keras.backend.square(square) + clip_delta*(error - square)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def make_cnn(lr=2.5e-4, stack_size=4, loss="huber"):
    """Output predefined CNN"""
    model = Sequential()
    model.add(Conv2D(32, (3, 3),
                    activation='relu',
                    input_shape=(80, 80, stack_size),
                    kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(64, (2, 2),
                    activation='relu',
                    kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=200,
                        activation='relu',
                        kernel_initializer="glorot_uniform"))
    model.add(Dense(units=2,
                        activation='softmax',
                        kernel_initializer="RandomNormal"))
    if loss == "huber":
        loss = huber_loss
    # elif loss == "custom":
    #     loss = custom_loss
    model.compile(loss=loss,
                        optimizer=Adam(lr),
                        metrics=['accuracy'])
    return model

# preprocessing used by Karpathy (cf. https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
def prepro(I):
    """ prepro 210x160x3 uint8 frame into (80x80) 2D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    I = I.astype(np.float)
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
    discounted_r /= np.std(discounted_r) #if np.std(discounted_r) is not 0 else 1 #idem
    discounted_r = np.clip(discounted_r, -1, 1)
    return discounted_r

def train_model(model,
                epochs=50,
                memory_size=150000,
                gamma=0.99,
                minibatch_size=32,
                stack_size=4,
                trainfreq=4,
                verbose=1,
                replay_start=None):

    ### ENVIRONMENT
    import gym
    from ReplayMemory import ReplayMemory
    if replay_start is None:
        replay_start=memory_size
    # gym initialization
    env = gym.make("PongNoFrameskip-v4")

    observation = env.reset()
    obs_stack = []
    for k in range(np.random.randint(stack_size, stack_size*2 + 2, size=1)[0]):
        observation, reward, done, info = env.step(0)
        obs_stack.append(observation)
    obs_stack = obs_stack[-stack_size:]
    obs_stack = [prepro(i) for i in obs_stack]

    # Macros
    NO_OP = 0
    UP_ACTION = 2
    DOWN_ACTION = 3


    from datetime import datetime
    import os

    # initialize variables
    running_reward = None
    epochs_before_saving = 5
    log_dir_name = "Logfolder"
    os.path.join(log_dir_name + datetime.now().strftime("%Y%m%d-%H%M%S"))
    log_dir = '.\\' + log_dir_name + datetime.now().strftime("%Y%m%d-%H%M%S") + "\\"

    TOTAL_EPOCHS = epochs
    frame_nb = 0

    def eps_greedy(frame, memory_size=memory_size):
        return max(0.05, 1 - (1/memory_size)*frame)

    RM = ReplayMemory(memory_size,
                    image_size=(80, 80),
                    stack_size=stack_size,
                    minibatch_size=minibatch_size)

    def copy_model(model):
        model.save('temp_model')
        return keras.models.load_model('temp_model')
    target_model = copy_model(model)

    ### main loop
    rng_split = {"choice": 0, "random": 0}
    x_next = np.array(obs_stack).reshape(80, 80, stack_size)
    frame_this_ep = 0
    ctr = 0
    reward_sum = 0
    f = open("log{}.txt".format(datetime.now().strftime("%Y%m%d-%H%M%S")), "w")
    while frame_nb < TOTAL_EPOCHS*50000:
        frame_this_ep += 1
        x = x_next
        out = target_model.predict(x.reshape(1, 80, 80, stack_size))
        if np.random.uniform() < eps_greedy(frame_nb):
            action = UP_ACTION if np.random.uniform() < 0.5 else DOWN_ACTION
            rng_split["random"] += 1
            proba = 0.5
        else:
            action = np.argmax(out) + 2
            rng_split["choice"] += 1
            proba = np.max(out)

        # forward the policy network and sample action according to the proba distribution
        action = UP_ACTION if np.random.uniform() < proba else DOWN_ACTION

        y = np.array([1, 0]) if action == 2 else np.array([0, 1]) # 0 and 1 are our labels
        if ctr < 0:
            print("UP" if action == UP_ACTION else "DOWN")
            print("with probability", proba)

        # do k step in our environment
        stack_reward = 0
        obs_stack = []
        for k in range(stack_size):
            frame_nb += 1
            observation, reward, done, info = env.step(action)
            obs_stack.append(observation)
            stack_reward += reward
            reward_sum += reward

        obs_stack = [prepro(i) for i in obs_stack]
        x_next = np.array(obs_stack).reshape(80, 80, stack_size)

        # log the input and label to train later
        RM.add_example(x, x_next, y, stack_reward, done)

        frame_nb += 1
        ctr += 1
        # end of an episode
        if done:
            observation = env.reset()
            obs_stack = []
            for k in range(np.random.randint(stack_size, stack_size*2 + 2, size=1)[0]):
                observation, reward, done, info = env.step(0)
                obs_stack.append(observation)
            obs_stack = obs_stack[-stack_size:]
            print("Episode finished with total reward: " + str(reward_sum))
            f.write("Episode finished with total reward: " + str(reward_sum) + "\n")
            if not RM.replay_full():
                print("Replay memory filled to: " + str(RM._counter))
                f.write("Replay memory filled to: " + str(RM._counter) + "\n")
            reward_sum = 0
            rng_split = {"choice": 0, "random": 0}
            frame_this_ep = 0
            RM._rewards[-frame_this_ep:] = discount_rewards(RM._rewards[-frame_this_ep:], gamma)

            ctr = 0

        if frame_nb > replay_start and frame_nb % trainfreq == 0:
            #f.writeline("Training at frame", frame_nb)
            xb, xb_next, y, r, d = RM.sample()
            if frame_nb % 10000 == 0:
                target_model = copy_model(model)
                print("Updating target model")
            y_target = np.clip(r + (gamma*y*np.logical_not(d)), -1, 1)
            model.fit(x=xb,
                    y=y_target,
                    verbose=verbose)#,
                    # callbacks=[tbCallBack],
                    #sample_weight=r.reshape(minibatch_size,))
            epoch_nb = int((frame_nb - replay_start)/ 50000)
            if frame_nb % 50000 == 0:
                print("Epoch: " + str(epoch_nb)+ "\n")
                f.write("Epoch: " + str(epoch_nb))
        # Saving the weights used by our model
        if frame_nb/50000 % epochs_before_saving == 0:
            model.save_weights('my_model_weights' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5')

    model.save_weights('my_model_weights' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5')
    f.close()
    return model, RM


def demo(model, steps=1200, stack_size=4, random_pct=0, filename=None):
    import gym
    frames = []

    # gym initialization
    env = gym.make("PongNoFrameskip-v4")

    observation = env.reset()
    obs_stack = []
    for k in range(np.random.randint(stack_size, stack_size*2 + 2, size=1)[0]):
        observation, reward, done, info = env.step(0)
        obs_stack.append(observation)
    obs_stack = obs_stack[-stack_size:]

    # Macros
    UP_ACTION = 2
    DOWN_ACTION = 3

    x_next = np.array(obs_stack).reshape(80, 80, stack_size)
    frame_this_ep = 0
    ctr = 0
    for i in range(steps):
        frame_this_ep += 1
        x = x_next
        out = model.predict(x.reshape(1, 80, 80, stack_size))
        if np.random.uniform() < random_pct:
            action = UP_ACTION if np.random.uniform() < 0.5 else DOWN_ACTION
            proba = 0.5
        else:
            action = np.argmax(out) + 2
            proba = np.max(out)

        y = np.array([1, 0]) if action == 2 else np.array([0, 1]) # 0 and 1 are our labels

        # do k step in our environment
        stack_reward = 0
        obs_stack = []
        for k in range(stack_size):
            frame_nb += 1
            observation, reward, done, info = env.step(action)
            obs_stack.append(observation)
            frames.append(observation)

        obs_stack = [prepro(i) for i in obs_stack]
        x_next = np.array(obs_stack).reshape(80, 80, stack_size)

    import matplotlib.pyplot as plt
    from matplotlib import animation

    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    if filename is not None:
        anim.save(filename, dpi=72, writer='imagemagick')
    return anim

#model = make_cnn()
#sess = setup_dist_sess()
#model_trained = train_model(model, epochs=12, memory_size=4, minibatch_size=2)
