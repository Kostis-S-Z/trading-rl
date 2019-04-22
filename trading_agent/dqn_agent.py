from deng_env import Deng as DengEnv
from trail_env import Trail as TrailEnv

# Keras modules
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, PReLU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session

# Keras - rl modules
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

# Miscellaneous imports
import plotter as plt
import numpy as np
import os
from pathlib import Path
import datetime
import json

ENV_NAME = 'trading-rl'

trailing = 'trailing'
deng = 'deng'

METHOD = deng  # Choose between environments

directory = str(Path.cwd().parent)  # Get the parent directory of the current working directory
data_directory = directory + "/data"

# Hardware Parameters
CPU = True  # Selection between CPU or GPU
CPU_cores = 1  # If CPU, how many cores
GPU_mem_use = 0.25  # In both cases the GPU mem is going to be used, choose fraction to use

# Data Parameters
train_data = data_directory + '/train_data.npy'  # path to training data
MAX_DATA_SIZE = 12000  # Maximum size of data
DATA_SIZE = MAX_DATA_SIZE  # Size of data you want to use for training

test_data = data_directory + '/test_data.npy'  # path to test data
TEST_EPOCHS = 1  # How many test runs / epochs
TEST_POINTS = [0]  # From which point in the time series to start in each epoch
TEST_STEPS = 2000  # For how many points to run the epoch

# Validation Data
VALIDATE = False  # Use a validation set if available
VAL_DATA = data_directory + '/validation_data.npy'  # path to validation data set
VAL_SIZE = None  # Set the size of the validation data you want to use
TEST_EPOCHS_GEN = None  # How many epochs for validation
TEST_STEPS_GEN = None  # How many steps in each epoch for validation

# Initialize random starts within the validation data
VAL_STARTS = None  # random.randint(low=0, high=VAL_SIZE-TEST_STEPS_GEN-1, size=TEST_EPOCHS_GEN)

# Environment Parameters
# 1: Trailing
RESET_FROM_MARGIN = True  # Reset each time agent deviates out of borders
MARGIN = 0.01  # Set the margin of the borders
TURN = 0.001  # Percentage of adjustment to agent's trail
COST = 0.3  # Cost of changing financial position

# cost in each change of financial position
# if false only when short->long, long->short (neutral in between dont count)
CE = False
# double penalty when the change from short->long or long->short is immediate
DP = False

# 2: Deng etal
COST_D = 0.005  # Different variable of cost for deng's method

NORMALIZE_IN = True  # Normalize the input using z-score scaling

# Algorithm Parameters
STEPS = 500
EPOCHS = 100
WINDOW_LENGTH = 100
ONE_HOT = True  # Agent Position Awareness

GAMMA = 0.95
LR = 0.001
LR_DEC = 0.
TAR_MOD_UP = 0.001
DELTA_CLIP = 1

ALL_STEPS = STEPS * EPOCHS
PERCENTAGE_EXPLORE = 0.8    # should be around 80% of all steps
EXPLORE_STEPS = int(ALL_STEPS * PERCENTAGE_EXPLORE)  # after how many steps should exploration be stabilized

# Neural Net Parameters
NODES = 16  # Neurons
BATCH_SIZE = 64
MEM_SIZE = 100000

PLOT_Q_VALUES = False  # in order to do this you need to edit appropriately the keras files

START_FROM_TRAINED = False  # If you want to already start training from some weights...
TRAINED_WEIGHTS = None  # Provide here the path to the h5f / hdf5 weight file

now = datetime.datetime.now()
DATE = str(now.day) + "." + str(now.month) + "_" + str(now.hour) + ":" + str(now.minute)
FOLDER = METHOD + "/e:" + str(EPOCHS) + "_s:" + str(STEPS) + "_w:" + str(WINDOW_LENGTH) + "_" + DATE


def config_hard():
    """
    Configuration of CPU and GPU
    """
    config = tf.ConfigProto()

    if CPU:
        config = tf.ConfigProto(device_count = {'GPU': 0},
                                intra_op_parallelism_threads=CPU_cores,
                                inter_op_parallelism_threads=CPU_cores)

    config.gpu_options.per_process_gpu_memory_fraction = GPU_mem_use

    set_session(tf.Session(config=config))


def main():
    """
    Initialization of all parameters, neural net, agent, training, validation and testing
    """
    write_model_info()  # save in a file the parameters you are using for this model

    # set up Environment and variables
    if METHOD == trailing:
        env = TrailEnv(FOLDER, STEPS, train_data, test_data, TEST_POINTS, val_data=VAL_DATA, val_starts=VAL_STARTS,
                       limit_data=DATA_SIZE, one_hot=ONE_HOT, cost=COST, margin=MARGIN, turn=TURN,
                       ce=CE, dp=DP, normalize_in=NORMALIZE_IN, reset_margin=RESET_FROM_MARGIN)
    else:
        env = DengEnv(FOLDER, STEPS, train_data, test_data, TEST_POINTS, val_data=VAL_DATA, val_starts=VAL_STARTS,
                      window_in=WINDOW_LENGTH, limit_data=DATA_SIZE, one_hot=ONE_HOT, cost=COST_D)

    # set up the model
    model = set_model(env)

    memory = SequentialMemory(limit=MEM_SIZE, window_length=WINDOW_LENGTH)

    # Exploration policy
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=1.0, value_min=0.1, value_test=0.05, nb_steps=EXPLORE_STEPS)

    nb_actions = env.action_space.n  # set up number of actions (outputs)

    # set up keras-rl agent
    dqn = DQNAgent(model=model, gamma=GAMMA, nb_actions=nb_actions, memory=memory,
                   batch_size=BATCH_SIZE, nb_steps_warmup=1000,
                   target_model_update=TAR_MOD_UP, policy=policy, delta_clip=DELTA_CLIP)

    dqn.compile(Adam(lr=LR, decay=LR_DEC), metrics=['mse'])

    if START_FROM_TRAINED:
        dqn.load_weights(TRAINED_WEIGHTS)

    if VALIDATE:
        train_w_validation(env, dqn)
    else:
        train(env, dqn)

    fin_stats(env, STEPS)
    test(env, dqn)


def set_model(env):
    model = Sequential()
    model.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
    model.add(Dense(NODES))
    model.add(PReLU())
    model.add(Dense(NODES * 2))
    model.add(PReLU())
    model.add(Dense(NODES * 4))
    model.add(PReLU())
    model.add(Dense(NODES * 2))
    model.add(PReLU())
    model.add(Dense(env.action_space.n))
    model.add(Activation('linear'))
    print(model.summary())

    model_json = model.to_json()
    with open(env.folder + '/model.json', "w") as json_file:
        json_file.write(model_json)
    return model


def train_w_validation(env, dqn):
    filepath = env.folder + '/validate/epochs/'
    os.makedirs(filepath)

    # Save each epoch's weight using the callback function of keras
    checkpointer = ModelCheckpoint(filepath=filepath + 'weights.{epoch:02d}.hdf5', monitor='val_loss', verbose=1,
                                   save_best_only=False, save_weights_only=True, mode='auto', period=1)

    dqn.fit(env, nb_steps=EPOCHS*STEPS, nb_max_episode_steps=STEPS,
            visualize=False, verbose=2, callbacks=[checkpointer])

    env.plot_actions()  # plot last epochs training actions
    env.calculate_pnl(env_type=METHOD)  # plot / calculate last epochs training pnl
    np.save(env.folder + '/memory.npy', env.memory)  # save agent's memory
    env.plot_train_rewards()  # plot training rewards
    env.validate = True

    best_epoch = ""
    best_reward = -1000000
    count_ep = 0

    # iterate through each epoch to find the one with the highest reward
    # (the reward might not be the best indicator to find the best epoch! Might want to try pnl as well)
    for weights_file in os.listdir(filepath):

        if weights_file.endswith(".hdf5"):
            count_ep += 1
            print(str(count_ep) + ": Loading: " + weights_file)
            dqn.load_weights(filepath + weights_file)

            env.rewards = []
            env.pnls = []
            env.val_starts_index = 0
            dqn.test(env, nb_episodes=TEST_EPOCHS_GEN, nb_max_episode_steps=TEST_STEPS_GEN, visualize=False)

            epoch_rewards = np.sum(env.rewards) / float(TEST_EPOCHS_GEN)
            if epoch_rewards > best_reward:
                best_epoch = weights_file
                best_reward = epoch_rewards
                print("BEST EPOCH: " + best_epoch + " with: " + str(best_reward))

    path = directory + '/' + filepath + best_epoch
    new_path = directory + '/' + env.folder + '/' + best_epoch
    os.rename(path, new_path)
    print("Loading: " + new_path)
    dqn.load_weights(new_path)

    env.validation_process = False
    env.validate = False


def train(env, dqn):
    dqn.fit(env, nb_steps=EPOCHS*STEPS, nb_max_episode_steps=STEPS, visualize=False, verbose=2)
    dqn.save_weights(env.folder + '/weights_epoch_{}.h5f'.format(EPOCHS), overwrite=True)

    if PLOT_Q_VALUES:
        title = "train_q_values"
        plt.plot_q_values(dqn.q_values_memory, FOLDER, title)
        dqn.q_values_memory = []

    env.plot_actions()
    env.calculate_pnl(env_type=METHOD)
    np.save(env.folder + '/memory.npy', env.memory)
    env.plot_train_rewards()
    with open(env.folder + '/train_rewards.out', "w") as text_file:
        text_file.write(str(env.rewards))


def test(env, dqn):
    env.testing = True
    for x in range(TEST_EPOCHS):
        dqn.test(env, nb_episodes=1, nb_max_episode_steps=TEST_STEPS, visualize=False)

        env.calculate_pnl(env_type=METHOD)
        np.save(env.test_folder + '/memory_' + str(env.test_starts_index) + '.npy', env.memory)
        env.plot_actions()
        if METHOD == trailing:
            env.plot_trail()

    fin_stats(env, TEST_STEPS)


def fin_stats(env, steps):
    longs = len(env.long_actions)
    shorts = len(env.short_actions)
    neutrals = steps - longs - shorts
    print("STATS: Long: ", longs , " Short: ", shorts , " Neutral: ", neutrals, " out of ", steps)


def write_model_info():
    info = {}
    info['date'] = DATE
    # Data parameters
    info['data_size'] = DATA_SIZE
    info['percentage_explore'] = PERCENTAGE_EXPLORE
    info['start_from_trained'] = START_FROM_TRAINED
    info['test_steps'] = TEST_STEPS
    info['test_steps_gen'] = TEST_STEPS_GEN
    if START_FROM_TRAINED:
        info['trained_weights'] = TRAINED_WEIGHTS

    # Neural Net parameters
    info['steps'] = STEPS
    info['epochs'] = EPOCHS
    info['batch_size'] = BATCH_SIZE
    info['mem_size'] = MEM_SIZE

    # Algorithm parameters
    info['gamma'] = GAMMA
    info['window_length_NN'] = WINDOW_LENGTH
    info['learning_rate'] = LR
    info['lr_decay'] = LR_DEC
    info['delta_clip'] = DELTA_CLIP
    info['target_model_update'] = TAR_MOD_UP

    # Environment variables
    info['one_hot'] = ONE_HOT
    info['normalize_in'] = NORMALIZE_IN
    info['cost'] = COST
    if METHOD == trailing:
        info['turn'] = TURN
        info['margin'] = MARGIN
        info['ce'] = CE
        info['dp'] = DP
        info['reset_from_margin'] = RESET_FROM_MARGIN

    os.makedirs(FOLDER)
    with open(FOLDER + '/agent_info.json', 'w') as f:
        json.dump(info, f)


if __name__ == '__main__':
    config_hard()
    main()
