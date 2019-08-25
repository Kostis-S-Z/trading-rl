# Import parent class
from env import Environment

# Miscellaneous imports
import os
import random
import numpy as np

# Set indices for different trade actions
BUY = 1
NEUTRAL = 0
SELL = -1

REWARD_STD = 0.006  # Normalise the reward to a more sensible range. NOTE: this is case specific


class Deng(Environment):

    def __init__(self, folder, steps, train_data, test_data, test_starts, **kwargs):
        """
        OpenAI override function
        Load data and set action and observation space
        """

        setattr(self, "window", kwargs.get("window", 10))

        # set the input shape depending on the window size
        input_shape = self.window

        super().__init__(folder, steps, train_data, test_data, test_starts, input_shape, **kwargs)

        self.mov_ave_train_rew = 0
        self.pnl_1 = 0
        self.pnl_e6 = 0
        self.pnl_e5 = 0

        self.data_mean = np.std(self.data[1:] - self.data[:-1])  # Get the mean of the data to normalise the input

        self.reset()

    def step(self, action):
        """
        OpenAI override function
        One step in the environment means:
        1) take action: move agent to next position and make a trade action (stay, sell, buy)
        2) store the action and the new value
        3) get reward

        return: the new state, reward and if the data are done
        """

        c_val = self.data[self.position]

        if action == 2:  # sell / short
            self.action = SELL
            self.short_actions.append([self.position, c_val])
        elif action == 1:  # buy / long
            self.action = BUY
            self.long_actions.append([self.position, c_val])
        else:
            self.action = NEUTRAL

        # If the are still more data to process
        if (self.position + 1) < self.data_size:
            # Save the current state
            state = [self.position, c_val, self.action]
            self.memory.append(state)

            # Move the agent to the next timestep
            self.position += 1

            # Calculate the reward of the agent
            self.reward = self.get_reward()
            self.epoch_reward += self.reward

            self.observation = self.input_s()
        else:
            self.done = True

        return self.observation, self.reward, self.done, {}

    def reset(self):
        """
        After each epoch or at a start of a process (train, test, validation) reset the variables.
        """
        # If its testing phase, save results in a different folder
        if self.testing:
            # if (len(self.memory) != 0):  #if memory is not empty
            #    self.render()

            self.data = np.load(self.test_data)
            self.data_size = len(self.data)
            self.position = self.test_starts[self.test_starts_index]
            self.test_starts_index += 1

            self.test_folder = self.folder + '/Test_' + str(self.test_starts_index)
            os.makedirs(self.test_folder)
        elif self.validation_process:
            self.data = np.load(self.validation_data)
            self.data_size = len(self.data)
            self.position = self.val_starts[self.val_starts_index]
            self.val_starts_index += 1
            # self.position = random.randint(begin, end)  # random
            # self.position = begin  # fixed
        else:
            begin = self.window + 1
            #end = self.limit_data
            end = self.data_size - self.steps - 1
            self.position = random.randint(begin, end)

        # note that the action being taken is for the position X but the input
        # starts from X - window (ex. X - 50)

        self.memory = []
        self.long_actions = []
        self.short_actions = []
        self.trades = []
        self.long_prec = 0
        self.short_prec = 0

        self.reward = 0
        self.rewards.append(self.epoch_reward)
        self.epoch_reward = 0
        self.action = 0
        self.prev_action = 0
        self.buy_flag = False
        self.sell_flag = False
        self.done = False

        self.observation = self.input_s()

        return self.observation

    def render(self, mode='human', close=False):
        """
        Gym function called at the end of a process.
        """
        super().calculate_pnl(env_type="deng_etal", save=True)
        super().reset()

    def input_s(self):
        """
        Prepare the input to the agent
        """
        zts = []
        index = self.position

        for i in range(self.window):
            c_val = self.data[index]
            pr_val = self.data[index-1]
            zt = ( c_val - pr_val ) / self.data_mean
            zts.append(zt)
            index -= 1

        input_a = np.asarray(zts)

        if self.one_hot:
            # one hot to self.action
            a = int(self.action == BUY)
            b = int(self.action == SELL)
            c = int(self.action != BUY and self.action != SELL)
            input_a = np.append(input_a, [a, b, c])

        observation = input_a

        return observation

    def get_reward(self):
        """
        The reward function of the agent. Based on his action calculate a pnl and a fee as a result
        Normalize the reward to a proper range
        """
        c_val = self.data[self.position]
        pr_val = self.data[self.position - 1]

        dt = self.action  # current action
        zt = ( c_val - pr_val )  # profit/loss
        c = self.cost * c_val  # mandatory fee if dt != dt-1
        m = np.abs(dt - self.prev_action)

        reward = dt * zt - c * m  # / std dev
        reward = reward / REWARD_STD  # divide with standard deviation ~[-2, 2]?

        self.calc_precision(c_val, pr_val)
        self.trade(c_val)

        self.prev_action = self.action
        return reward

    def calc_precision(self, c_val, pr_val):
        """
        Calculate if the actions taken by the agent were indeed correct
        """
        if self.prev_action == BUY:  # buy / long
            if c_val > pr_val or c_val == pr_val:
                self.long_prec += 1
        elif self.prev_action == SELL:  # sell / short
            if c_val < pr_val or c_val == pr_val:
                self.short_prec += 1
