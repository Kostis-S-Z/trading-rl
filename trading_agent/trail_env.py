# Import parent class
from env import Environment

# Import plotting functions
import plotter as plt

# Miscellaneous imports
import os
import random
import numpy as np

# Set indices for different trade actions
BUY = 1
NEUTRAL = 0
SELL = -1


class Trail(Environment):

    def __init__(self, folder, steps, train_data, test_data, test_starts, **kwargs):
        """
        OpenAI override function
        Load data and set action and observation space
        """

        # Input shape originally contains the distance between the current estimation
        # and the upper and the lower margin of the agent.
        # If one-hot is enabled then the input shape is automatically increased to size 5 (2 + 3 binary values)
        input_shape = 2

        super().__init__(folder, steps, train_data, test_data, test_starts, input_shape, **kwargs)

        # Set these default values if not manual values are not given
        var_defaults = {
            "margin": 0.01,
            "turn": 0.001,
            "ce": False,
            "dp": False,
            "reset_margin": True,
        }
        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))

        self.observations = []
        self.pnls_0 = []
        self.epoch_pnl_0 = 0

        # Initialize the environment
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

        c_val = self.data[self.position]  # this is p_t

        if action == 2:  # sell / short
            self.action = SELL
            self.value -= self.value * self.turn  # Update s(t+1) = s(t) + [s(t) * a]
            self.short_actions.append([self.position, c_val])
            self.prev_fin_pos = SELL
        elif action == 1:  # buy / long
            self.action = BUY
            self.value += self.value * self.turn  # Update s(t+1) = s(t) - [s(t) * a]
            self.long_actions.append([self.position, c_val])
            self.prev_fin_pos = BUY
        else:  # stay
            self.action = NEUTRAL  # Update s(t+1) = s(t)

        # If the are still more data to process
        if (self.position + 1) < self.data_size:
            # Save the current state
            state = [self.position, c_val, self.action, self.value]
            self.memory.append(state)

            # Move the agent to the next timestep
            self.position += 1

            # Calculate the reward of the agent
            self.reward = self.get_reward()
            self.epoch_reward += self.reward

            # If the agent gets out of boundaries reset its trade position
            if self.reset_margin:
                value = self.data[self.position]
                upper = value + self.margin
                lower = value - self.margin
                if self.value > upper or self.value < lower:
                    self.value = value

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
            self.data = np.load(self.test_data)
            self.data_size = len(self.data)
            self.position = self.test_starts[self.test_starts_index]
            self.test_starts_index += 1
            self.test_folder = self.folder + '/Test_' + str(self.test_starts_index)
            os.makedirs(self.test_folder)
        elif self.validation_process:  # Validation process is on different data
            self.data = np.load(self.validation_data)
            self.data_size = len(self.data)
            self.position = self.val_starts[self.val_starts_index]
            self.val_starts_index += 1
        else:
            begin = 0
            # In case you want to limit the number of data to use in training, uncomment the next line
            # end = self.limit_data
            end = self.data_size - self.steps - 1
            self.position = random.randint(begin, end)

        # Save previous epoch's rewards
        self.rewards.append(self.epoch_reward)
        # Reset all variables
        self.epoch_reward = 0
        self.memory = []
        self.long_actions = []
        self.short_actions = []
        self.trades = []
        self.long_prec = 0
        self.short_prec = 0

        self.reward = 0

        # Start again from a new index
        self.value = self.data[self.position]
        self.action = 0  # action in step t
        self.prev_action = 0  # action of t-1 step

        self.prev_fin_pos = 0  # Save the last financial position (either long or short)

        self.buy_flag = False
        self.sell_flag = False
        self.done = False

        # Get the first observation vector
        self.observation = self.input_s()

        return self.observation

    def render(self, mode='human', close=False):
        """
        Gym function called at the end of a process.
        """
        super().calculate_pnl(env_type="trailing", save=True)
        super().reset()
        if not self.validation_process:
            title_trail = '/test_trail_' + str(self.test_starts_index)
            plt.plot_trail(self.memory, self.test_folder, title_trail)

    def input_s(self):
        """
        Prepare the input to the agent
        """
        input_up = (self.data[self.position] + self.margin) - self.value  # [0] - up values_std: ~0.0006
        input_down = self.value - (self.data[self.position] - self.margin)  # [2] - down

        # Rescale input to a better range
        input_up = input_up / 0.0006
        input_down = input_down / 0.0006

        observation = np.array([input_up, input_down])

        if self.one_hot:
            a = int(self.action == BUY)
            b = int(self.action == SELL)
            c = int(self.action != BUY and self.action != SELL)
            observation = np.append(observation, [a, b, c])

        return observation

    def get_reward(self):
        """
        The reward function of the agent. Based on his action calculate a pnl and a fee as a result
        Normalize the reward to a proper range
        """

        c_val = self.data[self.position]  # p(t)
        up_margin = c_val + self.margin  # p(t) + m
        down_margin = c_val - self.margin  # p(t) - m

        # Because its almost impossible to get the exact number, use an acceptable slack
        if np.abs(c_val - self.value) < 0.00001:
            reward = 1
        elif self.value <= c_val:  # s(t) < p(t)
            reward = (self.value - down_margin) / (c_val - down_margin)  # same as ( s(t) - p(t) + m ) / m
        else:
            reward = (self.value - up_margin) / (c_val - up_margin)  # same as (p(t) - s(t) + m ) / m

        if self.ce:  # If there was a change in the financial position (trade action) of the agent, apply a fee
            if self.action != self.prev_action:
                reward = reward - np.abs(reward * self.cost)
        else:
            if (self.action == BUY or self.action == SELL) and (self.action != self.prev_fin_pos):
                reward = reward - np.abs(reward * self.cost)

        if self.dp:  # Add another extra penalty if the trade action was from short to long or from long to short
            if ((self.prev_action == BUY) and (self.action == SELL)) or ((self.prev_action == SELL) and (self.action == BUY)):
                reward = reward - np.abs(reward * self.cost)

        self.trade(c_val)
        self.prev_action = self.action

        return reward
