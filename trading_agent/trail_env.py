# Import parent class
from env import Environment

# Import plotting functions
import plotter as plt

# Miscellaneous imports
import os
import random
import numpy as np

BUY = 1
NEUTRAL = 0
SELL = -1


class Trail(Environment):

    def __init__(self, folder, steps, train_data, test_data, test_starts, **kwargs):
        """
        OpenAI override function
        Load data and set action and observation space
        """

        input_shape = 2

        super().__init__(folder, steps, train_data, test_data, test_starts, input_shape, **kwargs)

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

        self.reset()

    def step(self, action):
        """
        OpenAI override function
        One step in Env includes: take action(move agent), store the action and
        the new value, get reward, get ready for next step(+1 position)

        return: the new state, reward and if the data are done
        """

        value = self.data[self.position]

        if action == 2:  # sell / short
            self.action = SELL
            self.value -= self.value * self.turn
            self.short_actions.append([self.position, value])
            self.prev_fin_pos = SELL
        elif action == 1:  # buy / long
            self.action = BUY
            self.value += self.value * self.turn
            self.long_actions.append([self.position, value])
            self.prev_fin_pos = BUY
        else:
            self.action = 0

        if (self.position + 1) < self.data_size:
            state = [self.position, value, self.action, self.value]
            self.memory.append(state)

            self.reward = self.get_reward()
            self.epoch_reward += self.reward

            self.position += 1
            if self.reset_margin:
                # if the agent gets out of boundaries reset him
                value = self.data[self.position]
                upper = value + self.margin
                lower = value - self.margin
                if (self.value > upper or self.value < lower):
                    self.value = value

            self.observation = self.input_s()
        else:
            self.done = True

        return self.observation, self.reward, self.done, {}

    def reset(self):
        """
        After each epoch or at a start of a process (train, test, validation) reset the variables.
        """
        if self.testing:
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
        else:
            begin = 0
            # In case you want to limit the number of data to use in training, use this
            # end = self.limit_data
            end = self.data_size - self.steps - 1
            self.position = random.randint(begin, end)

        # note that the action being taken is for the position X but the input

        self.rewards.append(self.epoch_reward)
        self.epoch_reward = 0
        self.memory = []
        self.long_actions = []
        self.short_actions = []
        self.trades = []
        self.long_prec = 0
        self.short_prec = 0

        self.reward = 0

        self.value = self.data[self.position]
        self.action = 0  # action in step t
        self.prev_action = 0  # action of t-1 step

        self.prev_fin_pos = 0  # Save the last financial position (either long or short)

        self.buy_flag = False
        self.sell_flag = False
        self.done = False

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

        if self.one_hot:
            # A very simple normalization so that the input values are all in the same range (with the one hot)
            input_up *= 1000
            input_down *= 1000

        observation = np.array([input_up, input_down])

        if self.one_hot:
            a = int(self.action == BUY)
            b = int(self.action == SELL)
            c = int(self.action != BUY and self.action != SELL)
            observation = np.append(observation, [a,b,c])

        return observation

    def get_reward(self):
        """
        The reward function of the agent. Based on his action calculate a pnl and a fee as a result
        Normalize the reward to a proper range
        """

        up_margin = self.data[self.position] + self.margin
        c_val = self.data[self.position]
        pr_val = self.data[self.position - 1]
        down_margin = self.data[self.position] - self.margin

        # Because its almost impossible to get the exact number, use an acceptable slack
        if np.abs(c_val - self.value) < 0.00001:
            reward = 1
        elif self.value <= c_val:
            reward = ( self.value - down_margin ) / ( c_val - down_margin )
        else:
            reward = ( self.value - up_margin ) / ( c_val - up_margin )

        if self.ce:
            if self.action != self.prev_action:
                reward = reward - np.abs(reward * self.cost)
        else:
            if (self.action == BUY or self.action == SELL) and (self.action != self.prev_fin_pos):
                reward = reward - np.abs(reward * self.cost)

        if self.dp:
            if ((self.prev_action == BUY) and (self.action == SELL)) or ((self.prev_action == SELL) and (self.action == BUY)):
                reward = reward - np.abs(reward * self.cost)

        self.trade(c_val)
        self.prev_action = self.action

        return reward
