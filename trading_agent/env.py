# Import OpenAI Gym Environment
import gym

# Import plotting functions
import plotter as plt

# Miscellaneous imports
import numpy as np
import warnings

BUY = 1
NEUTRAL = 0
SELL = -1


class Environment(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, folder, steps, train_data, test_data, test_starts, input_shape, **kwargs):
        """
        OpenAI override function
        Load data and set action and observation space
        """

        var_defaults = {
            "limit_data": 1500000,
            "one_hot": True,
            "cost": 0.0001,
            "validation_data": None,
            "val_starts": None
        }

        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))

        self.folder = folder
        self.steps = steps
        self.test_folder = folder
        self.test_data = test_data
        self.data = np.load(train_data)
        self.data_size = len(self.data)

        self.test_starts = test_starts
        self.test_starts_index = 0
        self.val_starts_index = 0

        self.memory = []

        self.action_space = gym.spaces.Discrete(3)
        if self.one_hot:
            input_shape += 3
        self.observation_space = gym.spaces.Box(low=0.5, high=2.0, shape=(input_shape,))

        self.spec = None
        self.validation_process = False
        self.testing = False
        self.validate = False

        self.rewards = []
        self.epoch_reward = 0

        # Borrow this variable from Trail just to calculate the PnL correctly
        # This should always be True inside Deng
        self.ce = True

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human', close=False):
        """
        Gym function called at the end of a process.
        """
        print("Rendering")
        if not self.validation_process:
            np.save(self.test_folder + '/memory_' + str(self.test_starts_index) + '.npy', self.memory)
            title_act = '/test_actions_' + str(self.test_starts_index)
            plt.plot_actions(self.memory, self.long_actions, self.short_actions, self.test_folder, title_act)

    def input_s(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    def calculate_pnl(self, env_type, save=True):
        """
        Calculate the final PnL based on the actions of the agent with three different fee values (slippage)
        """
        actions = np.array([x[2] for x in self.memory])

        values = np.array([x[1] for x in self.memory])
        values = values.reshape((-1,))

        self.pnl_1 = self.pnl_of_trades(env_type, actions, values, slippage=0.0)
        self.pnl_e6 = self.pnl_of_trades(env_type, actions, values, slippage=0.000002)
        self.pnl_e5 = self.pnl_of_trades(env_type, actions, values, slippage=0.00002)

        if save:
            pnls = "Slippage 0.0: " + str(format(self.pnl_1, '.5f')) + "\n"
            pnls += "Slippage 0.000002: " + str(format(self.pnl_e6, '.5f')) + "\n"
            pnls += "Slippage 0.00002: " + str(format(self.pnl_e5, '.5f')) + "\n"

            if len(self.long_actions) != 0:
                l_prec = str(format((self.long_prec / float(len(self.long_actions))), '.2f'))
            else:
                l_prec = str(0)
            longs = str(len(self.long_actions))
            pnls += "Precision Long: " + l_prec + " ("+ str(self.long_prec) + " of " + longs + ")\n"

            if len(self.short_actions) != 0:
                s_prec = str(format((self.short_prec / float(len(self.short_actions))), '.2f'))
            else:
                s_prec = str(0)

            shorts = str(len(self.short_actions))
            pnls += "Precision Short: " + s_prec + " ("+ str(self.short_prec) + " of " + shorts + ")\n"
            pnls += "Test reward: " + str(self.epoch_reward) + "\n"
            print(pnls)
            if self.testing:
                file_sl = '/test_pnl_' + str(self.test_starts_index) + '.out'
                with open(self.test_folder + file_sl, "w") as text_file:
                    text_file.write(pnls)
                file_tr = '/test_trades_' + str(self.test_starts_index) + '.out'
                with open(self.test_folder + file_tr, "w") as text_file:
                    text_file.write(str(self.trades))
            else:
                file_sl = '/train_pnl.out'
                with open(self.folder + file_sl, "w") as text_file:
                    text_file.write(pnls)
                file_tr = '/train_trades.out'
                with open(self.folder + file_tr, "w") as text_file:
                    text_file.write(str(self.trades))
        return 0

    def pnl_of_trades(self, env_type, actions, values, slippage=0.0):
        """
        Function to calculate PnL based on trades
        """
        warnings.warn("No method implemented to calculate the PnL! Returning zero...", Warning)
        return 0

    def trade(self, c_val):
        """
        Save that a trade has been made at the current time step
        """
        if self.action == BUY:
            if not self.buy_flag:
                self.sell_flag = False
                self.buy_flag = True
                self.trades.append([self.position, c_val])
        elif self.action == SELL:
            if not self.sell_flag:
                self.sell_flag = True
                self.buy_flag = False
                self.trades.append([self.position, c_val])

    def plot_actions(self):
        plt.plot_actions(self.folder, self.memory, self.long_actions, self.short_actions)

    def plot_trail(self):
        plt.plot_trail(self.folder, self.memory)

    def plot_train_rewards(self):
        plt.plot_train_rewards(self.folder, self.rewards)
