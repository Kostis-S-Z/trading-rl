
## Deep Reinforcement Learning for financial trading using keras-rl

This code is part of the paper "Deep Reinforcement Learning for Financial Trading using Price Trailing" [[1]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8683161) presented at ICASSP 2019.

### Getting Started

Two models were developed in this project. 

The first model (Trail Environment) is our proposed method which is explained analytically the paper above. 

The second model (Deng_Environment) was based on the paper "Deep Direct Reinforcement Learning for Financial Signal Representation and Trading" [[2]](http://cslt.riit.tsinghua.edu.cn/mediawiki/images/a/aa/07407387.pdf) and was used to set a baseline to compare our results. For more information on how this model was used, please refer to our paper of ICASSP 2019.

For both approaches a DQN Agent as implemented in keras-rl [[3]](https://github.com/keras-rl/keras-rl) was used. The environment of the agent was built from scratch to provide a framework to a dataset of Forex exchange rates between two currencies.

### Prerequisites & Installation

I highly suggest using a virtual environment (e.g venv) to run this project.

The code is based mainly on the following packages:
- Python 2.7 / 3.5
- Tensorflow | see installation @ www.tensorflow.org/install/
- OpenAI Gym | pip install gym
- Keras + Keras-rl | pip install keras && keras-rl
- bokeh | pip install bokeh
- pandas | pip install pandas
- h5py | pip install h5py

To easily install all dependencies, run:

```
pip install -r requirements.txt
```

*Note: if you are using python 2.7 then there is only one line of code you need to change to make it work and can be found within the environments file*

Python 2
```
for var, default in var_defaults.iteritems():
```

Python 3
```
for var, default in var_defaults.items():
```


### Hardware

The code can be run on a CPU or a GPU. You can also choose has much RAM and how many CPU cores to allocate when training a model. This allows multiple models to be trained in parallel. You can change the parameters in "dqn_agent.py"

```python
CPU = True  # Selection between CPU or GPU
CPU_cores = 2  # If CPU, how many cores
GPU_mem_use = 0.25  # In both cases the GPU mem is going to be used, choose fraction to use

```

Each epoch usually takes about ~7(GTX 750 ti)/14(i7-7700) seconds (depending on different variable settings). The models were trained on GTX 750ti 2GB GRAM, which usually would take anything from 30min to 2.5hr.

### Dataset

This project was trained and tested with data of US-Euro exchange rates collected over many years. Feel free to try different currencies. The form of the data should be in a usual time-series format as a numpy 1-D array with shape e.g (1000, ).

**Note:** A lot of data of exchange rates found online and are free, can sometimes be incomplete or in a bad format. Usually some preprocessing needs to be done in order to get good results.


### Execute

The structure should look likes this

    .
    ├── trading_agent           # Folder that contains an agent, an environment and code to calculate the PnL
    │   ├── dqn_agent.py            # Code of the agent to execute
    │   └── Environments            # The environments of the agent
    |
    ├── data                  # Folder that contains a train and a test dataset
    │   ├── train_data.npy          # Train data of exchange rates
    │   └── test_data.npy           # A different dataset containing exchange rates to test on
    └── ...

The file _Environment_ is a parent class that contains methods that are used by both environments or that should be implemented. The files *TrailEnv* and *DengEnv* extend this class and implement environment-specific methods. Currently, a method To choose between environments use this variable:

```
METHOD = trailing  # trailing or deng
```

Then simply move to the directory of the agent you want to run and execute:

```
python dqn_agent.py
```

Thanks to keras, you can also stop training whenever you like and the model will save its progress and weights!

### Results

For each model you create a new directory will be created containing many useful information, such as:
- ***agent_info.json***: The parameter values of the model
- ***model.json***: The model structure which you can load back to keras in the future
- ***memory.npy***: The history of the states of the agent
- ***weights_epoch_X.h5f***: The weights of the model in the best epoch
- Information regarding training e.g rewards, the trades the agent made, pnl
- A folder for each test dataset it was tested on

Additionally, a history of the agents trades will show up in your browser using the bokeh environment.

### Note

_Unfortunately, this project is no longer active and I haven't had the chance to work on it for some time now, but I really appreciate that people are interested and I will try my best to answer any questions or fix bugs through pull requests. Note, that my answers might be slow at times since I need a bit of time to re-familiarize myself with the project before moving on to debug._

### References

1. K. S. Zarkias, N. Passalis, A. Tsantekidis and A. Tefas, "Deep Reinforcement Learning for Financial Trading Using Price Trailing," ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Brighton, United Kingdom, 2019, pp. 3067-3071.
doi: 10.1109/ICASSP.2019.8683161 URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8683161&isnumber=8682151

2. Y. Deng, F. Bao, Y. Kong, Z. Ren and Q. Dai, "Deep Direct Reinforcement Learning for Financial Signal Representation and Trading," in IEEE Transactions on Neural Networks and Learning Systems, vol. 28, no. 3, pp. 653-664, March 2017.
doi: 10.1109/TNNLS.2016.2522401

3. *Keras-RL* [github](https://github.com/keras-rl/keras-rl)


### Citation

```
@INPROCEEDINGS{8683161,
author={K. S. {Zarkias} and N. {Passalis} and A. {Tsantekidis} and A. {Tefas}},
booktitle={ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
title={Deep Reinforcement Learning for Financial Trading Using Price Trailing},
year={2019},
volume={},
number={},
pages={3067-3071},
keywords={Deep Reinforcement Learning;Financial Markets;Price Forecasting;Trading},
doi={10.1109/ICASSP.2019.8683161},
ISSN={2379-190X},
month={May},}
```

