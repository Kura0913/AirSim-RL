# AirSim Reinforcement Learning Platform

## Introduction

This platform is specifically designed for RL training in the AirSim environment and is built on [ **Stable-Baselines3** ](https://stable-baselines3.readthedocs.io/en/master/#).

It offers a convenient switching mechanism and high extensibility. Users can follow the instructions below to learn how to use this platform.

## Useage


### Requirement

* torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
* stable-baselines3
* opencv-python
* numpy==1.26.0
* msgpack-rpc-python
* airsim
* pyyaml

### Installation
* Clone this repo.

```bash
git clone https://github.com/Kura0913/AirSim-RL.git
```

* File tree
```
Airsim-RL
│  agent_settings.yaml
│  config.yaml
│  ouput.txt
│  Platform.py
│  README.md
│  requirements.txt
│  settings.json
│  test.py
│  train.py
│  __init__.py
│  
├─Agent
│      BaseAgent.py
│      DDPGAgent.py
│      PPOAgent.py
│      __init__.py
│      
├─Callback
│      BaseCustomCallback.py
│      DDPGCustomCallback.py
│      PPOCustomCallback.py
│      __init__.py
│      
├─Env
│      AirsimBaseEnv.py
│      AirsimEnv.py
│      __init__.py
│      
├─FeatureExtractor
│      DDPGFeaturesExtractor.py
│      PPOFeaturesExtractor.py
│      __init__.py
│      
├─Network
│      ActionNetwork.py
│      ModifiedResnet.py
│      ValueNetwork.py
│      __init__.py
│      
├─Policy
│      CustomPPOPolicy.py
│      CustomTD3Policy.py
│      __init__.py
│      
├─ReplayBuffer
│      PrioritizedReplayBuffer.py
│      
├─RewardCalculator
│      RewardCalculator.py
│      
└─Tools
        AirsimTools.py
        ShortestPath.py
```

### config.yaml

The <font color=#EB5757>`config.yaml`</font> file contains the settings for training and testing.

#### training_setting

| setting             | description                                                                                                                          |
|---------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| device              | Default is <font color=#EB5757>`auto`</font>. This setting determines whether to use <font color=#EB5757>`cpu`</font> or <font color=#EB5757>`cuda`</font> for training. It is recommended to use <font color=#EB5757>`auto`</font> or <font color=#EB5757>`cpu`</font> for training. |
| load_pretrain_model | Default is <font color=#EB5757>`False`</font>. Set this to True if you need to load a previously trained model.                                                   |
| pretrain_model_path | Specify the path of the model you want to load. If the path does not exist, the training will proceed in default mode.               |
| save_path           | The path where the trained model results will be saved.                                                                              |
| episodes            | Default is <font color=#EB5757>`5000`</font>. Specifies the number of training episodes.                                                                          |
| max_steps           | Default is <font color=#EB5757>`700`</font>. Sets the maximum number of actions executed per episode.                                                             |
| save_episodes       | Saves the model at the specified episode.                                                                                            |
| Env                 | Specifies the <font color=#EB5757>`Env`</font> to be loaded.                                                                                                |
| Algorithm           | Set the algorithm to be used for training.                                                                                           |


#### test_setting

| setting       | description                                                                               |
|---------------|-------------------------------------------------------------------------------------------|
| model_path    | Specify the path of the model you want to test.                                           |
| test_episodes | Default is <font color=#EB5757>`100`</font>. Specifies the number of testing episodes.    |
| save_path     | The path where the testing results will be saved.                                         |


### agent_settings.yaml

The <font color=#EB5757>`agent_settings.yaml`</font> file contains algorithm configurations. When the platform runs, it reads the corresponding <font color=#EB5757>`algorithm`</font> settings from this file based on the algorithm specified in <font color=#EB5757>`config.yaml`</font>.

The common elements in algorithm settings include <font color=#EB5757>`agent_class`</font>, <font color=#EB5757>`callback_class`</font>, and <font color=#EB5757>`features_extractor_class`</font>. The default values are predefined classes already integrated into this platform. Detailed instructions on customizing these classes will be provided in the **Customization** section.

If you want to add a new algorithm and configure it through <font color=#EB5757>`config.yaml`</font>, you need to add the relevant algorithm settings to this file.

### train

Use the following command to start training. The training results will be saved to the path specified in <font color=#EB5757>`config.yaml`</font>.

```bash
python train.py
```

### test
Use the following command to start testing. The model will be loaded based on the settings in <font color=#EB5757>`config.yaml`</font>.

The test results will be saved to the path specified in <font color=#EB5757>`config.yaml`</font>.

```bash
python test.py
```

## Customization 

### description

As mentioned at the beginning, this platform provides a convenient switching mechanism and good extensibility. This section will explain how to add custom classes and configure them for use through <font color=#EB5757>`config.yaml`</font>.

### Agent

Inside the <font color=#EB5757>`Agent`</font> folder, there is a file named <font color=#EB5757>`BaseAgent.py`</font>, which contains the <font color=#EB5757>`BaseAgent`</font> abstract class for users to implement. This class includes abstract methods that need to be defined.

To create a custom agent, follow these steps:

1. Create a new <font color=#EB5757>`.py`</font> file inside the <font color=#EB5757>`Agent`</font> folder.

2. Import <font color=#EB5757>`BaseAgent`</font> using the following code and implement its abstract methods.

```python
from Agent.BaseAgent import BaseAgent

class YourAgent(BaseAgent):
        '''
        Your code...
        '''
```

After implementation, navigate to the <font color=#EB5757>`__init__.py`</font> file in the <font color=#EB5757>`Agent`</font> directory. Import the new agent class and add a corresponding mapping name to the dictionary. Once saved, the new agent can be configured directly through <font color=#EB5757>`algorithm_settings.yaml`</font>.

```python
from Agent.DDPGAgent import DDPGAgent
from Agent.PPOAgent import PPOAgent
from Agent.YourAgent import YourAgent

available_agent = {
    'DDPGAgent': DDPGAgent,
    'PPOAgent': PPOAgent,
    'YourAgent': YourAgent
}
```

### Env
Inside the <font color=#EB5757>`Env`</font> folder, there is a file named <font color=#EB5757>`AirsimBaseEnv.py`</font>, which contains the <font color=#EB5757>`AirsimBaseEnv`</font> abstract class for users to implement. This class includes abstract methods that need to be defined.

To create a custom Env, follow these steps:

1. Create a new <font color=#EB5757>`.py`</font> file inside the <font color=#EB5757>`Env`</font> folder.

2. Import <font color=#EB5757>`AirsimBaseEnv`</font> using the following code and implement its abstract methods.

```python
from Env.AirsimBaseEnv import AirsimBaseEnv

class YourEnv(AirsimBaseEnv):
    '''
    Your code...
    '''
```

After implementation, navigate to the <font color=#EB5757>`__init__.py`</font> file in the <font color=#EB5757>`Env`</font> directory. Import the new Env class and add a corresponding mapping name to the dictionary. Once saved, the new Env can be configured directly through config.yaml.

```python
from Env.AirsimEnv import AirsimEnv
from Env.YourEnv import YourEnv

available_envs = {
    'AirsimEnv': AirsimEnv,
    'YourEnv': YourEnv
}
```


### FeatureExtractor

The base class for FeatureExtractor is already provided in Stable-Baselines3. Please use the following code to inherit and modify it.

```python
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class YourFeatureExtractor(BaseFeatureExtractor):
    '''
    Your code...
    '''
```
After implementation, navigate to the <font color=#EB5757>`__init__.py`</font> file in the <font color=#EB5757>`FeatureExtractor`</font> directory. Import the new FeatureExtractor class and add a corresponding mapping name to the dictionary. Once saved, the new FeatureExtractor can be configured directly through config.yaml.

```python
from FeatureExtractor.DDPGFeaturesExtractor import DDPGFeaturesExtractor
from FeatureExtractor.PPOFeaturesExtractor import PPOFeaturesExtractor
from FeatureExtractor.YourFeaturesExtractor import YourFeaturesExtractor


available_policy_component = {
    'DDPGFeaturesExtractor': DDPGFeaturesExtractor,
    'PPOFeaturesExtractor': PPOFeaturesExtractor,
    'YourFeaturesExtractor': YourFeaturesExtractor
}
```

### Policy
A base template for Policy is not provided because different algorithms require different inheritance structures. Therefore, it is necessary to carefully read the [**official Stable-Baselines3 documentation**]((https://stable-baselines3.readthedocs.io/en/master/#)) and inherit the appropriate class provided by the framework.

For reference, you can check <font color=#EB5757>`CustomPPOPolicy.py`</font> and <font color=#EB5757>`CustomTD3Policy.py`</font> in the <font color=#EB5757>`Policy`</font> directory to see how inheritance is implemented.


After implementation, navigate to the <font color=#EB5757>`__init__.py`</font> file in the <font color=#EB5757>`Policy`</font> directory. Import the new Policy class and add a corresponding mapping name to the dictionary.

```python
from Policy.CustomPPOPolicy import CustomPPOPolicy
from Policy.CustomTD3Policy import CustomTD3Policy
from Policy.YourPolicy import YourPolicy

available_policy = {
    'CustomPPOPolicy': CustomPPOPolicy,
    'CustomTD3Policy': CustomTD3Policy,
    'YourPolicy': YourPolicy

}
```

<font color=#EB5757>`Policy`</font> must be used in conjunction with <font color=#EB5757>`Agent`</font>. The <font color=#EB5757>`save`</font> and <font color=#EB5757>`load`</font> methods in <font color=#EB5757>`Agent`</font> must match the algorithm structure used in <font color=#EB5757>`Policy`</font> to function correctly.

Therefore, <font color=#EB5757>`Policy`</font> does not support switching via <font color=#EB5757>`config.yaml`</font> or <font color=#EB5757>`algorithm_settings.yaml`</font>.

Please specify the <font color=#EB5757>`Policy`</font> class directly in the <font color=#EB5757>`Agent`</font>.

You can refer to <font color=#EB5757>`DDPGAgent.py`</font> and <font color=#EB5757>`PPOAgent.py`</font> in the <font color=#EB5757>`Agent`</font> directory, where they import <font color=#EB5757>`CustomTD3Policy`</font> and <font color=#EB5757>`CustomPPOPolicy`</font>, respectively.