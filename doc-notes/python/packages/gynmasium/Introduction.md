>  version: v1.1.1
# Basic Usage
Gymnasium is a project that provides an API (application programming interface) for all single agent reinforcement learning environments, with implementations of common environments: cartpole, pendulum, mountain-car, mujoco, atari, and more. This page will outline the basics of how to use Gymnasium including its four key functions: [`make()`](https://gymnasium.farama.org/api/registry/#gymnasium.make "gymnasium.make"), [`Env.reset()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset "gymnasium.Env.reset"), [`Env.step()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.step "gymnasium.Env.step") and [`Env.render()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.render "gymnasium.Env.render").
>  Gymnasium 为单个智能体的 RL 环境提供了 API，同时实现了常见的环境：推车、摆、登山车、mujoco、atari 等
>  Gymnasium 的四个关键函数是 `make(), Env.reset(), Env.step(), Env.render()`

At the core of Gymnasium is [`Env`](https://gymnasium.farama.org/api/env/#gymnasium.Env "gymnasium.Env"), a high-level python class representing a markov decision process (MDP) from reinforcement learning theory (note: this is not a perfect reconstruction, missing several components of MDPs). The class provides users the ability generate an initial state, transition / move to new states given an action and visualize the environment. Alongside [`Env`](https://gymnasium.farama.org/api/env/#gymnasium.Env "gymnasium.Env"), [`Wrapper`](https://gymnasium.farama.org/api/wrappers/#gymnasium.Wrapper "gymnasium.Wrapper") are provided to help augment / modify the environment, in particular, the agent observations, rewards and actions taken.
>  Gymnasium 的核心类是 `Env` ，用于表示一个 MDP
>  `Env` 可以生成初始状态、在给定动作时转移到下一个状态、可视化环境
>  `Warpper` 类用于强化、修改环境，尤其是关于智能体的观测、奖励和执行的动作

## Initializing Environments
Initializing environments is very easy in Gymnasium and can be done via the [`make()`](https://gymnasium.farama.org/api/registry/#gymnasium.make "gymnasium.make") function:

```python
import gymnasium as gym
env = gym.make('CartPole-v1')
```

This function will return an [`Env`](https://gymnasium.farama.org/api/env/#gymnasium.Env "gymnasium.Env") for users to interact with. To see all environments you can create, use [`pprint_registry()`](https://gymnasium.farama.org/api/registry/#gymnasium.pprint_registry "gymnasium.pprint_registry"). Furthermore, [`make()`](https://gymnasium.farama.org/api/registry/#gymnasium.make "gymnasium.make") provides a number of additional arguments for specifying keywords to the environment, adding more or less wrappers, etc. See [`make()`](https://gymnasium.farama.org/api/registry/#gymnasium.make "gymnasium.make") for more information.

>  `make()` 用于初始化一个环境，它返回一个 `Env` 供用户交互
>  `pprint_registry()` 会展示我们可以创建的所有环境
>  `make()` 提供了参数用于为环境额外添加 wrappers

## Interacting with the Environment
In reinforcement learning, the classic “agent-environment loop” pictured below is a simplified representation of how an agent and environment interact with each other. The agent receives an observation about the environment, the agent then selects an action, which the environment uses to determine the reward and the next observation. The cycle then repeats itself until the environment ends (terminates).
>  RL 中，agent 收到关于环境的观测，选择一个动作，环境基于动作决定奖励和下一个观测
>  这一过程重复直到环境结束/终止

[![../../_images/AE_loop.png](https://gymnasium.farama.org/_images/AE_loop.png)](https://gymnasium.farama.org/_images/AE_loop.png)

For Gymnasium, the “agent-environment-loop” is implemented below for a single episode (until the environment ends). See the next section for a line-by-line explanation. Note that running this code requires installing swig (`pip install swig` or [download](https://www.swig.org/download.html)) along with `pip install "gymnasium[box2d]"`.
>  一个 agent-environment-loop 的示例如下

```python
import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

env.close()
```

The output should look something like this:

[![https://user-images.githubusercontent.com/15806078/153222406-af5ce6f0-4696-4a24-a683-46ad4939170c.gif](https://user-images.githubusercontent.com/15806078/153222406-af5ce6f0-4696-4a24-a683-46ad4939170c.gif)](https://user-images.githubusercontent.com/15806078/153222406-af5ce6f0-4696-4a24-a683-46ad4939170c.gif)

### Explaining the code
First, an environment is created using [`make()`](https://gymnasium.farama.org/api/registry/#gymnasium.make "gymnasium.make") with an additional keyword `"render_mode"` that specifies how the environment should be visualized. See [`Env.render()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.render "gymnasium.Env.render") for details on the default meaning of different render modes. In this example, we use the `"LunarLander"` environment where the agent controls a spaceship that needs to land safely.
>  上述代码中，首先用 `make()` 创建了一个环境，参数 `render_mode` 指定了环境应该如何可视化

After initializing the environment, we [`Env.reset()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset "gymnasium.Env.reset") the environment to get the first observation of the environment along with an additional information. For initializing the environment with a particular random seed or options (see the environment documentation for possible values) use the `seed` or `options` parameters with `reset()`.
>  初始化后，`Env.reset()` 重置环境，得到了第一个观测和额外信息

As we wish to continue the agent-environment loop until the environment ends, which is in an unknown number of timesteps, we define `episode_over` as a variable to know when to stop interacting with the environment along with a while loop that uses it.

Next, the agent performs an action in the environment, [`Env.step()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.step "gymnasium.Env.step") executes the selected action (in this case random with `env.action_space.sample()`) to update the environment. This action can be imagined as moving a robot or pressing a button on a games’ controller that causes a change within the environment. As a result, the agent receives a new observation from the updated environment along with a reward for taking the action. This reward could be for instance positive for destroying an enemy or a negative reward for moving into lava. One such action-observation exchange is referred to as a **timestep**.
>  得到观测后，`Env.step()` 在环境中执行选定的一个动作，进而更新环境，这里动作直接用 `env.action_space_sample()` 进行采样
>  `step()` 会返回执行该动作后的新观测以及奖励
>  至此，我们完成了一个时间步

However, after some timesteps, the environment may end, this is called the terminal state. For instance, the robot may have crashed, or may have succeeded in completing a task, the environment will need to stop as the agent cannot continue. In Gymnasium, if the environment has terminated, this is returned by `step()` as the third variable, `terminated`. Similarly, we may also want the environment to end after a fixed number of timesteps, in this case, the environment issues a truncated signal. If either of `terminated` or `truncated` are `True` then we end the episode but in most cases users might wish to restart the environment, this can be done with `env.reset()`.
>  一定时间步后，环境达到了终止状态，智能体不能再继续，此时 `step()` 返回的第三个变量 `terminated` 将为 `true`
>  如果环境达到了设定的结束轮次，环境也会终止，并释放截断信号，此时 `step()` 返回的第四个变量 `truncated` 将为 `true`
>  结束后，我们就完成了一个回合
>  之后，可以通过 `env.reset()` 重新启动环境

## Action and observation spaces
Every environment specifies the format of valid actions and observations with the [`action_space`](https://gymnasium.farama.org/api/env/#gymnasium.Env.action_space "gymnasium.Env.action_space") and [`observation_space`](https://gymnasium.farama.org/api/env/#gymnasium.Env.observation_space "gymnasium.Env.observation_space") attributes. This is helpful for knowing both the expected input and output of the environment, as all valid actions and observations should be contained with their respective spaces. In the example above, we sampled random actions via `env.action_space.sample()` instead of using an agent policy, mapping observations to actions which users will want to make.
>  环境通过 `action_space, observation_space` 属性指定其有效的动作和观测空间

Importantly, `Env.action_space` and `Env.observation_space` are instances of `Space`, a high-level python class that provides the key functions: `Space.contains()` and `Space.sample()`. Gymnasium has support for a wide range of spaces that users might need:
>  `Env.action_space, Env.observation_space` 都是类 `Space` 的实例，该类提供了 `containes(), sample()` 方法
>  Gymnasium 内建了许多空间

- [`Box`](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Box "gymnasium.spaces.Box"): describes bounded space with upper and lower limits of any n-dimensional shape.
- [`Discrete`](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Discrete "gymnasium.spaces.Discrete"): describes a discrete space where `{0, 1, ..., n-1}` are the possible values our observation or action can take.
- [`MultiBinary`](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.MultiBinary "gymnasium.spaces.MultiBinary"): describes a binary space of any n-dimensional shape.
- [`MultiDiscrete`](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.MultiDiscrete "gymnasium.spaces.MultiDiscrete"): consists of a series of [`Discrete`](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Discrete "gymnasium.spaces.Discrete") action spaces with a different number of actions in each element.
- [`Text`](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Text "gymnasium.spaces.Text"): describes a string space with a minimum and maximum length.
- [`Dict`](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Dict "gymnasium.spaces.Dict"): describes a dictionary of simpler spaces.
- [`Tuple`](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Tuple "gymnasium.spaces.Tuple"): describes a tuple of simple spaces.
- [`Graph`](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Graph "gymnasium.spaces.Graph"): describes a mathematical graph (network) with interlinking nodes and edges.
- [`Sequence`](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Sequence "gymnasium.spaces.Sequence"): describes a variable length of simpler space elements.

For example usage of spaces, see their [documentation](https://gymnasium.farama.org/introduction/api/spaces) along with [utility functions](https://gymnasium.farama.org/introduction/api/spaces/utils). There are a couple of more niche spaces [`Graph`](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Graph "gymnasium.spaces.Graph"), [`Sequence`](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Sequence "gymnasium.spaces.Sequence") and [`Text`](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Text "gymnasium.spaces.Text").

## Modifying the environment
Wrappers are a convenient way to modify an existing environment without having to alter the underlying code directly. Using wrappers will allow you to avoid a lot of boilerplate code and make your environment more modular. Wrappers can also be chained to combine their effects. Most environments that are generated via [`gymnasium.make()`](https://gymnasium.farama.org/api/registry/#gymnasium.make "gymnasium.make") will already be wrapped by default using the [`TimeLimit`](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.TimeLimit "gymnasium.wrappers.TimeLimit"), [`OrderEnforcing`](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.OrderEnforcing "gymnasium.wrappers.OrderEnforcing") and [`PassiveEnvChecker`](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.PassiveEnvChecker "gymnasium.wrappers.PassiveEnvChecker").
>  Wrappers 用于在不修改底层代码的情况下修改现存的环境，Wrappers 之间可以互相叠加，结合各自的效果
>  大多数由 `make()` 生成的环境默认已经被 `TimeLimit, OrderEnforcing, PassiveEnvChecker` wrapped

In order to wrap an environment, you must first initialize a base environment. Then you can pass this environment along with (possibly optional) parameters to the wrapper’s constructor:
>  我们创建一个基础环境，然后将该环境和对应的参数传入 wrapper 的构造函数，得到 wrapped 的环境

```
>>> import gymnasium as gym
>>> from gymnasium.wrappers import FlattenObservation
>>> env = gym.make("CarRacing-v3")
>>> env.observation_space.shape
(96, 96, 3)
>>> wrapped_env = FlattenObservation(env)
>>> wrapped_env.observation_space.shape
(27648,)
```

Gymnasium already provides many commonly used wrappers for you. Some examples:
>  Gymnasium 提供了内建的 wrappers

- [`TimeLimit`](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.TimeLimit "gymnasium.wrappers.TimeLimit"): Issues a truncated signal if a maximum number of timesteps has been exceeded (or the base environment has issued a truncated signal).
- [`ClipAction`](https://gymnasium.farama.org/api/wrappers/action_wrappers/#gymnasium.wrappers.ClipAction "gymnasium.wrappers.ClipAction"): Clips any action passed to `step` such that it lies in the base environment’s action space.
- [`RescaleAction`](https://gymnasium.farama.org/api/wrappers/action_wrappers/#gymnasium.wrappers.RescaleAction "gymnasium.wrappers.RescaleAction"): Applies an affine transformation to the action to linearly scale for a new low and high bound on the environment.
- [`TimeAwareObservation`](https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.TimeAwareObservation "gymnasium.wrappers.TimeAwareObservation"): Add information about the index of timestep to observation. In some cases helpful to ensure that transitions are Markov.

For a full list of implemented wrappers in Gymnasium, see [wrappers](https://gymnasium.farama.org/api/wrappers/).

If you have a wrapped environment, and you want to get the unwrapped environment underneath all the layers of wrappers (so that you can manually call a function or change some underlying aspect of the environment), you can use the [`unwrapped`](https://gymnasium.farama.org/api/env/#gymnasium.Env.unwrapped "gymnasium.Env.unwrapped") attribute. If the environment is already a base environment, the [`unwrapped`](https://gymnasium.farama.org/api/env/#gymnasium.Env.unwrapped "gymnasium.Env.unwrapped") attribute will just return itself.
>  wrapped 环境的 `unwrapped` 属性存储了没有被 wrapped 的底层环境，基础环境的 `unwrapped` 指向它自己

```
>>> wrapped_env
<FlattenObservation<TimeLimit<OrderEnforcing<PassiveEnvChecker<CarRacing<CarRacing-v3>>>>>>
>>> wrapped_env.unwrapped
<gymnasium.envs.box2d.car_racing.CarRacing object at 0x7f04efcb8850>
```

# Training an Agent
This page provides a short outline of how to train an agent for a Gymnasium environment, in particular, we will use a tabular based Q-learning to solve the Blackjack v1 environment. For a full complete version of this tutorial and more training tutorials for other environments and algorithm, see [this](https://gymnasium.farama.org/introduction/train_agent/#../tutorials/training_agents). Please read [basic usage](https://gymnasium.farama.org/introduction/basic_usage/) before reading this page. Before we implement any code, here is an overview of Blackjack and Q-learning.

Blackjack is one of the most popular casino card games that is also infamous for being beatable under certain conditions. This version of the game uses an infinite deck (we draw the cards with replacement), so counting cards won’t be a viable strategy in our simulated game. The observation is a tuple of the player’s current sum, the value of the dealers face-up card and a boolean value on whether the player holds a usable case. The agent can pick between two actions: stand (0) such that the player takes no more cards and hit (1) such that the player will take another card. To win, your card sum should be greater than the dealers without exceeding 21. The game ends if the player selects stand or if the card sum is greater than 21. Full documentation can be found at [https://gymnasium.farama.org/environments/toy_text/blackjack](https://gymnasium.farama.org/environments/toy_text/blackjack).
>  二十一点是赌场中备受欢迎的纸牌游戏之一。此版本的游戏使用无限牌组（我们有放回地抽牌），在我们模拟的游戏中，算牌并非可行之策。
>  观察结果是玩家当前的点数和、庄家明牌的点数，以及一个表示玩家是否持有可用 case 的布尔值所组成的元组。
>  智能体可在两种动作中选择：停牌（0），即玩家不再拿牌；或击牌（1），即玩家再拿一张牌。
>  若玩家的牌点总和超过庄家且不超过 21 点，则玩家获胜。当玩家选择停牌或牌点总和超过 21 点时，游戏结束。

Q-learning is a model-free off-policy learning algorithm by Watkins, 1989 for environments with discrete action spaces and was famous for being the first reinforcement learning algorithm to prove convergence to an optimal policy under certain conditions.
>  Q-Learning 是无需模型的异策略学习算法，针对离散的动作空间，在一定条件下，Q-Learning 被证明会收敛到最优策略

## Executing an action
After receiving our first observation, we are only going to use the `env.step(action)` function to interact with the environment. This function takes an action as input and executes it in the environment. Because that action changes the state of the environment, it returns four useful variables to us. These are:

- `next observation`: This is the observation that the agent will receive after taking the action.
- `reward`: This is the reward that the agent will receive after taking the action.
- `terminated`: This is a boolean variable that indicates whether or not the environment has terminated, i.e., ended due to an internal condition.
- `truncated`: This is a boolean variable that also indicates whether the episode ended by early truncation, i.e., a time limit is reached.
- `info`: This is a dictionary that might contain additional information about the environment.

>  接收到第一个观测后，我们使用 `env.step(action)`  和环境交互，`env.step()` 返回的变量如上

The `next observation`, `reward`, `terminated` and `truncated` variables are self-explanatory, but the `info` variable requires some additional explanation. This variable contains a dictionary that might have some extra information about the environment, but in the Blackjack-v1 environment you can ignore it. For example in Atari environments the info dictionary has a `ale.lives` key that tells us how many lives the agent has left. If the agent has 0 lives, then the episode is over.
>  其中，`info` 变量是一个字典，其中包含了关于环境的额外信息

Note that it is not a good idea to call `env.render()` in your training loop because rendering slows down training by a lot. Rather try to build an extra loop to evaluate and showcase the agent after training.

## Building an agent
Let’s build a Q-learning agent to solve Blackjack! We’ll need some functions for picking an action and updating the agents action values. To ensure that the agents explores the environment, one possible solution is the epsilon-greedy strategy, where we pick a random action with the percentage `epsilon` and the greedy action (currently valued as the best) `1 - epsilon`.

```python
from collections import defaultdict
import gymnasium as gym
import numpy as np

class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
```

## Training the agent
To train the agent, we will let the agent play one episode (one complete game is called an episode) at a time and update it’s Q-values after each action taken during the episode. The agent will have to experience a lot of episodes to explore the environment sufficiently.
>  训练时，我们让 agent 执行多个回合游戏，然后在回合中的每个动作执行后更新其 Q-values

```python
# hyperparameters
learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = BlackjackAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)
```

Info: The current hyperparameters are set to quickly train a decent agent. If you want to converge to the optimal policy, try increasing the `n_episodes` by 10x and lower the learning_rate (e.g. to 0.001).

```python
from tqdm import tqdm

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()
```

You can use `matplotlib` to visualize the training reward and length.

```python
from matplotlib import pyplot as plt

def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

# Smooth over a 500 episode window
rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(
    env.return_queue,
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(
    env.length_queue,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)

axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(
    agent.training_error,
    rolling_length,
    "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
plt.tight_layout()
plt.show()
```

![](https://gymnasium.farama.org/_images/blackjack_training_plots.png)

Hopefully this tutorial helped you get a grip of how to interact with Gymnasium environments and sets you on a journey to solve many more RL challenges.

It is recommended that you solve this environment by yourself (project based learning is really effective!). You can apply your favorite discrete RL algorithm or give Monte Carlo ES a try (covered in `Sutton & Barto <http://incompleteideas.net/book/the-book-2nd.html>`_, section 5.3) - this way you can compare your results directly to the book.

Best of luck!

# Create a Custom Environment
This page provides a short outline of how to create custom environments with Gymnasium, for a more [complete tutorial](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/) with rendering, please read [basic usage](https://gymnasium.farama.org/introduction/basic_usage/) before reading this page.

We will implement a very simplistic game, called `GridWorldEnv`, consisting of a 2-dimensional square grid of fixed size. The agent can move vertically or horizontally between grid cells in each timestep and the goal of the agent is to navigate to a target on the grid that has been placed randomly at the beginning of the episode.

Basic information about the game

- Observations provide the location of the target and agent.
- There are 4 discrete actions in our environment, corresponding to the movements “right”, “up”, “left”, and “down”.
- The environment ends (terminates) when the agent has navigated to the grid cell where the target is located.
- The agent is only rewarded when it reaches the target, i.e., the reward is one when the agent reaches the target and zero otherwise.

## Environment `__init__`
Like all environments, our custom environment will inherit from [`gymnasium.Env`](https://gymnasium.farama.org/api/env/#gymnasium.Env "gymnasium.Env") that defines the structure of environment. One of the requirements for an environment is defining the observation and action space, which declare the general set of possible inputs (actions) and outputs (observations) of the environment. As outlined in our basic information about the game, our agent has four discrete actions, therefore we will use the `Discrete(4)` space with four options.
>  自定义环境需要继承 `Env` 类，环境应该定义其观测和动作空间
>  我们用 `Discrete(4)` 空间作为动作空间

For our observation, there are a couple options, for this tutorial we will imagine our observation looks like `{"agent": array([1, 0]), "target": array([0, 3])}` where the array elements represent the x and y positions of the agent or target. Alternative options for representing the observation is as a 2d grid with values representing the agent and target on the grid or a 3d grid with each “layer” containing only the agent or target information. Therefore, we will declare the observation space as [`Dict`](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Dict "gymnasium.spaces.Dict") with the agent and target spaces being a [`Box`](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Box "gymnasium.spaces.Box") allowing an array output of an int type.
>  我们用 `Dict` 作为状态空间，`Dict` 中，`"agent"` 对应一个 `Box`，表示 agent 可能的位置，`"target"` 对应一个 `Box` ，表示 target 可能的位置

For a full list of possible spaces to use with an environment, see [spaces](https://gymnasium.farama.org/api/spaces/)

```python
from typing import Optional
import numpy as np
import gymnasium as gym

class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 5):
        # The size of the square grid
        self.size = size

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)
        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }
```

## Constructing Observations
Since we will need to compute observations both in [`Env.reset()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset "gymnasium.Env.reset") and [`Env.step()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.step "gymnasium.Env.step"), it is often convenient to have a method `_get_obs` that translates the environment’s state into an observation. However, this is not mandatory and you can compute the observations in [`Env.reset()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset "gymnasium.Env.reset") and [`Env.step()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.step "gymnasium.Env.step") separately.
>  在 `Env.reset(), Env.step()` 被调用时，我们都需要计算观测
>  我们一般会定义 `_get_obs` 方法将环境状态转化为观测

```python
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
```

We can also implement a similar method for the auxiliary information that is returned by [`Env.reset()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset "gymnasium.Env.reset") and [`Env.step()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.step "gymnasium.Env.step"). In our case, we would like to provide the manhattan distance between the agent and the target:
>  `Env.reset(), Env.step()` 还会返回辅助信息，我们为它的处理定义方法，本例中，我们会返回 agent 和 target 之间的曼哈顿距离作为环境的额外信息

```python
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
```

Oftentimes, info will also contain some data that is only available inside the [`Env.step()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.step "gymnasium.Env.step") method (e.g., individual reward terms). In that case, we would have to update the dictionary that is returned by `_get_info` in [`Env.step()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.step "gymnasium.Env.step").
>  `info` 时常会包含在 `Env.step()` 方法中才可见的数据，例如独立的奖励项，因此我们一般会在 `Env.step()` 中调用 `_get_info()` 之后，进一步更新 `_get_info()` 返回的字典，再返回

## Reset function
The purpose of [`reset()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset "gymnasium.Env.reset") is to initiate a new episode for an environment and has two parameters: `seed` and `options`. The seed can be used to initialize the random number generator to a deterministic state and options can be used to specify values used within reset. On the first line of the reset, you need to call `super().reset(seed=seed)` which will initialize the random number generate ([`np_random`](https://gymnasium.farama.org/api/env/#gymnasium.Env.np_random "gymnasium.Env.np_random")) to use through the rest of the [`reset()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset "gymnasium.Env.reset").
>  `reset()` 的目的是为环境初始化一个新的回合，`reset()` 有两个参数 `seed, options` 
>  `seed` 用于控制随机初始化，在 `reset()` 定义的第一行中，我们需要调用 `super().reset(seed=seed)` ，以初始化随机数生成器

Within our custom environment, the [`reset()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset "gymnasium.Env.reset") needs to randomly choose the agent and target’s positions (we repeat this if they have the same position). The return type of [`reset()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset "gymnasium.Env.reset") is a tuple of the initial observation and any auxiliary information. Therefore, we can use the methods `_get_obs` and `_get_info` that we implemented earlier for that:
>  在 `reset()` 的定义中，它需要随机选择 agent 和 target 的位置，`reset()` 的返回类型为元组，包含了初始位置信息和其他辅助信息

```python
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
```

## Step function
The [`step()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.step "gymnasium.Env.step") method usually contains most of the logic for your environment, it accepts an `action` and computes the state of the environment after the applying the action, returning a tuple of the next observation, the resulting reward, if the environment has terminated, if the environment has truncated and auxiliary information.
>  `step()` 方法接受 `action`，并计算执行动作后的环境状态，返回下一个观测和奖励，以及环境是否终止、中断的信息，以及辅助信息

For our environment, several things need to happen during the step function:

> - We use the `self._action_to_direction` to convert the discrete action (e.g., 2) to a grid direction with our agent location. To prevent the agent from going out of bounds of the grid, we clip the agent’s location to stay within bounds.
>     
> - We compute the agent’s reward by checking if the agent’s current position is equal to the target’s location.
>     
> - Since the environment doesn’t truncate internally (we can apply a time limit wrapper to the environment during [`make()`](https://gymnasium.farama.org/api/registry/#gymnasium.make "gymnasium.make")), we permanently set truncated to False.
>     
> - We once again use `_get_obs` and `_get_info` to obtain the agent’s observation and auxiliary information.

>  在我们的 `step()` 中：
>  我们用 `self._action_to_direction` 将动作转化为 agent 移动的方向，如果移动出圈，就让 agent 保持不动
>  我们检查 agent 目前的位置是否等于目标位置，以计算奖励
>  因为环境不会自我中断 (当然我们可以在 `make()` 中传入一个时间限制 wrapper)，我们始终返回 `truncated=False`
>  我们使用 `_get_obs, _get_info` 获取 agent 的观测和辅助信息

```python
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid bounds
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # An environment is completed if and only if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = False
        reward = 1 if terminated else 0  # the agent is only reached at the end of the episode
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
```

## Registering and making the environment
While it is possible to use your new custom environment now immediately, it is more common for environments to be initialized using [`gymnasium.make()`](https://gymnasium.farama.org/api/registry/#gymnasium.make "gymnasium.make"). In this section, we explain how to register a custom environment then initialize it.
>  我们需要注册自定义环境，使得它可以被 `make()` 创建

The environment ID consists of three components, two of which are optional: an optional namespace (here: `gymnasium_env`), a mandatory name (here: `GridWorld`) and an optional but recommended version (here: v0). It may have also be registered as `GridWorld-v0` (the recommended approach), `GridWorld` or `gymnasium_env/GridWorld`, and the appropriate ID should then be used during environment creation.
>  环境 ID 包含三部分：可选的命名空间、必须的名字、可选的版本号，在使用 `make()` 时，传入的就是环境 ID

The entry point can be a string or function, as this tutorial isn’t part of a python project, we cannot use a string but for most environments, this is the normal way of specifying the entry point.
>  环境的入口点可以是字符串或函数

Register has additionally parameters that can be used to specify keyword arguments to the environment, e.g., if to apply a time limit wrapper, etc. See [`gymnasium.register()`](https://gymnasium.farama.org/api/registry/#gymnasium.register "gymnasium.register") for more information.
>  `gym.register()` 注册环境

```python
gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldEnv,
)
```

For a more complete guide on registering a custom environment (including with a string entry point), please read the full [create environment](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/) tutorial.

Once the environment is registered, you can check via [`gymnasium.pprint_registry()`](https://gymnasium.farama.org/api/registry/#gymnasium.pprint_registry "gymnasium.pprint_registry") which will output all registered environment, and the environment can then be initialized using [`gymnasium.make()`](https://gymnasium.farama.org/api/registry/#gymnasium.make "gymnasium.make"). A vectorized version of the environment with multiple instances of the same environment running in parallel can be instantiated with [`gymnasium.make_vec()`](https://gymnasium.farama.org/api/registry/#gymnasium.make_vec "gymnasium.make_vec").
>  `pprint_registry()` 可以用于检查环境是否注册成功
>  `make_vec()` 可以用于创建向量化的环境，它包含了多个并行运行的环境

```
import gymnasium as gym
>>> gym.make("gymnasium_env/GridWorld-v0")
<OrderEnforcing<PassiveEnvChecker<GridWorld<gymnasium_env/GridWorld-v0>>>>
>>> gym.make("gymnasium_env/GridWorld-v0", max_episode_steps=100)
<TimeLimit<OrderEnforcing<PassiveEnvChecker<GridWorld<gymnasium_env/GridWorld-v0>>>>>
>>> env = gym.make("gymnasium_env/GridWorld-v0", size=10)
>>> env.unwrapped.size
10
>>> gym.make_vec("gymnasium_env/GridWorld-v0", num_envs=3)
SyncVectorEnv(gymnasium_env/GridWorld-v0, num_envs=3)
```

## Using Wrappers
Oftentimes, we want to use different variants of a custom environment, or we want to modify the behavior of an environment that is provided by Gymnasium or some other party. Wrappers allow us to do this without changing the environment implementation or adding any boilerplate code. Check out the [wrapper documentation](https://gymnasium.farama.org/api/wrappers/) for details on how to use wrappers and instructions for implementing your own. In our example, observations cannot be used directly in learning code because they are dictionaries. However, we don’t actually need to touch our environment implementation to fix this! We can simply add a wrapper on top of environment instances to flatten observations into a single array:
>  wrapper 可以被添加到环境中，我们为环境添加 `FlattenObservation` wrapper，用于将观测展平为数组

```
>>> from gymnasium.wrappers import FlattenObservation

>>> env = gym.make('gymnasium_env/GridWorld-v0')
>>> env.observation_space
Dict('agent': Box(0, 4, (2,), int64), 'target': Box(0, 4, (2,), int64))
>>> env.reset()
({'agent': array([4, 1]), 'target': array([2, 4])}, {'distance': 5.0})
>>> wrapped_env = FlattenObservation(env)
>>> wrapped_env.observation_space
Box(0, 4, (4,), int64)
>>> wrapped_env.reset()
(array([3, 0, 2, 1]), {'distance': 2.0})
```

# Recording Agents
During training or when evaluating an agent, it may be interesting to record agent behaviour over an episode and log the total reward accumulated. This can be achieved through two wrappers: `RecordEpisodeStatistics` and `RecordVideo`, the first tracks episode data such as the total rewards, episode length and time taken and the second generates mp4 videos of the agents using the environment renderings.
>  `RecordEpisodeStatistics` 和 `RecordVideo` 可以用于记录 agent 在回合中的行为，并且记录 agent 获取的总奖励
>  `RecordEpisodeStatistics` 追踪回合数据，包括了总奖励、回合长度、时间等，`RecordVideo` 使用环境渲染生成 agent 的行为的视频展示

We show how to apply these wrappers for two types of problems; the first for recording data for every episode (normally evaluation) and second for recording data periodically (for normal training).

## Recording Every Episode
Given a trained agent, you may wish to record several episodes during evaluation to see how the agent acts. Below we provide an example script to do this with the `RecordEpisodeStatistics` and `RecordVideo`.

```python
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

num_eval_episodes = 4

env = gym.make("CartPole-v1", render_mode="rgb_array")  # replace with your environment
env = RecordVideo(env, video_folder="cartpole-agent", name_prefix="eval",
                  episode_trigger=lambda x: True)
env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

for episode_num in range(num_eval_episodes):
    obs, info = env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()  # replace with actual agent
        obs, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated
env.close()

print(f'Episode time taken: {env.time_queue}')
print(f'Episode total rewards: {env.return_queue}')
print(f'Episode lengths: {env.length_queue}')
```

In the script above, for the `RecordVideo` wrapper, we specify three different variables: `video_folder` to specify the folder that the videos should be saved (change for your problem), `name_prefix` for the prefix of videos themselves and finally an `episode_trigger` such that every episode is recorded. This means that for every episode of the environment, a video will be recorded and saved in the style “cartpole-agent/eval-episode-x.mp4”.
>  初始化 `RecordViode` wrapper 时，我们指定了 `video_folder` 表示存储视频的目录，`name_prefix` 表示视频的前缀，`episode_trigger` 使得每个回合都被记录

For the `RecordEpisodeStatistics`, we only need to specify the buffer lengths, this is the max length of the internal `time_queue`, `return_queue` and `length_queue`. Rather than collect the data for each episode individually, we can use the data queues to print the information at the end of the evaluation.
>  初始化 `RecordEpisodeStatistics` wrapper 时，我们指定缓存长度即可，它表示了 `time_queue, reture_queue, length_queue` 的最大长度，每个回合结束后，我们都可以利用这些队列打印出信息

For speed ups in evaluating environments, it is possible to implement this with vector environments in order to evaluate `N` episodes at the same time in parallel rather than series.

## Recording the Agent during Training
During training, an agent will act in hundreds or thousands of episodes, therefore, you can’t record a video for each episode, but developers might still want to know how the agent acts at different points in the training, recording episodes periodically during training. While for the episode statistics, it is more helpful to know this data for every episode. The following script provides an example of how to periodically record episodes of an agent while recording every episode’s statistics (we use the python’s logger but [tensorboard](https://www.tensorflow.org/tensorboard), [wandb](https://docs.wandb.ai/guides/track) and other modules are available).
>  训练时，agent 会进行成百上千个回合，因此不可能为每个回合记录视频
>  为了了解 agent 在训练时的信息，我们可以周期性记录视频，同时记录每个回合的日志

```python
import logging

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

training_period = 250  # record the agent's episode every 250
num_training_episodes = 10_000  # total number of training episodes

env = gym.make("CartPole-v1", render_mode="rgb_array")  # replace with your environment
env = RecordVideo(env, video_folder="cartpole-agent", name_prefix="training",
                  episode_trigger=lambda x: x % training_period == 0)
env = RecordEpisodeStatistics(env)

for episode_num in range(num_training_episodes):
    obs, info = env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()  # replace with actual agent
        obs, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated

    logging.info(f"episode-{episode_num}", info["episode"])
env.close()
```

# Speeding Up Training
Reinforcement Learning can be a computationally difficult problem that is both sample inefficient and difficult to scale to more complex environments. In this page, we are going to talk about general strategies for speeding up training: vectorizing environments, optimizing training and algorithmic heuristics.

## Vectorized environments
Normally in training, agents will sample from a single environment limiting the number of steps (samples) per second to the speed of the environment. Training can be substantially increased through acting in multiple environments at the same time, referred to as vectorized environments where multiple instances of the same environment run in parallel (on multiple CPUs). Gymnasium provide two built in classes to vectorize most generic environments: [`gymnasium.vector.SyncVectorEnv`](https://gymnasium.farama.org/api/vector/sync_vector_env/#gymnasium.vector.SyncVectorEnv "gymnasium.vector.SyncVectorEnv") and [`gymnasium.vector.AsyncVectorEnv`](https://gymnasium.farama.org/api/vector/async_vector_env/#gymnasium.vector.AsyncVectorEnv "gymnasium.vector.AsyncVectorEnv") which can be easily created with [`gymnasium.make_vec()`](https://gymnasium.farama.org/api/registry/#gymnasium.make_vec "gymnasium.make_vec").
>  为了加速训练，可以让 agent 在多个环境中同时决策，在向量化的环境中，一个环境的多个实例会并行运行，Gymnasium 中相关的类有 `SyncVectorEnv, AsyncAvectorEnv` ，它们可以由 `make_vec()` 创建

It should be noted that vectorizing environments might require changes to your training algorithm and can cause instability in training for very large numbers of sub-environments.
>  注意向量化环境可能需要改变训练算法，否则可能会导致训练不稳定

## Optimizing training
Speeding up training can generally be achieved through optimizing your code, in particular, for deep reinforcement learning that use GPUs in training through the need to transfer data to and from RAM and the GPU memory.

For code written in PyTorch and Jax, they provide the ability to `jit` (just in time compilation) the code order for CPU, GPU and TPU (for jax) to decrease the training time taken.

## Algorithmic heuristics
Academic researchers are consistently explore new optimizations to improve agent performance and reduce the number of samples required to train an agent. In particular, sample efficient reinforcement learning (高效采样的 RL) is a specialist sub-field of reinforcement learning that explores optimizations for training algorithms and environment heuristics that reduce the number of agent observation need for an agent to maximise performance (减少 agent 达到最优表现所需要的观测数量) . As the field is consistently improving, we refer readers to find to survey papers and the latest research to know what the most efficient algorithmic improves that exist currently.
