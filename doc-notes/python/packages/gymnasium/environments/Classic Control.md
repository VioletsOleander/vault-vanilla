# Cart Pole

[![../../../_images/cart_pole.gif](https://gymnasium.farama.org/_images/cart_pole.gif)](https://gymnasium.farama.org/_images/cart_pole.gif)

This environment is part of the Classic Control environments which contains general information about the environment.

| | |
|---|---|
|Action Space|`Discrete(2)`|
|Observation Space|`Box([-4.8 -inf -0.41887903 -inf], [4.8 inf 0.41887903 inf], (4,), float32)`|
|import|`gymnasium.make("CartPole-v1")`|

## Description
This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in [“Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem”](https://ieeexplore.ieee.org/document/6313077). A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart.

## Action Space
The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction of the fixed force the cart is pushed with.

- 0: Push cart to the left
- 1: Push cart to the right

>  action 是 shape 为 `(1,)` 的 `ndarray` ，值为 `0` 表示向左施加固定的力，值为 `1` 表示向右施加固定的力

**Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it
>  施加的力对小车的速度影响不是固定的，它取决于杆子的指向角度，杆子的中心会改变下方小车移动所需的能量值

## Observation Space
The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

|Num|Observation|Min|Max|
|---|---|---|---|
|0|Cart Position|-4.8|4.8|
|1|Cart Velocity|-Inf|Inf|
|2|Pole Angle|~ -0.418 rad (-24°)|~ 0.418 rad (24°)|
|3|Pole Angular Velocity|-Inf|Inf|

>  observation 是 shape `(4,)` 的 `ndarray` 包含了小车的位置、速度和杆的角度、角速度信息

**Note:** While the ranges above denote the possible values for observation space of each element, it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:

- The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates if the cart leaves the `(-2.4, 2.4)` range.
- The pole angle can be observed between `(-.418, .418)` radians (or **±24°**), but the episode terminates if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

## Rewards
Since the goal is to keep the pole upright for as long as possible, by default, a reward of `+1` is given for every step taken, including the termination step. The default reward threshold is 500 for v1 and 200 for v0 due to the time limit on the environment.

If `sutton_barto_reward=True`, then a reward of `0` is awarded for every non-terminating step and `-1` for the terminating step. As a result, the reward threshold is 0 for v0 and v1.

## Starting State
All observations are assigned a uniformly random value in `(-0.05, 0.05)`

## Episode End
The episode ends if any one of the following occurs:

1. Termination: Pole Angle is greater than ±12°
2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
3. Truncation: Episode length is greater than 500 (200 for v0)

## Arguments
Cartpole only has `render_mode` as a keyword for `gymnasium.make`. On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

```
>>> import gymnasium as gym
>>> env = gym.make("CartPole-v1", render_mode="rgb_array")
>>> env
<TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>
>>> env.reset(seed=123, options={"low": -0.1, "high": 0.1})  # default low=-0.05, high=0.05
(array([ 0.03647037, -0.0892358 , -0.05592803, -0.06312564], dtype=float32), {})
```

|Parameter|Type|Default|Description|
|---|---|---|---|
|`sutton_barto_reward`|**bool**|`False`|If `True` the reward function matches the original sutton barto implementation|

## Vectorized environment
To increase steps per seconds, users can use a custom vector environment or with an environment vectorizor.

```
>>> import gymnasium as gym
>>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="vector_entry_point")
>>> envs
CartPoleVectorEnv(CartPole-v1, num_envs=3)
>>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
>>> envs
SyncVectorEnv(CartPole-v1, num_envs=3)
```

## Version History
- v1: `max_time_steps` raised to 500.
    - In Gymnasium `1.0.0a2` the `sutton_barto_reward` argument was added (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/790))
- v0: Initial versions release.