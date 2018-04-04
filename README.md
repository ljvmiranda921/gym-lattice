# Lattice

![python 3.5+](https://img.shields.io/badge/python-3.5+-blue.svg)

An HP 2D Lattice Environment with a Gym-like API for the protein folding
problem

This is a Python library that implements Lau and Dill's (1989)
hydrophobic-polar two-dimensional lattice model for protein structures. It
follows OpenAI Gym's API, easing integration for reinforcement learning
solutions.

## Dependencies

- numpy==1.14.2

## Usage

There are four possible actions in this environment: `L` (left), `R` (right),
`U` (up), and `D` (down). Let's try this out with a random agent on the
protein sequence `HHPHH`!

```python
from lattice.environments import Lattice2D
import numpy as np

np.random.seed(42)

seq = 'HHPHH'
action_ space = ['L', 'R', 'U', 'D']
env = Lattice2D(seq)

for i_episodes in range(5):
    env.reset()
    while True
        # Sample randomly from action space
        action = np.random.choice(action_space)
        obs, reward, done, info = env.step(action)
        if done:
            print("Episode finished! Reward: {} | Collisions: {} | Actions: {}".format(reward, info['collisions']))
```

Sample output:

```
Episode finished! Reward: -2 | Collisions: 1 | Actions: ['U', 'L', 'U', 'U']
Episode finished! Reward: -2 | Collisions: 0 | Actions: ['D', 'L', 'L', 'U']
Episode finished! Reward: -2 | Collisions: 0 | Actions: ['R', 'U', 'U', 'U']
Episode finished! Reward: -3 | Collisions: 1 | Actions: ['U', 'L', 'D', 'D']
Episode finished! Reward: -2 | Collisions: 2 | Actions: ['D', 'R', 'R', 'D']
```

An episode ends when all polymers are added to the lattice. Whenever a
collision is detected, the agent should enter another action. We account for
the number of collisions so you can use them when you customize your own
reward functions.

In addition, this environment gives **sparse rewards**, that is, reward is
only computed at the end of each episode.

## Task List
- Add test cases and set-up continuous integration
- Rendering fuctionality
- Lattice3D (?) *maybe some time in the future*
