# -*- coding: utf-8 -*-

"""
Implements the 2D Lattice Environment
"""
# Import gym modules
import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from collections import OrderedDict


class Lattice2DEnv(gym.Env):
    """A 2-dimensional lattice environment from Dill and Lau, 1989
    [dill1989lattice]_.

    It follows an absolute Cartesian coordinate system, the location of
    the polymer is stated independently from one another. Thus, we have
    four actions (left, right, up, and down) and a chance of collision.

    The environment will first place the initial polymer at the origin.
    Then, for each step, agents place another polymer to the lattice. An episode
    ends when all polymers are placed, i.e. when the length of the action
    chain is equal to the length of the input sequence minus 1. We then
    compute the reward using the energy minimization rule.

    Attributes
    ----------
    seq : str
        Polymer sequence describing a particular protein.
    state : OrderedDict
        Dictionary of the current polymer chain with coordinates and
        polymer type (H or P).
    actions : list
        List of actions performed by the model.
    collisions : int
        Number of collisions incurred by the model.
    trapped : int
        Number of times the agent was trapped.

    .. [dill1989lattice] Lau, K.F., Dill, K.A.: A lattice statistical
    mechanics model of the conformational and se quence spaces of proteins.
    Marcromolecules 22(10), 3986–3997 (1989)
    """
    metadata = {'render.modes': ['human']}

    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    def __init__(self, seq):
        """Initializes the lattice

        Parameters
        ----------
        seq : str, must only consist of 'H' or 'P'
            Sequence containing the polymer chain.

        Raises
        ------
        AssertionError
            If a certain polymer is not 'H' or 'P'
        """
        assert set(seq.upper()) <= set('HP'), "Invalid input sequence!"
        self.seq = seq.upper()
        self.state = OrderedDict({(0,0) : self.seq[0]})
        self.actions = []
        self.collisions = 0
        self.trapped = 0

    def step(self, action):
        """Updates the current chain with the specified action.

        This method returns a set of values similar to the OpenAI gym,
        that is, a tuple :code:`(observations, reward, done, info)`.

        The reward is calculated at the end of every episode, that is, when
        the length of the chain is equal to the length of the input sequence.

        Parameters
        ----------
        action : str, {'L', 'R', 'U', 'D'}
            Specifies the position where the next polymer will be placed
            relative to the previous one:
                - 'L' : left
                - 'R' : right
                - 'U' : up
                - 'D' : down

        Returns
        -------
        OrderedDict
            Current state of the lattice.
        int or None
            Reward for the current episode.
        bool
            Control signal when the episode ends.
        dict
            Additional information regarding the environment.

        Raises
        ------
        AssertionError
            When the specified action is invalid.
        """
        assert action in ['L', 'R', 'U', 'D'], "Invalid action specified!"
        is_trapped = False
        # Obtain coordinate of previous polymer
        x, y = next(reversed(self.state))
        # Get all adjacant coords and next move based on action
        adj_coords = self._get_adjacent_coords((x,y))
        next_move = adj_coords[action]
        # Detects for collision or traps in the given coordinate
        idx = len(self.state)
        if set(adj_coords.values()).issubset(self.state):
            self.trapped += 1
            is_trapped = True
        elif next_move in self.state:
            self.collisions += 1
        else:
            self.actions.append(action)
            self.state.update({next_move : self.seq[idx]})
        # Done signal
        done = True if len(self.state) == len(self.seq) or is_trapped else False

        # Compute for reward
        reward = self._get_reward(self.state) if done else None

        # Organize info
        info = {
            'chain_length' : len(self.state),
            'seq_length'   : len(self.seq),
            'collisions'   : self.collisions,
            'actions'      : self.actions,
            'is_trapped'   : is_trapped
        }

        return (self.state, reward, done, info)

    def reset(self):
        """Resets the environment"""
        self.state = OrderedDict({(0,0) : self.seq[0]})
        self.actions = []
        self.collisions = 0
        self.trapped = 0

    def _get_adjacent_coords(self, coords):
        """Obtains all adjacent coordinates of the current position

        Parameters
        ----------
        coords : 2-tuple
            Coordinates (X-y) of the current position

        Returns
        -------
        dictionary
            All adjacent coordinates
        """
        x, y = coords
        adjacent_coords = {
            'L' : (x - 1, y),
            'R' : (x + 1, y),
            'U' : (x, y + 1),
            'D' : (x, y - 1)
        }

        return adjacent_coords

    def _get_reward(self, chain):
        """Computes the reward given the lattice's state

        This environment gives you sparse rewards, i.e., the reward is only
        computed at the end of each episode. This follow the same energy
        function given by Dill et. al. [dill1989lattice]_

        Recall that the goal is to find the configuration with the lowest
        energy.

        .. [dill1989lattice] Lau, K.F., Dill, K.A.: A lattice statistical
        mechanics model of the conformational and se quence spaces of proteins.
        Marcromolecules 22(10), 3986–3997 (1989)

        Parameters
        ----------
        chain : OrderedDict
            Current chain in the lattice

        Returns
        -------
        int
            Computed reward
        """
        h_polymers = [x for x in chain if chain[x] == 'H']
        h_pairs = [(x, y) for x in h_polymers for y in h_polymers]

        # Compute distance between all hydrophobic pairs
        h_adjacent = []
        for pair in h_pairs:
            dist = np.linalg.norm(np.subtract(pair[0], pair[1]))
            if dist == 1.0: # adjacent pairs have a unit distance
                h_adjacent.append(pair)

        # Remove duplicate pairs of pairs
        reward = - len(h_adjacent) / 2
        return int(reward)