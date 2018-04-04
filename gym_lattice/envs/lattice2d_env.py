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
    chain : OrderedDict
        Dictionary of polymer coordinates.
    actions : list
        List of actions performed by the model.
    collisions : int
        Number of collisions incurred by the model.

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
        self.chain = OrderedDict({(0,0) : self.seq[0]})
        self.actions = []
        self.collisions = 0

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

        # Obtain the last coordinate in the chain
        x, y = next(reversed(self.chain))
        # Get new coords based on action
        new_coords = self._add_poly((x,y), action)
        # Detects for collision in the given coordinate
        idx = len(self.chain)
        if new_coords in self.chain:
            self.collisions += 1
        else:
            self.actions.append(action)
            self.chain.update({new_coords : self.seq[idx]})
        # Done signal
        done = True if len(self.chain) == len(self.seq) else False
        # Compute for reward
        reward = self._get_reward(self.chain) if done else None
        # Organize info
        info = {
            'chain_length' : len(self.chain),
            'seq_length'   : len(self.seq),
            'collisions'   : self.collisions,
            'actions'      : self.actions
        }

        return (self.chain, reward, done, info)

    def reset(self):
        """Resets the environment"""
        self.chain = OrderedDict({(0,0) : self.seq[0]})
        self.actions = []
        self.collisions = 0

    def _add_poly(self, coords, action):
        """Obtains the coordinate of the next polymer
        
        Parameters
        ----------
        coords : tuple
            X-y coordinates of the previous polymer.
        action : str
            Action to be done.

        Returns
        -------
        tuple
            Coordinates of the next polymer.
        """
        x, y = coords
        if action == 'L':
            new_coords = (x - 1, y)
        elif action == 'R':
            new_coords = (x + 1, y)
        elif action == 'U':
            new_coords = (x, y + 1)
        elif action == 'D':
            new_coords = (x, y - 1)
        return new_coords

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