#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Implements the 2D Lattice Environment
"""

# Import modules
import numpy as np
from collections import OrderedDict


class Lattice2D(object):
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
        self.chain = OrderedDict({(0,0):self.seq[0]})
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
        ValueError
            When the specified action is invalid.
        """
        # Obtain the last coordinate in the chain
        x, y = next(reversed(self.chain))
        # Get new coords based on action
        new_coords = self._add_poly((x,y), action)
        # Detects for collision in the given coordinate
        idx = len(self.chain)
        if new_coords in self.chain:
            self.collisions += 1
        else:
            self.chain.update({new_coords : self.seq[idx]})
        # Done signal
        done = True if len(self.chain) == len(self.seq) else False
        # Compute for reward
        reward = self._get_reward(self.chain) if done else None
        # Organize info
        info = {
            'chain length' : len(self.chain),
            'seq length'   : len(self.seq),
            'collisions'   : self.collisions
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