# -*- coding: utf-8 -*-

"""
Implements the 2D Lattice Environment
"""
# Import gym modules
import sys
from collections import OrderedDict

import gym
import numpy as np
from gym import (spaces, utils, logger)
from six import StringIO

# Human-readable
ACTION_TO_STR = {
    0 : 'L', 1 : 'D',
    2 : 'U', 3 : 'R'}

POLY_TO_INT = {
    'H' : 1, 'P' : -1
}

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
    grid_length : int
        Length of one side of the grid.
    midpoint : tuple
        Coordinate containing the midpoint of the grid.
    grid : numpy.ndarray
        Actual grid containing the polymer chain.

    .. [dill1989lattice] Lau, K.F., Dill, K.A.: A lattice statistical
    mechanics model of the conformational and se quence spaces of proteins.
    Marcromolecules 22(10), 3986–3997 (1989)
    """
    metadata = {'render.modes': ['human', 'ansi']}

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
        self.state = OrderedDict({(0, 0) : self.seq[0]})
        self.actions = []
        self.collisions = 0
        self.trapped = 0
        # Grid attributes
        self.grid_length = 2 * len(seq) + 1
        self.midpoint = (len(seq), len(seq))
        self.grid = np.zeros(shape=(self.grid_length, self.grid_length), dtype=int)
        # Automatically assign first element into grid
        self.grid[self.midpoint] = POLY_TO_INT[self.seq[0]]
        # Define action-observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-2, high=1,
                                            shape=(self.grid_length, self.grid_length),
                                            dtype=int)
        self.last_action = None

    def step(self, action):
        """Updates the current chain with the specified action.

        The action supplied by the agent should be an integer from 0
        to 3. In this case:
            - 0 : left
            - 1 : down
            - 2 : up
            - 3 : right
        The best way to remember this is to note that they are similar to the
        'h', 'j', 'k', and 'l' keys in vim.

        This method returns a set of values similar to the OpenAI gym, that
        is, a tuple :code:`(observations, reward, done, info)`.

        The observations are arranged as a :code:`numpy.ndarray` matrix, more
        suitable for agents built using convolutional neural networks. The
        'H' is represented as :code:`1`s whereas the 'P's as :code:`-1`s.
        However, for the actual chain, that is, an :code:`OrderedDict` and
        not its grid-like representation, can be accessed from
        :code:`info['state_chain]`.

        The reward is calculated at the end of every episode, that is, when
        the length of the chain is equal to the length of the input sequence.

        Parameters
        ----------
        action : int, {0, 1, 2, 3}
            Specifies the position where the next polymer will be placed
            relative to the previous one:
                - 0 : left
                - 1 : down
                - 2 : up
                - 3 : right

        Returns
        -------
        numpy.ndarray
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
        IndexError
            When :code:`step()` is still called even if done signal
            is already :code:`True`.
        """
        assert self.action_space.contains(action), logger.error("%r (%s) invalid"%(action, type(action)))

        self.last_action = action
        is_trapped = False
        # Obtain coordinate of previous polymer
        x, y = next(reversed(self.state))
        # Get all adjacant coords and next move based on action
        adj_coords = self._get_adjacent_coords((x, y))
        next_move = adj_coords[action]
        # Detects for collision or traps in the given coordinate
        idx = len(self.state)
        if set(adj_coords.values()).issubset(self.state):
            logger.info('Your agent was trapped! Ending the episode.')
            self.trapped += 1
            is_trapped = True
        elif next_move in self.state:
            self.collisions += 1
        else:
            self.actions.append(action)
            try:
                self.state.update({next_move : self.seq[idx]})
            except IndexError:
                logger.error('All molecules have been placed! Nothing can be added to the protein chain.')
                raise


        # Set-up return values
        grid = self._draw_grid(self.state)
        done = True if len(self.state) == len(self.seq) or is_trapped else False
        reward = self._compute_free_energy(self.state) if done else None
        info = {
            'chain_length' : len(self.state),
            'seq_length'   : len(self.seq),
            'collisions'   : self.collisions,
            'actions'      : [ACTION_TO_STR[i] for i in self.actions],
            'is_trapped'   : is_trapped,
            'state_chain'  : self.state
        }

        return (grid, reward, done, info)

    def reset(self):
        """Resets the environment"""
        self.state = OrderedDict({(0,0) : self.seq[0]})
        self.actions = []
        self.collisions = 0
        self.trapped = 0
        self.grid = np.zeros(shape=(self.grid_length, self.grid_length), dtype=int)
        # Automatically assign first element into grid
        self.grid[self.midpoint] = POLY_TO_INT[self.seq[0]]

        return self.grid

    def render(self, mode='human'):
        """Renders the environment"""

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        desc = self.grid.astype(str)

        # Convert everything to human-readable symbols
        desc[desc == '0'] = '*'
        desc[desc == '1'] = 'H'
        desc[desc == '-1'] = 'P'

        # Obtain all x-y indices of elements
        x_free, y_free = np.where(desc == '*')
        x_h, y_h = np.where(desc == 'H')
        x_p, y_p = np.where(desc == 'P')

        # Decode if possible
        desc.tolist()
        try:
            desc = [[c.decode('utf-8') for c in line] for line in desc]
        except AttributeError:
            pass

        # All unfilled spaces are gray
        for xy in zip(x_free, y_free):
            desc[xy] = utils.colorize(desc[xy], "gray")

        # All hydrophobic molecules are bold-green
        for xy in zip(x_h, y_h):
            desc[xy] = utils.colorize(desc[xy], "green", bold=True)

        # All polar molecules are cyan
        for xy in zip(x_p, y_p):
            desc[xy] = utils.colorize(desc[xy], "cyan")

        # Provide prompt for last action
        if self.last_action is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Up", "Right"][self.last_action]))
        else:
            outfile.write("\n")

        # Draw desc
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile

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
            0 : (x - 1, y),
            1 : (x, y - 1),
            2 : (x, y + 1),
            3 : (x + 1, y),
        }

        return adjacent_coords

    def _draw_grid(self, chain):
        """Constructs a grid with the current chain

        Parameters
        ----------
        chain : OrderedDict
            Current chain/state

        Returns
        -------
        numpy.ndarray
            Grid of shape :code:`(n, n)` with the chain inside
        """
        for coord, poly in chain.items():
            trans_x, trans_y = tuple(sum(x) for x in zip(self.midpoint, coord))
            # Recall that a numpy array works by indexing the rows first
            # before the columns, that's why we interchange.
            self.grid[(trans_y, trans_x)] = POLY_TO_INT[poly]

        return np.flipud(self.grid)

    def _compute_free_energy(self, chain):
        """Computes the Gibbs free energy given the lattice's state

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
            Computed free energy
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