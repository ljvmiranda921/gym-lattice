# -*- coding: utf-8 -*-

"""Test cases for Lattice2DEnv"""

import pytest
import numpy as np
import string

def generate_sequence(length):
    """Generates a random sequence given a length"""
    possible_chars = 'HhPp'
    return "".join([np.random.choice(list(possible_chars)) for i in range(length)])

def generate_invalid_sequence():
    """Generates an invalid sequence of length 10"""
    return ''.join(np.random.choice(list(string.ascii_uppercase + string.digits), size=10))

def test_init_invalid_sequence():
    """Exception must be raised with invalid input"""
    with pytest.raises(AssertionError):
        from gym_lattice.envs import Lattice2DEnv
        seq = generate_invalid_sequence()
        Lattice2DEnv(seq)

def test_get_adjacent_coords():
    """Tests private method _get_adjacent_coords()"""
    from gym_lattice.envs import Lattice2DEnv
    seq = generate_sequence(10)
    env = Lattice2DEnv(seq)
    test_coords = (0,0)
    result = env._get_adjacent_coords(test_coords)
    expected =  {
        0 : (-1,0), 1 : (0,-1), 
        2 : (0,1),  3 : (1,0)}
    assert result == expected

@pytest.mark.parametrize("actions,expected_coords", 
    [(0, (2,1)), (1, (3,2)), (2, (1,2)), (3, (2,3))])
def test_draw_grid(actions, expected_coords):
    """Tests private method _draw_grid()"""
    from gym_lattice.envs import Lattice2DEnv
    seq = 'HH'
    env = Lattice2DEnv(seq)
    result = env._draw_grid(env.state)
    env.step(actions)
    expected = env.grid
    expected[expected_coords] = 1
    assert np.array_equal(expected, result)

@pytest.mark.parametrize("seq,actions,expected",
    [('HH', [0], -1), ('HHHH', [0,1,3], -4),
     ('HP', [0], 0),  ('PPPPH', [0,1,2,3], 0)])
def test_compute_free_energy(seq, actions, expected):
    """Tests private method _compute_free_energy()"""
    from gym_lattice.envs import Lattice2DEnv
    env = Lattice2DEnv(seq)
    for action in actions:
        env.step(action)
    result = env._compute_free_energy(env.state)
    assert expected == result

@pytest.fixture
def lattice2d_env():
    from gym_lattice.envs import Lattice2DEnv
    seq = generate_sequence(10)
    return Lattice2DEnv(seq)

@pytest.mark.parametrize("action", [5, -2, 'L', 'F', '2'])
def test_invalid_actions(action, lattice2d_env):
    """Exception must be raised when step() receives invalid action"""
    with pytest.raises(AssertionError):
        lattice2d_env.step(action)

def test_illegal_step_call(lattice2d_env):
    """Exception must be raised when step() is called even if the episode is already done"""
    with pytest.raises(IndexError):
        for random_action in np.random.choice(np.arange(4), size=20):
            lattice2d_env.step(random_action)

def test_done_signal(lattice2d_env):
    """Test if done signal responds at the end
    of an episode"""
    test_actions = np.zeros(shape=(len(lattice2d_env.seq)-1,), dtype=int)
    done = False # Must change to True
    for action in test_actions:
        _, _, done, _ = lattice2d_env.step(action)
    assert done

def test_step_return_values():
    """Test if step() returns a 4-tuple"""
    pass