# -*- coding: utf-8 -*-

"""Test cases for Lattice2DEnv"""

import string
import pytest
import numpy as np

def generate_sequence(length):
    """Generates a random sequence given a length"""
    possible_chars = 'HhPp'
    return "".join([np.random.choice(list(possible_chars)) for i in range(length)])

def generate_invalid_sequence():
    """Generates an invalid sequence of length 10"""
    return ''.join(np.random.choice(list(string.ascii_uppercase + string.digits), size=10))

@pytest.mark.parametrize("sequence", [generate_invalid_sequence(), 100023493])
def test_init_invalid_sequence(sequence):
    """Exception must be raised with invalid input"""
    with pytest.raises((ValueError, AttributeError)):
        from gym_lattice.envs import Lattice2DEnv
        Lattice2DEnv(sequence)

@pytest.mark.parametrize("penalty", [10, 0, -5.2, 'hello'])
def test_init_illegal_collision_penalty(penalty):
    """Exception must be raised when illegal penalty is given"""
    with pytest.raises((ValueError, TypeError)):
        from gym_lattice.envs import Lattice2DEnv
        seq = generate_sequence(10)
        Lattice2DEnv(seq, collision_penalty=penalty)

@pytest.mark.parametrize("penalty", [20, -4, 'hello'])
def test_init_illegal_trap_penalty(penalty):
    """Exception must be raised when illegal penalty is given"""
    with pytest.raises((ValueError, TypeError)):
        from gym_lattice.envs import Lattice2DEnv
        seq = generate_sequence(10)
        Lattice2DEnv(seq, trap_penalty=penalty)

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
    [('HH', [0], 0), ('HHHH', [0,1,3], -1),
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
    """Lattice2DEnv with a random sequence"""
    from gym_lattice.envs import Lattice2DEnv
    seq = generate_sequence(10)
    return Lattice2DEnv(seq)

@pytest.fixture
def lattice2d_fixed_env():
    """Lattice2DEnv with a fixed sequence"""
    from gym_lattice.envs import Lattice2DEnv
    seq = 'HHHH'
    return Lattice2DEnv(seq)

def test_compute_reward_no_penalty(lattice2d_fixed_env):
    """Test reward function with a normal, no-penalty action"""
    for _ in range(3):
        _, reward, _, _ = lattice2d_fixed_env.step(0)
        expected_reward = 0

        assert expected_reward == reward

def test_compute_reward_with_collision(lattice2d_fixed_env):
    """Test reward function with a collision"""
    for i, action in enumerate([0, 2, 1, 0]):
        _, reward, _, _ = lattice2d_fixed_env.step(action)
        if i == 2: # When action == 1, there is collision
            expected_reward = lattice2d_fixed_env.collision_penalty 
            assert expected_reward == reward

def test_compute_reward_with_trap():
    """Test reward function when agent is trapped"""
    from gym_lattice.envs import Lattice2DEnv
    seq = 'H' * 20 # sequence of 20 Hs
    env = Lattice2DEnv(seq)
    expected_reward = 3 - (len(seq) * env.trap_penalty) # (12 bonds) - (20 * 0.5)
    # Define sequence of actions that will trap the agent
    actions = [0, 2, 2, 3, 3, 1, 0, 1]
    for _ , action in enumerate(actions):
        _, reward, done, _ = env.step(action)
        if done:
            assert expected_reward == reward

@pytest.mark.parametrize("action", [5, -2, 'L', 'F', '2'])
def test_invalid_actions(action, lattice2d_env):
    """Exception must be raised when step() receives invalid action"""
    with pytest.raises(ValueError):
        lattice2d_env.step(action)

def test_illegal_step_call(lattice2d_env):
    """Exception must be raised when step() is called even if the episode is already done"""
    nb_of_actions = 100 # overkill to accommodate possible collisions
    with pytest.raises(IndexError):
        for random_action in np.random.choice(np.arange(4), size=nb_of_actions):
            lattice2d_env.step(random_action)

def test_done_signal(lattice2d_env):
    """Test if done signal responds at the end of an episode"""
    test_actions = np.zeros(shape=(len(lattice2d_env.seq)-1,), dtype=int)
    done = False # Must change to True
    for action in test_actions:
        _, _, done, _ = lattice2d_env.step(action)
    assert done
