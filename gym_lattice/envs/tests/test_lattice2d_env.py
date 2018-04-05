# -*- coding: utf-8 -*-

"""Test cases for Lattice2DEnv"""

import pytest
import string, random

def generate_sequence(length):
    """Generates a random sequence given a length"""
    possible_chars = 'HHpp'
    return "".join([random.choice(possible_chars) for i in range(length)])

def generate_invalid_sequence():
    """Generates an invalid sequence of length 10"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

# We use the ten benchmark lengths by PNS
@pytest.fixture(params=[20, 24, 25, 36, 48, 
                        50, 60, 64, 85, 100])
def lattice2d_env(request):
    """Returns a Lattice2DENv class with as sequence
    initialized at different lengths"""
    from gym_lattice.envs import Lattice2DEnv
    seq = generate_sequence(request.param)
    return Lattice2DEnv(seq)

def test_init_invalid_sequence():
    """Exception must be raised with invalid input"""
    with pytest.raises(AssertionError):
        from gym_lattice.envs import Lattice2DEnv
        seq = generate_invalid_sequence()
        Lattice2DEnv(seq)

def test_get_adjacent_coords(lattice2d_env):
    """Tests private method _get_adjacent_coords()"""
    pass