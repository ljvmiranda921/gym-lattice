from gym.envs.registration import register

register(
    id='Lattice2D-v0',
    entry_point='gym_lattice.envs:Lattice2DEnv',
)