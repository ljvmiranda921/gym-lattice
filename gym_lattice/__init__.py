from gym.envs.registration import register

register(
    id='lattice2d-v0',
    entry_point='gym_lattice.envs:Lattice2DEnv',
)