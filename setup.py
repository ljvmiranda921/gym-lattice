from setuptools import setup

requirements = ["gym==0.23.1", "numpy==1.22.3"]

dev_requirements = ["pytest==7.1.1", "tox==3.24.5"]

setup(
    name="gym_lattice",
    version="0.0.2",
    install_requires=requirements,
    description="An HP 2D Lattice Gym Environment for Protein Folding",
    author="Lester James V. Miranda",
    author_email="ljvmiranda@gmail.com",
    url="https://github.com/ljvmiranda921/gym-lattice",
    test_require=dev_requirements,
)
