from setuptools import find_packages, setup

setup(
    name="miniclip",
    author="Denis Mazur",
    packages=find_packages(exclude=["venv", "test"]),
)
