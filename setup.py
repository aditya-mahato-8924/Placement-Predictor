from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str) -> List[str]:
    """ This function will return the list of requirements mentioned in the requirements.txt file"""

    with open('requirements.txt') as file:
        requirements = file.read().splitlines()
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name="Placement Predictor",
    version="1.0.0",
    description="A machine learning model to predict student placements based on various features.",
    author="Aditya Mahato",
    author_email="adityamahato8924@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)