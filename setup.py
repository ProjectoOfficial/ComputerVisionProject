from setuptools import setup, find_packages

setup(
    name='ComputerVisionProject', 
    version='1.0.0', 
    packages=find_packages(),
    author="Daniel Rossi, Riccardo Salami, Filippo Ferrari",
    author_email='miniprojectsofficial@gmail.com',
    install_requires=["opencv-python", "numpy", "pynput", "torch"],
    url="https://github.com/ProjectoOfficial/ComputerVisionProject"
    )