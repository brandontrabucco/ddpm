from setuptools import find_packages
from setuptools import setup


setup(name='ddpm', version='1.0', license='MIT',
      packages=find_packages(include=['ddpm', 'ddpm.*']),
      install_requires=['einops', 'numpy', 'torch'],
      author='Brandon Trabucco',
      author_email='brandon@btrabucco.com',
      url='https://github.com/brandontrabucco/ddpm',
      keywords=['Deep Learning', 'Neural Networks',
                'Denoising Diffusion Probabilistic Models'])