from setuptools import setup

setup(name='amlc',
      python_requires='>3.5',
      version='1.0',
      description='Analytic marginalization over linear continuum parameters',
      author='Kirill Tchernyshyov',
      author_email='ktcherny@gmail.com',
      url='https://github.com/ktchrn/amlc',
      license='MIT',
      packages=['amlc'],
      install_requires=['numpy','scipy']
      )
