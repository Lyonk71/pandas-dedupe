from setuptools import setup

setup(name='pandas_dedupe',
      version='0.2',
      description='The Dedupe library made easy with Pandas.',
      # url='http://github.com/storborg/funniest',
      author='Keith Lyons',
      author_email='lyonk71@gmail.com',
      license='MIT',
      packages=['pandas_dedupe'],
      install_requires=[
          'dedupe',
          'unidecode',
          'pandas',
      ],
      zip_safe=False)
