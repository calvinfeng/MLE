from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='distributions',
      version='0.0.1',
      description='Gaussian distributions',
      long_description=long_description,
      author="Calvin Feng",
      author_email="calvin.j.feng@gmail.com",
      packages=['distributions'],
      install_requires=['numpy', 'matplotlib'],
      python_requires='>=3.6',
      zip_safe=False,
)