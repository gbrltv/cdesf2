from setuptools import setup, find_packages

# TODO: Add more meta-data (keywords, classifiers...)
setup(
  name="cdesf2",
  version="0.0.1",
  description="Concept-drift on Event Stream Framework",
  url="https://github.com/gbrltv/cdesf2",
  author="Gabriel Marques Tavares",
  package_dir={'': 'cdesf2'},
  packages=find_packages(where="cdesf2"),
  install_requires=[
    "networkx==2.5",
    "numpy==1.20.*",
    "pandas==1.2.*",
    "scipy==1.6.*",
    "matplotlib==3.3.*",
    "seaborn==0.11.*",
    "pydot==1.4.*",
    "pydot-ng==2.0.*",
    "pygraphviz==1.7",
    "pm4py==2.2.*"
  ]
)
