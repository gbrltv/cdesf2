CDESF: Concept-drift on Event Stream Framework
=========================================================================================
|github-ci| |pip| |downloads| |sonar_quality| |sonar_maintainability| |codacy|
|code_climate_maintainability|

Latest version of CDESF framework

How do I install this package?
----------------------------------------------
Clone the repository into your local machine:

.. code:: console

    pip install cdesf

Make sure you have the `graphviz` and `libgraphviz-dev` installed on your system. These dependencies are needed to
generate the output graphs.

.. code:: console

    $ sudo apt-get install graphviz graphviz-dev

For more detailed installation instructions please refer to the 
`installation guide <https://github.com/pygraphviz/pygraphviz/blob/main/INSTALL.txt>`__ on the `pygraphviz` repository.

Tests Coverage
----------------------------------------------
Since some software handling coverages sometimes
get slightly different results, here's three of them:

|coveralls| |sonar_coverage| |code_climate_coverage|

Reference
----------------------------------------------

Please, use “Overlapping Analytic Stages in Online Process Mining”
(https://ieeexplore.ieee.org/abstract/document/8813959) for reference.


.. code-block:: bibtex

    @INPROCEEDINGS{8813959,
    author={G. M. {Tavares} and P. {Ceravolo} and V. G. {Turrisi Da Costa}
    and E. {Damiani} and S. {Barbon Junior}},
    booktitle={2019 IEEE International Conference on Services Computing
    (SCC)},
    title={Overlapping Analytic Stages in Online Process Mining},
    year={2019},
    volume={},
    number={},
    pages={167-175}}

.. |github-ci| image:: https://github.com/gbrltv/cdesf2/workflows/CI/badge.svg?branch=master
   :alt: GitHub CI build

.. |sonar_quality| image:: https://sonarcloud.io/api/project_badges/measure?project=gbrltv_CDESF2&metric=alert_status
    :target: https://sonarcloud.io/dashboard/index/gbrltv_CDESF2
    :alt: SonarCloud Quality

.. |sonar_maintainability| image:: https://sonarcloud.io/api/project_badges/measure?project=gbrltv_CDESF2&metric=sqale_rating
    :target: https://sonarcloud.io/dashboard/index/gbrltv_CDESF2
    :alt: SonarCloud Maintainability

.. |sonar_coverage| image:: https://sonarcloud.io/api/project_badges/measure?project=gbrltv_CDESF2&metric=coverage
    :target: https://sonarcloud.io/dashboard/index/gbrltv_CDESF2
    :alt: SonarCloud Coverage

.. |coveralls| image:: https://coveralls.io/repos/github/gbrltv/CDESF2/badge.svg?branch=master
    :target: https://coveralls.io/github/gbrltv/CDESF2?branch=master
    :alt: Coveralls Coverage

.. |pip| image:: https://img.shields.io/pypi/v/cdesf
    :target: https://pypi.org/project/cdesf/
    :alt: PyPI project

.. |downloads| image:: https://img.shields.io/pypi/dm/cdesf
    :target: https://pypi.org/project/cdesf/
    :alt: PyPI total project downloads

.. |codacy| image:: https://app.codacy.com/project/badge/Grade/832aa5a76fc649b9ad3586e5e19709b4
    :target: https://www.codacy.com/manual/gbrltv/CDESF2?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=gbrltv/CDESF2&amp;utm_campaign=Badge_Grade
    :alt: Codacy Maintainability

.. |code_climate_maintainability| image:: https://api.codeclimate.com/v1/badges/9fceda1f4665e4a1596f/maintainability
    :target: https://codeclimate.com/github/gbrltv/CDESF2/maintainability
    :alt: Maintainability

.. |code_climate_coverage| image:: https://api.codeclimate.com/v1/badges/9fceda1f4665e4a1596f/test_coverage
    :target: https://codeclimate.com/github/gbrltv/CDESF2/test_coverage
    :alt: Code Climate Coverage
