Metallic
========

`Metallic <https://github.com/Renovamen/metallic>`_ is a minimal library for meta-learning research based on `PyTorch <https://github.com/pytorch/pytorch>`_ and `higher <https://github.com/facebookresearch/higher>`_.

Different from other meta-learning libraries (like `learn2learn <https://github.com/learnables/learn2learn>`_ and `Torchmeta <https://github.com/tristandeleu/pytorch-meta>`_), Metallic aims at providing an lightweight framework with the least amount of code for setting up meta-learning experiments quickly.

This library is **work in progress**.


Installation
------------

.. code-block:: bash

    git clone https://github.com/Renovamen/metallic.git
    cd metallic
    python setup.py install

or

.. code-block:: bash

    pip install git+https://github.com/Renovamen/metallic.git --upgrade


Github
------------

https://github.com/Renovamen/metallic



.. toctree::
   :maxdepth: 1
   :caption: API Documentation

   api/data
   api/data.datasets
   api/data.benchmarks
   api/models
   api/transforms


Acknowledgements
----------------

Metallic is highly inspired by the following awesome libraries:

- `learn2learn <https://github.com/learnables/learn2learn>`_
- `Torchmeta <https://github.com/tristandeleu/pytorch-meta>`_


License
-------

Metallic is MIT licensed, see the `LICENSE <https://github.com/Renovamen/metallic/blob/master/LICENSE>`_ file for more details.
