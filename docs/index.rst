Metallic
========

`Metallic <https://github.com/Renovamen/metallic>`_ is a library for meta-learning research based on `PyTorch <https://github.com/pytorch/pytorch>`_.

Different from other meta-learning libraries, Metallic tries to:

- Provide a clean, lightweight and modularized toolbox for setting up meta-learning experiments quickly with the least amount of code.
- Support more optimizers instead of SGD only using `higher <https://github.com/facebookresearch/higher>`_.

The library is **work in progress**.


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
   api/data.transforms
   api/models
   api/metalearners
   api/trainer
   api/utils


Acknowledgements
----------------

Metallic is highly inspired by the following awesome libraries:

- `learn2learn <https://github.com/learnables/learn2learn>`_
- `Torchmeta <https://github.com/tristandeleu/pytorch-meta>`_
- `higher <https://github.com/facebookresearch/higher>`_


License
-------

Metallic is MIT licensed, see the `LICENSE <https://github.com/Renovamen/metallic/blob/master/LICENSE>`_ file for more details.
