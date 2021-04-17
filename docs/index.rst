########
Metallic
########

`Metallic <https://github.com/Renovamen/metallic>`_ is a meta-learning library based on `PyTorch <https://github.com/pytorch/pytorch>`_.

Different from other meta-learning libraries, Metallic tries to:

- Provide a clean, lightweight and modularized toolbox for setting up meta-learning experiments quickly with the least amount of code.
- For gradient-based meta-learning methods (like MAML), support more optimizers instead of SGD only using `higher <https://github.com/facebookresearch/higher>`_.

The library is **work in progress**.


********
Features
********

Algorithms
==========

The supported interface algorithms currently include:

Gradient-based
--------------

- `Model-Agnostic Meta-Learning (MAML) <https://arxiv.org/abs/1703.03400>`_, including first-order version (FOMAML)
- `Reptile <https://arxiv.org/abs/1803.02999>`_
- `Minibatch Proximal Update <https://panzhous.github.io/assets/pdf/2019-NIPS-metaleanring.pdf>`_
- `Almost No Inner Loop (ANIL) <https://arxiv.org/pdf/1909.09157.pdf>`_

Metric-based
------------

- `Matching Networks <https://arxiv.org/abs/1606.04080>`_
- `Prototypical Networks <https://arxiv.org/abs/1703.05175>`_


Datasets
========

The supported datasets currently include:

- `Omniglot <https://science.sciencemag.org/content/350/6266/1332>`_
- `Mini-ImageNet <https://arxiv.org/abs/1606.04080>`_


************
Installation
************

.. code-block:: bash

    git clone https://github.com/Renovamen/metallic.git
    cd metallic
    python setup.py install

or

.. code-block:: bash

    pip install git+https://github.com/Renovamen/metallic.git --upgrade


******
Github
******

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
   api/functional
   api/utils


****************
Acknowledgements
****************

Metallic is highly inspired by the following awesome libraries:

- `learn2learn <https://github.com/learnables/learn2learn>`_
- `Torchmeta <https://github.com/tristandeleu/pytorch-meta>`_
- `higher <https://github.com/facebookresearch/higher>`_


*******
License
*******

Metallic is MIT licensed, see the `LICENSE <https://github.com/Renovamen/metallic/blob/master/LICENSE>`_ file for more details.
