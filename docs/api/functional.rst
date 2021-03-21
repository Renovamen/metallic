metallic.functional
=======================

.. automodule:: metallic.functional


Loss Functions
-----------------------

.. autoclass:: ProximalRegLoss
    :members:
    :undoc-members:
    :show-inheritance:


Distance Functions
-----------------------

Some distance computing functions for calculating similarity between two tensors. They are useful in metric-based meta-learning algorithms.

.. autofunction:: euclidean_distance

.. autofunction:: cosine_distance


Gradients
-----------------------

Some operations on gradients, which are are useful in gradient-based meta-learning algorithms.

.. autofunction:: apply_grads


Prototypes
-----------------------

.. autofunction:: get_prototypes
