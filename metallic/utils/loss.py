from typing import Callable, Optional
from torch import Tensor, nn

class ProximalRegLoss(nn.Module):
    """
    Add an explicitly l2 regularization term based on meta-parameters and
    model-parameters to the loss function. This is because we want model-parameters
    to retain a close dependence on meta-parameters.

    This loss function has been used in [1] and [2].

    Args:
        loss_function (callable, optional): Loss function
        lamb (float, optional, float=0.1): Regularization strength of the inner
            level proximal regularization

    .. admonition:: References

        1. "`Efficient Meta Learning via Minibatch Proximal Update. \
            <https://panzhous.github.io/assets/pdf/2019-NIPS-metaleanring.pdf>`_" \
            Pan Zhou, et al. NIPS 2019. The supplementary file can be found \
            `here <https://panzhous.github.io/assets/pdf/2019-NIPS-metaleanring-supplementary.pdf>`_.
        2. "`Meta-Learning with Implicit Gradients. <https://arxiv.org/abs/1909.04630>`_" \
            Aravind Rajeswaran, et al. NIPS 2019.
    """

    def __init__(
        self, loss_function: Optional[Callable] = None, lamb: float = 0.1
    ) -> None:
        super(ProximalRegLoss, self).__init__()
        self.loss_function = loss_function
        self.lamb = lamb

    def __call__(
        self, input: Tensor, target: Tensor, init_params: Tensor, params: Tensor
    ) -> Tensor:
        sq_diff = sum([
            ((init_param - param) ** 2).sum()
            for init_param, param in zip(init_params, params)
        ])
        reg_term = 0.5 * self.lamb * sq_diff
        return self.loss_function(input, target) + reg_term
