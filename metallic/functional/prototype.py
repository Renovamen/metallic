import torch

def get_prototypes(
    inputs: torch.FloatTensor, targets: torch.LongTensor
) -> torch.FloatTensor:
    """
    Compute the **prototypes** for each class in the task. Each prototype is the
    mean vector of the embedded support points belonging to its class.

    Args:
        inputs (torch.FloatTensor): Embeddings of the support points, with
            shape ``(n_samples, embed_dim)``
        targets (torch.LongTensor): Targets of the support points, with shape
            ``(n_samples)``

    Returns:
        prototypes (torch.FloatTensor): Prototypes for each class, with shape \
            ``(n_way, embed_dim)``.
    """

    n_way = torch.unique(targets).size(0)  # number of classes per task
    k_shot = targets.size(0) // n_way  # number of samples per class
    embed_dim = inputs.size(-1)  # embedding size

    indices = targets.unsqueeze(-1).expand_as(inputs)  # (n_samples, embed_dim)
    prototypes = inputs.new_zeros(n_way, embed_dim)  # (n_way, embed_dim)
    prototypes.scatter_add_(0, indices, inputs).div_(k_shot)

    return prototypes
