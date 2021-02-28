import os
import sys
import unittest
import numpy as np
from itertools import combinations
import torch
from torchvision.transforms import Compose, Resize, ToTensor

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

from metallic.data.datasets import Omniglot, MetaDataset, TaskDataset
from metallic.data.dataloader import MetaDataLoader

N_WAY = 5
K_SHOT = {
    "support": 10,
    "query": 5
}
DATA_PATH = os.path.join(base_path, 'data')
BATCH_SIZE = 20


def sample_task(dataset: MetaDataset, n_way: int) -> TaskDataset:
    """
    Sample a task containing `n_way` classes.
    """
    class_indices = np.random.choice(
        dataset.n_classes,
        size = n_way,
        replace = False
    ) # [class_id1, class_id2, ...]
    return dataset[tuple(class_indices)]


class TestData(unittest.TestCase):
    def test_meta_dataset(self) -> None:
        samples_per_class = 20

        dataset = Omniglot(root = DATA_PATH, n_way = N_WAY)
        assert isinstance(dataset, MetaDataset)

        # sample a task
        task = sample_task(dataset, N_WAY)
        assert task.n_classes == N_WAY
        assert len(task) == N_WAY * samples_per_class

    def test_meta_dataset_split(self) -> None:
        dataset = Omniglot(
            root = DATA_PATH,
            n_way = N_WAY,
            k_shot_support = K_SHOT["support"],
            k_shot_query = K_SHOT["query"]
        )

        # sample a task
        task = sample_task(dataset, N_WAY)
        assert set(task.keys()) == set(["support", "query"])

        for split in ["support", "query"]:
            assert task[split].n_classes == N_WAY
            assert len(task[split]) == N_WAY * K_SHOT[split]

    def test_metaloader(self):
        transform = Compose([Resize(28), ToTensor()])
        dataset = Omniglot(
            root = DATA_PATH,
            n_way = N_WAY,
            transform = transform,
            k_shot_support = K_SHOT["support"],
            k_shot_query = K_SHOT["query"]
        )
        dataloader = MetaDataLoader(dataset, batch_size=BATCH_SIZE)

        batch = next(iter(dataloader))
        assert isinstance(batch, dict)

        for split in ["support", "query"]:
            assert split in batch
            inputs, targets = batch[split]

            assert isinstance(inputs, torch.Tensor)
            assert isinstance(targets, torch.Tensor)
            assert inputs.ndim == N_WAY
            assert inputs.shape[:2] == (BATCH_SIZE, N_WAY * K_SHOT[split])
            assert targets.ndim == 2
            assert targets.shape[:2] == (BATCH_SIZE, N_WAY * K_SHOT[split])


if __name__ == '__main__':
    unittest.main()
