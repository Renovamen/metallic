import torch
from torchvision.transforms import Compose, Resize, ToTensor

from metacraft.datasets.benchmarks import Omniglot
from metacraft.datasets.benchmarks.base import MetaDataset, SubsetTask
from metacraft.datasets.transforms import Rotation
from metacraft.datasets.data import MetaSplitter, BatchMetaDataLoader

dataset_path = '/Users/zou/Desktop/metacraft/data/omniglot'
n_way = 5
k_shot_support = 10
k_shot_query = 5
batch_size = 4

transform = Compose([Resize(28), ToTensor()])
class_augmentations = [Rotation([90, 180, 270])]
dataset_transform = MetaSplitter(
    shuffle = True,
    k_shot_support = k_shot_support, 
    k_shot_query = k_shot_query
)

dataset = Omniglot(
    root = dataset_path,
    n_way = n_way,
    meta_split = 'train',
    transform = transform,
    dataset_transform = dataset_transform,
    class_augmentations = class_augmentations
)

# dataset is a class `MetaDataset`
assert isinstance(dataset, MetaDataset)

# sample a task
task = dataset.sample_task()

# task is a dictionary with keys [train, test]
assert isinstance(task, dict)
assert set(task.keys()) == set(['support', 'query'])

# support set
assert isinstance(task['support'], SubsetTask)
assert task['support'].n_classes == n_way
assert len(task['support']) == n_way * k_shot_support

# query set
assert isinstance(task['query'], SubsetTask)
assert task['query'].n_classes == n_way
assert len(task['query']) == n_way * k_shot_query

# batch dataloader
dataloader = BatchMetaDataLoader(dataset, batch_size = batch_size, shuffle = True)
batch = next(iter(dataloader))
assert isinstance(batch, dict)
assert 'support' in batch
assert 'query' in batch

support_inputs, support_targets = batch['support']
query_inputs, query_targets = batch['query']

# support sets in batches
assert isinstance(support_inputs, torch.Tensor)
assert isinstance(support_targets, torch.Tensor)
assert support_inputs.ndim == n_way
assert support_inputs.shape[:2] == (batch_size, n_way * k_shot_support)
assert support_targets.ndim == 2
assert support_targets.shape[:2] == (batch_size, n_way * k_shot_support)

# query sets in batches
assert isinstance(query_inputs, torch.Tensor)
assert isinstance(query_targets, torch.Tensor)
assert query_inputs.ndim == n_way
assert query_inputs.shape[:2] == (batch_size, n_way * k_shot_query)
assert query_targets.ndim == 2
assert query_targets.shape[:2] == (batch_size, n_way * k_shot_query)