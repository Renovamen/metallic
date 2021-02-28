from collections import OrderedDict
from typing import Any, Union
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset

class MetaCollate:
    def __init__(self) -> None:
        self.collate_fn = default_collate

    def collate_task(self, task: Union[Dataset, OrderedDict]) -> Any:
        if isinstance(task, Dataset):
            return self.collate_fn([sample for sample in task])
        # deal with task that has been splited in support / query set
        elif isinstance(task, OrderedDict):
            return OrderedDict([
                (key, self.collate_task(subtask))
                for (key, subtask) in task.items()
            ])
        else:
            raise NotImplementedError()

    def __call__(self, batch) -> Any:
        return self.collate_fn([self.collate_task(task) for task in batch])
