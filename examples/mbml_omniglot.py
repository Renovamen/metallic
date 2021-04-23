import os
import sys

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

from torch import optim

from metallic.data.benchmarks import get_benchmarks
from metallic.data.dataloader import MetaDataLoader
from metallic.models import OmniglotCNN
from metallic.metalearners import ProtoNet, MatchNet
from metallic.trainer import Trainer
from metallic.utils import Logger

# ---- hyperparameters ----
ALGO = 'protonet'
BATCH_SIZE = 16
N_WAY = 5
K_SHOT = 1
LR = 0.001
N_EPOCHES = 100
N_ITERS_PER_EPOCH = 100
N_ITERS_TEST = 100
N_WORKERS = 5
# -------------------------

ALGO_LIST = {
    'protonet': ProtoNet,
    'matchnet': MatchNet
}

def set_trainer():
    train_dataset, val_dataset, _ = get_benchmarks(
        name = 'omniglot',
        root = os.path.join(base_path, 'data'),
        n_way = N_WAY,
        k_shot = K_SHOT,
    )

    train_loader = MetaDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = MetaDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = OmniglotCNN()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    metalearner = ALGO_LIST[ALGO](
        model = model,
        optim = optimizer,
        root = os.path.join(base_path, 'checkpoints')
    )

    logger = Logger(
        root = os.path.join(base_path, 'logs'),
        n_iters_per_epoch = N_ITERS_PER_EPOCH,
        log_basename = metalearner.alg_name,
        log_interval = 10
    )

    trainer = Trainer(
        metalearner = metalearner,
        train_loader = train_loader,
        val_loader = val_loader,
        n_epoches = N_EPOCHES,
        n_iters_per_epoch = N_ITERS_PER_EPOCH,
        n_iters_test = N_ITERS_TEST,
        logger = logger
    )
    return trainer


if __name__ == '__main__':
    trainer = set_trainer()
    trainer.run_train()
