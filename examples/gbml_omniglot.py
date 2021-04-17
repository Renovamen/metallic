import os
import sys

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

from torch import optim

from metallic.data.benchmarks import get_benchmarks
from metallic.data.dataloader import MetaDataLoader
from metallic.models import OmniglotCNN
from metallic.metalearners import FOMAML, MAML, Reptile, MinibatchProx, ANIL
from metallic.trainer import Trainer
from metallic.utils import Logger

# ---- hyperparameters ----
ALGO = 'minibatchprox'
BATCH_SIZE = 16
N_WAY = 5
K_SHOT = 1
OUTER_LR = 0.001
INNER_LR = 0.4
INNER_STEPS = 1
N_EPOCHES = 100
N_ITERS_PER_EPOCH = 500
N_ITERS_TEST = 600
N_WORKERS = 5
# -------------------------

ALGO_LIST = {
    'maml': MAML,
    'fomaml': FOMAML,
    'reptile': Reptile,
    'minibatchprox': MinibatchProx,
    'anil': ANIL
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

    model = OmniglotCNN(N_WAY)

    if ALGO == 'anil':
        in_optim = optim.SGD(model.classifier.parameters(), lr=INNER_LR)
    else:
        in_optim = optim.SGD(model.parameters(), lr=INNER_LR)
    out_optim = optim.Adam(model.parameters(), lr=OUTER_LR)

    metalearner = ALGO_LIST[ALGO](
        model = model,
        in_optim = in_optim,
        out_optim = out_optim,
        root = os.path.join(base_path, 'checkpoints'),
        inner_steps = INNER_STEPS
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
