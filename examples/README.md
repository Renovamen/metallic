# Examples

This folder contains some examples to showcase the features of Metacraft and try to reproduce some popular meta-learning experiments.

Now it includes:

- MAML (`maml`) and FOMAML (`fomaml`)

    **Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.** *Chelsea Finn, et al.* ICML 2017. [Paper](https://arxiv.org/pdf/1703.03400.pdf) [Code](https://github.com/cbfinn/maml)


&nbsp;

## Usage

First, edit the configuration files under [`configs`](configs) folder manually.

&nbsp;

### Train

```bash
python examples/train.py --config configs/example.yaml
```

where `configs/example.yaml` is the path to your configuration file.

&nbsp;

### Test

```bash
python examples/test.py --config configs/example.yaml
```