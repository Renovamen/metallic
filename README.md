# Metallic

Metallic is a meta-learning library based on [PyTorch](https://github.com/pytorch/pytorch).

Different from other meta-learning libraries, Metallic tries to:

- Provide a clean, lightweight and modularized toolbox for setting up meta-learning experiments quickly with the least amount of code.
- For gradient-based meta-learning methods (like MAML), support more optimizers instead of SGD only using [higher](https://github.com/facebookresearch/higher).

The library is **work in progress**.


&nbsp;

## Features

### Algorithms

The supported interface algorithms currently include:

#### Gradient-based

- [Model-Agnostic Meta-Learning (MAML)](https://arxiv.org/abs/1703.03400), including first-order version
- [Reptile](https://arxiv.org/abs/1803.02999)
- [Minibatch Proximal Update](https://panzhous.github.io/assets/pdf/2019-NIPS-metaleanring.pdf)

#### Metric-based

- [Matching Networks](https://arxiv.org/abs/1606.04080)
- [Prototypical Networks](https://arxiv.org/abs/1703.05175)


### Datasets

The supported datasets currently include:

- [Omniglot](https://science.sciencemag.org/content/350/6266/1332)
- [Mini-ImageNet](https://arxiv.org/abs/1606.04080)


&nbsp;

## Installation

```bash
git clone https://github.com/Renovamen/metallic.git
cd metallic
python setup.py install
```

or

```bash
pip install git+https://github.com/Renovamen/metallic.git --upgrade
```


&nbsp;

## Documentations

Check the API documentation here: [metallic-docs.vercel.app](https://metallic-docs.vercel.app)


&nbsp;

## Acknowledgements

Metallic is highly inspired by the following awesome libraries:

- [learn2learn](https://github.com/learnables/learn2learn)
- [Torchmeta](https://github.com/tristandeleu/pytorch-meta)
- [higher](https://github.com/facebookresearch/higher)

&nbsp;

## License

Metallic is MIT licensed, see the [LICENSE](LICENSE) file for more details.
