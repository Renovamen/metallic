# Metallic

Metallic is a library for meta-learning research based on [PyTorch](https://github.com/pytorch/pytorch).

Different from other meta-learning libraries, Metallic tries to:

- Provide a clean, lightweight and modularized toolbox for setting up meta-learning experiments quickly with the least amount of code.
- For gradient-based meta-learning methods (like MAML), support more optimizers instead of SGD only using [higher](https://github.com/facebookresearch/higher).

The library is **work in progress**.


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
