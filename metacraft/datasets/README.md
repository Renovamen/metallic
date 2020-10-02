# Datasets for Meta-Learning

Some datasets and operations commonly used in meta-learning tasks, adapted from [Torchmeta](https://github.com/tristandeleu/pytorch-meta). 

In fact, it can be achieved by `import torchmeta` easily. I write the code just for understanding meta-dataset loading and minibatching better.

&nbsp;
## Omniglot

The Omniglot dataset contains 1623 handwritten characters (classes) collected from 50 alphabets. There are 20 examples associated with each character, where each example is drawn by a different human subject. It is first introduced by:

**Human-level Concept Learning Through Probabilistic Program Induction.** *Brenden M. Lake, et al.* Science 2015. [[Paper]](http://www.sciencemag.org/content/350/6266/1332.short) [[Dataset]](https://github.com/brendenlake/omniglot)

The Omniglot dataset should be downloaded [here (images_background)](https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip) and [here (images_evaluation)](https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip) and unziped first.

The meta train / val / test splits are taken from [jakesnell/prototypical-networks](https://github.com/jakesnell/prototypical-networks). These splits are over 1028 / 172 / 423 classes (characters).