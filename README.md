Computational experiments on over-the-air statistical estimation
================================================================

_Chuan-Zheng Lee <czlee@stanford.edu>_

This repository contains work relating to computational experiments for over-the-air statistical estimation.

Work on analysing results of running this code is in the [czlee/kyanite](https://github.com/czlee/kyanite) repository.

Papers related to this work:

- C.Z. Lee, L.P. Barnes, A. Özgür, "[Over-the-Air Statistical Estimation](https://ieeexplore.ieee.org/document/9322345)", IEEE GLOBECOM 2020
- C.Z. Lee, L.P. Barnes, A. Özgür, "[Lower Bounds for Over-the-Air Statistical Estimation](https://2021.ieee-isit.org/Papers/ViewPaper.asp?PaperNum=1944)", IEEE ISIT 2021

This repository is mostly licensed under the MIT License, except for the file data/resnet.py, which is copied from [Yerlan Idelbayev's repository](https://github.com/akamaster/pytorch_resnet_cifar10/) and licensed under the BSD 2-Clause License.

Installation
------------

The requirements are in `requirements.txt`, so just
```
pip install -r requirements.txt
```
should set up everything necessary to run the scripts. The main requirements are Python 3.6 or later, and PyTorch.

You need to copy `config.example.yaml` to `config.yaml` and set the variables inside it to point to useful directories. The `DATA_DIRECTORY` directory should contain the datasets.

The CIFAR-10 and Fashion-MNIST datasets are supported, and are retrieved via `torchvision`, which will download them on first use if the required data isn't already in the location specified in `DATA_DIRECTORY`.

Some of the experiments we ran use the ["epsilon" dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon). The simplest setup is to download the `epsilon_normalized.bz2` and `epsilon_normalized.t.bz2` files from this site, to put them in a directory named `epsilon`, and then to set `DATA_DIRECTORY` to the _parent_ of `epsilon`. For example, you could put the two files in `/home/you/jadeite/data/sources/epsilon`, then set `DATA_DIRECTORY` to `/home/you/jadeite/data/sources`. Fair warning: These datasets are about 15 GB when uncompressed.

Usage
-----

### Experiments

The entry script for running experiments is `run.py`. It takes a subcommand, a short name for the experiment. Currently, the experiments are:

- `simple`, which runs simple vanilla gradient descent, no federation involved.
- `fedavg`, which runs federated averaging without communication constraints.
- `overtheair`, which runs a simple version of our over-the-air analog scheme.
- `dynpower`, which runs a version of our analog scheme with dynamic power scaling.
- `stocquant`, which runs a stochastic quantization-based digital scheme.
- `dynquant`, which runs a digital scheme with dynamically adjusted (stochastic) quantization.
- `dynrange`, which runs the same dynamic quantization as `dynquant`, but with unconstrained communication.

To start an experiment, use, for example:
```bash
python run.py dynpower --dataset=cifar10-simple
```

All of these subcommands have `--help` available, so for example:

```bash
python run.py dynpower --help
```

### Convenience scripts

- `list_results.py`, or `./lsr` for short, prints a summary of what's in the results directory.

  It lists directory names, commit hashes, parameters used, whether it's still in progress, and how many experiments in the test matrix have finished.

- `list_unfinished.py` lists experiments that appear to have terminated without finishing.

  One way to clean these up quickly is: `python list_unfinished.py | xargs rm -r`. Be careful with this—it deletes (potentially lots of) files, so inspect the output of `python list_unfinished.py` first.

- `show_command.py` prints the command used to invoke an experiment (specified by a directory).

  This is sometimes useful for rerunning experiments.

- `show_duration.py` prints the duration of each experiment in the given directory.

### Extending or stopping repeated experiments early

Most scripts take the `--repeat` or `-q` argument, which specifies how many times the experiment (or experiment matrix, if multiple values are specified for `--clients` or `--noise`) should be repeated. It is possible to change this while the script is running by overwriting the `repeats` file in the experiment directory with a new number.

For example, if you had initially specified `-q 10` and it's currently up to iteration 5 (starting from 0), then writing `6` to `repeats` will stop it after it finishes the current iteration of the experiment matrix. Writing `20` to `repeats` will extend the experiment to 20 repetitions. An easy way to effect this is to use `echo 20 > latest/repeats`.

You can also stop the script as soon as the current experiment is finished, without it having to get to the end of the current experiment matrix. To do this, create a `stop-now` file, for example, using `touch latest/stop-now`.

How this works: The scripts check after each experiment whether the `stop-now` file exists, and checks the value in the `repeats` file after each full sweep of the experiment matrix.


Development
-----------

### Adding a new experiment

To add a new experiment:

1. Write a new subclass of `BaseExperiment` in the `experiments` directory. (There are a number of base subclasses that new subclasses can also inherit from, for example, `BaseFederatedExperiment`.)

  - Be sure to set the `description` class attribute.
  - The `add_arguments()` class method and `default_params` class attribute should be overridden to add whatever parameters this class needs. You don't need to add arguments defined by parent classes—just call `super().add_arguments()` at the end of your implementation. (At the end, not the beginning, because the base implementation also sets all the default arguments based on `default_params`.)

2. Add this subclass to `experiments_by_name` in `experiments/__init__.py`.

That should be sufficient for it to show up in `python run.py -h`, then you can run the experiment in the same way as the others.

### Coding principles

I don't claim these to be best practice or anything, they're just what I was trying to do when writing this code.

- **Use object-oriented structures to minimize duplication.**

  This isn't just a theoretical thing. It makes it easier to "mix and match" experiment structures. Most experiment structures share common code in various ways; an object-oriented approach allows implementations to be written (and fixes and improvements to be made) in exactly one place.

- **Parameters and arguments should be specified in the same class.**

  This helps with separation of concerns. The flipside is that we then need an infrastructure to make sure all the arguments get added to the `argparse.ArgumentParser` object, which is why each class has an `add_arguments()` class method, which relies heavily on Python's method resolution order to work correctly.

- **Write everything to files straight away in a text-based form.**

  Don't wait until the end of some period—you never know when a simulation will stop unexpectedly. Better to have the data already saved. If a lot of the information may end up redundant, it should be cleaned up with later scripts.

- **Store data in a text-based form.**

  This is inefficient, but it's a lot easier to work with data this way. For example, it can be inspected directly for quick debugging.

  Plots shouldn't be generated on the fly—they should only be generated from text-based data after the fact. This allows plots to be fine-tuned for presentation without having to rerun the experiments.

- **Log enough information to rerun the same experiment.**

  This includes the Git commit hash, whether any files were changed, timestamps and the command used to invoke the experiment. This might seem like it contains redundant information, but sometimes argument definitions change, so it can be hard to be sure _exactly_ how an experiment was started with older code.
