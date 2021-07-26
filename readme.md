Computational experiments on over-the-air statistical estimation
================================================================

_Chuan-Zheng Lee <czlee@stanford.edu>_

This repository contains work relating to computational experiments for over-the-air statistical estimation.

Installation
------------

Requirements are in `requirements.txt`, so just `pip install -r requirements.txt`. The main requirement is PyTorch.

You need to copy `config.example.yaml` to `config.yaml` and set the variables inside it to point to useful directories. The `DATA_DIRECTORY` directory should contain the datasets.

A lot of the experiments we ran use the ["epsilon" dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon). The simplest setup is to download the `epsilon_normalized.bz2` and `epsilon_normalized.t.bz2` files from this site, to put them in a directory named `epsilon`, and then to set `DATA_DIRECTORY` to the _parent_ of `epsilon`. For example, you could put the two files in `/home/you/jadeite/data/sources/epsilon`, then set `DATA_DIRECTORY = '/home/you/jadeite/data/sources`.

The CIFAR-10 and FashionMNIST datasets are also supported, and are retrieved via `torchvision`, which will download them on first use if the required data isn't already in the location specified in `DATA_DIRECTORY`.

Usage
-----

### Experiments

The main scripts for running experiments are:

- `simple.py`, which runs simple vanilla gradient descent, no federation involved.
- `fedavg.py`, which runs federated averaging without communication constraints.
- `overtheair.py`, which runs a simple version of our over-the-air analog scheme.
- `dynpower.py`, which runs a version of our analog scheme with dynamic power scaling.
- `stocquant.py`, which runs a stochastic quantization-based digital scheme.
- `dynquant.py`, which runs a digital scheme with dynamically adjusted (stochastic) quantization.

All of these scripts have `--help` available, so for example:

```bash
python overtheair.py --help
```

### Convenience scripts

- `list_results.py`, or `./lsr` for short, prints a summary of what's in the results directory.

  It lists directory names, commit hashes, parameters used, whether it's still in progress, and how many experiments in the test matrix have finished.
  
- `list_unfinished.py` lists experiments that appear to have terminated without finishing.

  One way to clean these up quickly is: `python list_unfinished.py | xargs rm -r`. Be careful with this—it deletes (potentially lots of) files, so inspect the output of `python list_unfinished.py` first.

- `show_command.py` prints the command used to invoke an experiment (specified by a directory).

  This is sometimes useful for rerunning experiments.

Coding principles
-----------------
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
