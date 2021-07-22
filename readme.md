Computational experiments on over-the-air statistical estimation
================================================================

This repository contains work relating to computational experiments for over-the-air statistical estimation.

Related documents and other files are in the "iolite" repository at https://github.com/czlee/iolite.

Installation
------------

Requirements are in `requirements.txt`, so just `pip install -r requirements.txt`.

You need to copy `config.py.example` to `config.py` and set the variables inside it to point to useful directories. One of these directories should contain the datasets.

Currently, the only supported dataset is the ["epsilon" dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon). The simplest setup is to download the `epsilon_normalized.bz2` and `epsilon_normalized.t.bz2` files from this site, to put them in a directory named `epsilon`, and then to set `DATA_DIRECTORY` to the _parent_ of `epsilon`. For example, you could put the two files in `/home/you/jadeite/data/sources/epsilon`, then set `DATA_DIRECTORY = '/home/you/jadeite/data/sources`.

Usage
-----

The main scripts are:

- `simple.py`, which runs simple vanilla gradient descent, no federation involved.
- `fedavg.py`, which runs federated averaging without communication constraints.
- `overtheair.py`, which runs a simple version of our over-the-air analog scheme.
- `dynpower.py`, which runs a version of our analog scheme with dynamic power scaling.
- `stocquant.py`, which runs a stochastic quantization-based digital scheme.

All of these scripts have `--help` available, so for example:

```bash
python overtheair.py --help
```

Coding principles
-----------------
I don't claim these to be best practice or anything, they're just what I was trying to do when writing this code.

- **Use object-oriented structures to minimize duplication.**

  This isn't just a theoretical thing. It makes it easier to "mix and match" experiment structures.

- **Write everything to files straight away in a text-based form.**

  Don't wait until the end of some period—you never know when a simulation will stop unexpectedly. Better to have the data already saved. If a lot of the information may end up redundant, it should be cleaned up with later scripts.

- **Store data in a text-based form.**

  This is inefficient, but it's a lot easier to work with data this way. For example, it can be inspected directly for quick debugging.

  Plots shouldn't be generated on the fly—they should only be generated from text-based data after the fact. This allows plots to be fine-tuned for presentation without having to rerun the experiments.

- **Parameters and arguments should be specified in the same class.**

  This helps with separation of concerns. The flipside is that we then need an infrastructure to make sure all the arguments get added to the `argparse.ArgumentParser` object, which is why each class has an `add_arguments()` class method, which relies heavily on Python's method resolution order to work correctly.
