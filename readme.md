Computational experiments on over-the-air statistical estimation
================================================================

This repository contains work relating to computational experiments for over-the-air statistical estimation.

Related documents and other files are in the "iolite" repository at https://github.com/czlee/iolite.


Design principles
-----------------
- **Use object-oriented structures to minimize duplication.**

  This isn't just a theoretical thing. It makes it easier to "mix and match" experiment structures.

- **Write everything to files straight away in a text-based form.**

  Don't wait until the end of some period—you never know when a simulation will stop unexpectedly. Better to have the data already saved. If a lot of the information may end up redundant, it should be cleaned up with later scripts.

- **Store data in a text-based form.**

  This is inefficient, but it's a lot easier to work with data this way. For example, it can be inspected directly for quick debugging.

  Plots shouldn't be generated on the fly—they should only be generated from text-based data after the fact. This allows plots to be fine-tuned for presentation without having to rerun the experiments.
