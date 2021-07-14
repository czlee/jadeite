from . import epsilon


def get_datasets(name):
    # Not currently used.
    # (I wrote it then realized it didn't make sense to separate this from
    # models and loss functions.)

    if name == "epsilon":
        train_dataset = epsilon.EpsilonDataset(train=True, small=False)
        test_dataset = epsilon.EpsilonDataset(train=False, small=False)

    elif name == "epsilon-small":
        train_dataset = epsilon.EpsilonDataset(train=True, small=True)
        test_dataset = epsilon.EpsilonDataset(train=False, small=True)

    else:
        raise ValueError(f"No dataset with name: {name}")

    return train_dataset, test_dataset
