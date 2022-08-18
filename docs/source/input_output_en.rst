Input/Output
============
Dataloader
-------------------------------
:code:`Dataloader` is the interface of loading dataset, it is used to encapsulate \
the custom Dataset into an array according to the size of :code:`batch_size` and \
whether it is shuffle, etc, for network training.

:code:`Dataloader` is consists of dataset and sampler, and the initialization parameters \
are as follows:

- **dataset(Dataset)** -- the dataset to be loaded
- **batch_size(int, optional)** -- the number of samples in each batch, the default is 1
- **shuffle(bool, optional)** -- whether reorder the data at the beginning of each epoch, the default is False
- **sampler(Sampler, optional)** -- customize the strategy for taking samples from the dataset
- **batch_sampler(Sampler, optional)** -- Similar to sampler, but only returns the index of one batch
- **collate_fn(callable, optional)** -- A function that composes a list of samples into a mini-batch
- **drop_last(bool, optional)** -- If set to True, for the last batch, if the number of samples is less than batch_size, it will be thrown away. For example, if batch_size is set to 64, and the dataset has only 100 samples, then the last 36 samples will be trained during training. will be thrown away. If False (default), normal execution will continue, but the final batch_size will be smaller.

Loading MNIST dataset as example：

.. code-block:: python

    root = './Datasets/MNIST' # root of data
    train_set = dataset(root, is_train=True)   # Train set
    test_set = dataset(root, is_train=False)   # Test set
    bat_size = 20
    # Create DataLoader
    train_loader = spaic.Dataloader(train_set, batch_size=bat_size, shuffle=True)
    test_loader = spaic.Dataloader(test_set, batch_size=bat_size, shuffle=False)


.. note::

   To be mentioned: \
    1. If :code:`sampler` has been specified when creating :code:`Dataloader`, the :code:`shuffle` must be False.

    2. If :code:`batch_sampler` has been specified, then, :code:`batch_size`, :code:`shuffle`, :code:`sampler` and :code:`drop_last` can no longer be specified.