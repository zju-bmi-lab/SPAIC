# -*- coding: utf-8 -*-
"""
Created on 2020/8/12
@project: SPAIC
@filename: Dataloader
@author: Hong Chaofei
@contact: hongchf@gmail.com
@description:
定义数据导入模块
"""
from .sampler import *
import numpy as np


# Dataloader class is written by referring to https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py.
class _BaseDatasetFetcher(object):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index):
        raise NotImplementedError()


class _MapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_MapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)


def default_collate(batch):
    # shape of data is [batch_size, *shape]
    data = [item[0] for item in batch]
    data = np.array(data)
    target = [item[1] for item in batch]
    target = np.array(target)
    return [data, target]


class Dataloader(object):
    """
    sampler的作用是生成一系列的index
    而batch_sampler则是将sampler生成的indices打包分组，得到一个又一个batch的index
    """
    __initialized = False

    def __init__(self, dataset, batch_size=1, shuffle=False,
                 sampler=None, batch_sampler=None, collate_fn=None, drop_last=False):
        self.dataset = dataset
        self.data = None
        self.label = None
        # self.source = None
        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')
        if batch_sampler is not None:
            # auto_collation with custom batch_sampler
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            batch_size = None
            drop_last = False
        elif batch_size is None:
            # no auto_collation
            if drop_last:
                raise ValueError('batch_size=None option disables auto-batching '
                                 'and is mutually exclusive with drop_last')

        if sampler is None:
            if shuffle:
                # Cannot statically verify that dataset is Sized
                # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
                sampler = RandomSampler(dataset)  # type: ignore
            else:
                sampler = SequentialSampler(dataset)

        if batch_size is not None and batch_sampler is None:
            # auto_collation without custom batch_sampler
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        if collate_fn is None:
            collate_fn = default_collate

        self.collate_fn = collate_fn
        self.__initialized = True

        # self.try_fetch()

    def __setattr__(self, attr, val):
        if self.__initialized and attr in ('batch_size', 'sampler', 'drop_last'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

        super(Dataloader, self).__setattr__(attr, val)

    def __iter__(self):
        return _SingleProcessDataLoaderIter(self)

    # 在循环前获得一个batch的数据给Node结点用于build
    def try_fetch(self):
        for i, item in enumerate(_SingleProcessDataLoaderIter(self)):
            self.data = item[0]
            self.label = item[1]
            if 'maxNum' in self.dataset.data.keys():
                self.num = self.dataset.data['maxNum']
            else:
                shape = self.data.shape[1:]
                self.num = int(np.prod(shape))
            break
        return self.data, self.label

    @property
    def _auto_collation(self):
        return self.batch_sampler is not None

    @property
    def _index_sampler(self):
        # The actual sampler used for generating indices for `_DatasetFetcher`
        # to read data at each time.
        if self._auto_collation:
            return self.batch_sampler
        else:
            return self.sampler

    def __len__(self):
        return len(self._index_sampler)


class _BaseDataLoaderIter(object):
    def __init__(self, loader: Dataloader):
        self._dataset = loader.dataset
        self._auto_collation = loader._auto_collation
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._sampler_iter = iter(self._index_sampler)
        self._collate_fn = loader.collate_fn
        self._num_yielded = 0

    def __iter__(self):
        return self

    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration

    def __next__(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.index_sampler)

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("{} cannot be pickled", self.__class__.__name__)


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(_SingleProcessDataLoaderIter, self).__init__(loader)

        self._dataset_fetcher = _MapDatasetFetcher(self._dataset, self._auto_collation, self._collate_fn,
                                                   self._drop_last)

    def __next__(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        return data

    next = __next__
