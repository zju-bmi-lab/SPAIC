# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import numpy as np


import json
import pickle

from spaic.IO.utils import load_kp_data, save_kp_feature, save_mfcc_feature, load_mfcc_data, load_aedat_v3, un_tar
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

class Dataset(object):
    r"""
    All datasets that represent a map from keys to data samples should subclass it.
    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a data
    sample for a given key.
    Subclasses should also overwrite :meth:`__len__`, which is expected to return
    the size of the sample dataset.
    """

    def __init__(self, **kwargs):
        super().__init__()

    # 根据索引返回数据内容和标签
    def __getitem__(self, index):
        raise NotImplementedError

    # 返回数据集大小
    def __len__(self):
        raise NotImplementedError

class CustomDataset(Dataset):
    r"""
    自定义数据集：
    个人采集的实值数据
    """

    def __init__(self, data=None, label=None):
        super().__init__()
        data_type = type(data)
        label_type = type(label)
        assert data_type is list or data_type is np.ndarray, "The type of data should be list or np.ndarray"
        assert label_type is list or label_type is np.ndarray, "The type of label should be list or np.ndarray"

        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = np.float32(self.data[index])
        label = np.int64(self.label[index])
        return data, label

    def __len__(self):
        return len(self.data)

class CustomSpikeDataset(Dataset):
    r"""
    自定义数据集：
    编码后的脉冲数据（仅支持[spike_time, neuron_ids]表示）
    """

    def __init__(self, spike_times=None, neuron_ids=None, label=None):
        super().__init__()
        # The shape of spike_times should be [sample_num, spikes_times_num]
        spike_times_type = type(spike_times)
        neuron_ids_type = type(neuron_ids)
        label_type = type(label)
        assert spike_times_type is list or spike_times_type is np.ndarray, "The type of data should be list or np.ndarray"
        assert neuron_ids_type is list or neuron_ids_type is np.ndarray, "The type of data should be list or np.ndarray"
        assert label_type is list or label_type is np.ndarray, "The type of label should be list or np.ndarray"

        self.spike_times = spike_times
        self.neuron_ids = neuron_ids
        self.label = label

    def __getitem__(self, index):
        spike_time = self.spike_times[index]
        neuron_id = self.neuron_ids[index]
        label = self.label[index]
        spiking = [spike_time, neuron_id]
        return spiking, label

    def __len__(self):
        return len(self.data)


class SpecifiedDataset(Dataset):

    r"""
    labels load from json file
    """
    def __init__(self, image_file, label_file):
        super().__init__()
        # 加载数据集
        label_file = label_file
        img_folder = image_file

        fp = open(label_file, 'r')
        data_dict = json.load(fp)

        # 如果图像数和标签数不匹配说明数据集标注生成有问题，报错提示
        assert len(data_dict['images']) == len(data_dict['labels'])
        num_data = len(data_dict['images'])

        self.filenames = []
        self.labels = []
        self.img_folder = img_folder
        for i in range(num_data):
            self.filenames.append(data_dict['images'][i])
            self.labels.append(data_dict['labels'][i])

    def __getitem__(self, index):
        img_name = np.float32(self.img_folder + self.filenames[index])
        label = np.int64(self.labels[index])
        img = plt.imread(img_name)
        return img, label

    def __len__(self):
        return len(self.filenames)

class cifar10(Dataset):
    files = {
        "train_dataset1": 'data_batch_1',
        "train_dataset2": 'data_batch_2',
        "train_dataset3": 'data_batch_3',
        "train_dataset4": 'data_batch_4',
        "train_dataset5": 'data_batch_5',
        "test_dataset": 'test_batch'
    }

    def __init__(self, root, is_train=True):
        super().__init__()
        self.root = root
        self._is_train = is_train
        self.data = {
            'train_images': [],
            'test_images': [],
            'train_labels': [],
            'test_labels': []
        }
        if not isinstance(self._is_train, bool):
            raise TypeError(">> is_train should be boolean value")
        if self._dataset_exists():
            self._to_numpy_format()
        else:
            raise ValueError(">> Failed to load the set, file not exist. You should download the dataset firstly.")

    def __getitem__(self, index):
        # 数据归一化到[0,1]
        # mean = (0.4914, 0.4822, 0.4465)
        # std = (0.2023, 0.1994, 0.2010)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if self._is_train:
            img = (self.data['train_images'][index]/255.0 - mean)/std
            img = np.float32(img.transpose(2, 0, 1))
            # img = np.float32(self.data['train_images'][index])
            label = np.int64(self.data['train_labels'][index])
        else:
            img = (self.data['test_images'][index] / 255.0 - mean) / std
            img = np.float32(img.transpose(2, 0, 1))
            label = np.int64(self.data['test_labels'][index])

        return img, label

    def __len__(self):
        if self._is_train:
            return len(self.data['train_images'])
        else:
            return len(self.data['test_images'])

    @property
    def dataset_folder(self):
        return os.path.join(self.root, self.__class__.__name__)

    @property
    def data_dict(self):
        return self.data

    @property
    def is_train(self):
        return self._is_train

    @is_train.setter
    def is_train(self, is_train):
        assert is_train in [True, False], ">> Invalid is_train setting"
        self._is_train = is_train

    def load_cifar10_batch(self, folder_path, batch_id):

        with open(folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
            batch = pickle.load(file, encoding='latin1')

        # features and labels
        features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)  # batch['data']
        labels = batch['labels']

        return features, labels

    def _to_numpy_format(self):

        self.data['train_images'], self.data['train_labels'] = self.load_cifar10_batch(self.root, 1)

        for batch_id in range(2, 6):
            features, labels = self.load_cifar10_batch(self.root, batch_id)
            self.data['train_images'] = np.concatenate([self.data['train_images'], features])
            self.data['train_labels'] = np.concatenate([self.data['train_labels'], labels])

        with open(self.root + '/test_batch', mode='rb') as f:
            batch = pickle.load(f, encoding='latin1')
            self.data['test_images'] = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)  # batch['data']
            self.data['test_labels'] = batch['labels']

        print(">> Dataset loaded")

    def _dataset_exists(self):
        if os.path.exists(os.path.join(self.root)):
            for file in cifar10.files.values():
                if not os.path.isfile(os.path.join(self.root, file)):
                    return False
            return True
        else:
            return False


class ImageNet(Dataset):
    files = {
        "train_dataset": 'ILSVRC2012_img_train.tar',
        "val_dataset": 'ILSVRC2012_img_val.tar',
        "val_label": 'ILSVRC2012_devkit_t12.tar.gz'
    }

    def __init__(self, root, is_train=True):
        super().__init__()
        self.root = root
        self._is_train = is_train
        self.data = {
            'train_images': [],
            'val_images': [],
            'train_labels': [],
            'val_labels': []
        }
        if not isinstance(self._is_train, bool):
            raise TypeError(">> is_train should be boolean value")
        if self._dataset_exists():
            self.untar_train_tar(ImageNet.files['train_dataset'].split('.')[0])
            val_dir = ImageNet.files['val_dataset'].split('.')[0]
            devkit_dir = ImageNet.files['val_label'].split('.')[0]
            self.move_val_img(val_dir=val_dir, devkit_dir=devkit_dir)

            pass
            # self._to_numpy_format()
        else:
            raise ValueError(">> Failed to load the set, file not exist. You should download the dataset firstly.")

    def __getitem__(self, index):
        # 数据归一化到[0,1]
        if self._is_train:
            img = np.float32(self.data['train_images'][index])
            label = np.int64(self.data['train_labels'][index])
        else:
            img = np.float32(self.data['test_images'][index])
            label = np.int64(self.data['test_labels'][index])

        return img / 255.0, label

    def __len__(self):
        if self._is_train:
            return len(self.data['train_images'])
        else:
            return len(self.data['test_images'])

    @property
    def dataset_folder(self):
        return os.path.join(self.root, self.__class__.__name__)

    @property
    def data_dict(self):
        return self.data

    @property
    def is_train(self):
        return self._is_train

    @is_train.setter
    def is_train(self, is_train):
        assert is_train in [True, False], ">> Invalid is_train setting"
        self._is_train = is_train

    def untar_train_tar(self, train_tar):
        """
        untar images from train_tar and save in corresponding folders
        organize like:
        /train
           /n01440764
               images
           /n01443537
               images
            .....
        """
        root, _, files = next(os.walk(os.path.join(self.root, train_tar)))
        for file in files:
            un_tar(os.path.join(root, file), os.path.join(self.root, 'train'))

    def move_val_img(self, val_dir, devkit_dir):
        """
        move val_img to correspongding folders.
        val_id(start from 1) -> ILSVRC_ID(start from 1) -> WIND
        organize like:
        /val
           /n01440764
               images
           /n01443537
               images
            .....
        """
        # load synset, val ground truth and val images list
        from scipy import io
        import shutil
        devkit_dir_name = os.path.join(self.root, devkit_dir)
        synset = io.loadmat(os.path.join(devkit_dir_name, 'ILSVRC2012_devkit_t12', 'data', 'meta.mat'))

        ground_truth = open(os.path.join(devkit_dir_name, 'ILSVRC2012_devkit_t12', 'data', 'ILSVRC2012_validation_ground_truth.txt'))
        lines = ground_truth.readlines()
        labels = [int(line[:-1]) for line in lines]

        val_dir_name = os.path.join(self.root, val_dir)
        root, _, filenames = next(os.walk(val_dir_name))
        for filename in filenames:
            # val image name -> ILSVRC ID -> WIND
            val_id = int(filename.split('.')[0].split('_')[-1])
            ILSVRC_ID = labels[val_id - 1]
            WIND = synset['synsets'][ILSVRC_ID - 1][0][1][0]
            print("val_id:%d, ILSVRC_ID:%d, WIND:%s" % (val_id, ILSVRC_ID, WIND))

            # move val images
            output_dir = os.path.join(self.root, 'val', WIND)
            if os.path.isdir(output_dir):
                pass
            else:
                os.mkdir(output_dir)
            shutil.move(os.path.join(root, filename), os.path.join(output_dir, filename))

    def _dataset_exists(self):
        if os.path.exists(os.path.join(self.root)):
            for file in ImageNet.files.values():
                if os.path.isfile(os.path.join(self.root, file)):
                    # file_name = os.path.join(self.root, file)
                    un_tar(os.path.join(self.root, file), self.root)
                else:
                    return False
            return True
        else:
            return False


class MNIST(Dataset):
    r"""
    A 10-class multi-class classfication
    Args:
    root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
    is_train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
    """
    class_number = 10
    maxNum = 28*28
    resources = [
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
    ]

    files = {
        "train_images": 'train-images-idx3-ubyte',
        "test_images": 't10k-images-idx3-ubyte',
        "train_labels": 'train-labels-idx1-ubyte',
        "test_labels": 't10k-labels-idx1-ubyte',
    }

    def __init__(self, root, is_train=True):

        super().__init__()
        self.root = root
        self._is_train = is_train
        self.data = {}
        if not isinstance(self._is_train, bool):
            raise TypeError(">> is_train should be boolean value")
        if self._dataset_exists():
            self._to_numpy_format()
        else:
            self.download()

    def __getitem__(self, index):
        # 数据归一化到[0,1]
        if self._is_train:
            img = np.float32(self.data['train_images'][index])
            label = np.int64(self.data['train_labels'][index])
        else:
            img = np.float32(self.data['test_images'][index])
            label = np.int64(self.data['test_labels'][index])

        return img / 255.0, label

    def __len__(self):
        if self._is_train:
            return len(self.data['train_images'])
        else:
            return len(self.data['test_images'])

    @property
    def dataset_folder(self):
        return os.path.join(self.root, self.__class__.__name__)

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    @property
    def data_dict(self):
        return self.data

    @property
    def is_train(self):
        return self._is_train

    @is_train.setter
    def is_train(self, is_train):
        assert is_train in [True, False], ">> Invalid is_train setting"
        self._is_train = is_train

    def _to_numpy_format(self):
        with open(
                os.path.join(self.root, MNIST.files['train_images']), 'rb'
        ) as f:
            self.data['train_images'] = np.frombuffer(
                f.read(), np.uint8, offset=16
            ).reshape(-1, 28 ** 2)
        with open(
                os.path.join(self.root, MNIST.files['train_labels']), 'rb'
        ) as f:
            self.data['train_labels'] = np.frombuffer(
                f.read(), np.uint8, offset=8
            )
        with open(
                os.path.join(self.root, MNIST.files['test_images']), 'rb'
        ) as f:
            self.data['test_images'] = np.frombuffer(
                f.read(),
                np.uint8,
                offset=16
            ).reshape(-1, 28 ** 2)
        with open(
                os.path.join(self.root, MNIST.files['test_labels']), 'rb'
        ) as f:
            self.data['test_labels'] = np.frombuffer(
                f.read(), np.uint8, offset=8
            )
        print(">> Dataset loaded")

    def _dataset_exists(self):
        if os.path.exists(os.path.join(self.root)):
            for file in MNIST.files.values():
                if not os.path.isfile(os.path.join(self.root, file)):
                    return False
            return True
        else:
            return False

    def download(self):
        if self._dataset_exists():
            print(">> Dataset already exists. ")
            return
        else:
            raise ValueError(">> Failed to load the set, file not exist. You should download the dataset firstly.")
            pass


class FashionMNIST(Dataset):
    r"""
    A 10-class multi-class classfication
    """
    class_number = 10
    maxNum = 28*28
    resources = [
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
    ]

    files = {
        "train_images": 'train-images-idx3-ubyte',
        "test_images": 't10k-images-idx3-ubyte',
        "train_labels": 'train-labels-idx1-ubyte',
        "test_labels": 't10k-labels-idx1-ubyte',
    }

    def __init__(self, root, is_train=True):

        super().__init__()
        self.root = root
        self._is_train = is_train
        self.data = {}
        if not isinstance(self._is_train, bool):
            raise TypeError(">> is_train should be boolean value")
        if self._dataset_exists():
            self._to_numpy_format()
        else:
            self.download()

    def __getitem__(self, index):
        # 数据归一化到[0,1]
        if self._is_train:
            img = np.float32(self.data['train_images'][index])
            label = np.int64(self.data['train_labels'][index])
        else:
            img = np.float32(self.data['test_images'][index])
            label = np.int64(self.data['test_labels'][index])

        return img / 255.0, label

    def __len__(self):
        if self._is_train:
            return len(self.data['train_images'])
        else:
            return len(self.data['test_images'])

    @property
    def dataset_folder(self):
        return os.path.join(self.root, self.__class__.__name__)

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    @property
    def data_dict(self):
        return self.data

    @property
    def is_train(self):
        return self._is_train

    @is_train.setter
    def is_train(self, is_train):
        assert is_train in [True, False], ">> Invalid is_train setting"
        self._is_train = is_train

    def _to_numpy_format(self):
        with open(
                os.path.join(self.root, FashionMNIST.files['train_images']), 'rb'
        ) as f:
            self.data['train_images'] = np.frombuffer(
                f.read(), np.uint8, offset=16
            ).reshape(-1, 28 ** 2)
        with open(
                os.path.join(self.root, FashionMNIST.files['train_labels']), 'rb'
        ) as f:
            self.data['train_labels'] = np.frombuffer(
                f.read(), np.uint8, offset=8
            )
        with open(
                os.path.join(self.root, FashionMNIST.files['test_images']), 'rb'
        ) as f:
            self.data['test_images'] = np.frombuffer(
                f.read(),
                np.uint8,
                offset=16
            ).reshape(-1, 28 ** 2)
        with open(
                os.path.join(self.root, FashionMNIST.files['test_labels']), 'rb'
        ) as f:
            self.data['test_labels'] = np.frombuffer(
                f.read(), np.uint8, offset=8
            )
        print(">> Dataset loaded")

    def _dataset_exists(self):
        if os.path.exists(os.path.join(self.root)):
            for file in FashionMNIST.files.values():
                if not os.path.isfile(os.path.join(self.root, file)):
                    return False
            return True
        else:
            return False


class PathMNIST(Dataset):
    r"""
    A 9-class multi-class classfication
    """
    resources = 'https://drive.google.com/drive/folders/1Tl_SP-ffDQg-jDG_EWPlWKgZTmGbvFXU'
    class_number = 9
    maxNum = 28*28*3
    def __init__(self, root, is_train=True):
        super().__init__()
        self.root = root
        self._is_train = is_train
        self.data = {
            'train_images': [],
            'test_images': [],
            'train_labels': [],
            'test_labels': []
        }
        if not isinstance(self._is_train, bool):
            raise TypeError(">> is_train should be boolean value")
        if self._dataset_exists():
            self._to_numpy_format()
        else:
            print(">> Failed to load the set, file not exist. You should download the dataset firstly.")

    def __getitem__(self, index):
        if self._is_train:
            img = np.float32(self.data['train_images'][index].flatten() / 255)
            label = np.int64(self.data['train_labels'][index])
            return img, label
        else:
            img = np.float32(self.data['test_images'][index].flatten() / 255)
            label = np.int64(self.data['test_labels'][index])
            return img, label

    def __len__(self, ):
        if self._is_train:
            return len(self.data['train_images'])
        else:
            return len(self.data['test_images'])

    @property
    def data_dict(self):
        return self.data

    @property
    def is_train(self):
        return self._is_train

    @is_train.setter
    def is_train(self, is_train):
        assert is_train in [True, False], ">> Invalid is_train setting"
        self._is_train = is_train

    def _to_numpy_format(self):
        unzip_data = np.load(os.path.join(self.root, 'pathmnist.npz'))
        self.data['train_images'] = unzip_data['train_images']
        self.data['test_images'] = unzip_data['test_images']
        self.data['train_labels'] = unzip_data['train_labels'].squeeze()
        self.data['test_labels'] = unzip_data['test_labels'].squeeze()
        return

    def _dataset_exists(self):
        if os.path.isfile(os.path.join(self.root, 'pathmnist.npz')):
            return True
        else:
            return False


class OctMNIST(Dataset):
    r"""
    A 4-class multi-class classfication
    """
    resources = 'https://drive.google.com/drive/folders/1Tl_SP-ffDQg-jDG_EWPlWKgZTmGbvFXU'
    class_number = 4
    maxNum = 28 * 28

    def __init__(self, root, is_train=True):
        super().__init__()
        self.root = root
        self._is_train = is_train
        self.data = {
            'train_images': [],
            'test_images': [],
            'train_labels': [],
            'test_labels': []
        }
        if not isinstance(self._is_train, bool):
            raise TypeError(">> is_train should be boolean value")
        if self._dataset_exists():
            self._to_numpy_format()
        else:
            print(">> Failed to load the set, file not exist. You should download the dataset firstly.")

    def __getitem__(self, index):
        if self._is_train:
            img = np.float32(self.data['train_images'][index].flatten() / 255)
            label = np.int64(self.data['train_labels'][index])
            return img, label
        else:
            img = np.float32(self.data['test_images'][index].flatten() / 255)
            label = np.int64(self.data['test_labels'][index])
            return img, label

    def __len__(self, ):
        if self._is_train:
            return len(self.data['train_images'])
        else:
            return len(self.data['test_images'])

    @property
    def data_dict(self):
        return self.data

    @property
    def is_train(self):
        return self._is_train

    @is_train.setter
    def is_train(self, is_train):
        assert is_train in [True, False], ">> Invalid is_train setting"
        self._is_train = is_train

    def _to_numpy_format(self):
        unzip_data = np.load(os.path.join(self.root, 'octmnist.npz'))
        self.data['train_images'] = unzip_data['train_images']
        self.data['test_images'] = unzip_data['test_images']
        self.data['train_labels'] = unzip_data['train_labels'].squeeze()
        self.data['test_labels'] = unzip_data['test_labels'].squeeze()
        return

    def _dataset_exists(self):
        if os.path.isfile(os.path.join(self.root, 'octmnist.npz')):
            return True
        else:
            return False


class RWCP10(Dataset):
    r"""
    """
    classes = {
        "ring": 0,
        "whistle1": 1,
        "phone4": 2,
        "cymbals": 3,
        "horn": 4,
        "bells5": 5,
        "buzzer": 6,
        "kara": 7,
        "metal15": 8,
        "bottle1": 9
    }
    class_number = 10

    def __init__(self, root, is_train=True, **kwargs):
        super().__init__()
        self.scale = kwargs.get('scale', 0.1)
        preprocessing = kwargs.get('preprocessing', 'mfcc')
        self.preprocessing = preprocessing.lower()
        self.root = root
        npz_name = kwargs.get('npz_name', ('mfcc_feature', 'kp_feature'))

        self._is_train = is_train
        self.data = {
            'train_audios': [],
            'test_audios': [],
            'train_ids': [],
            'test_ids': [],
            'train_labels': [],
            'test_labels': [],
            'Time': [],
            'neuron_num': []
        }
        if not isinstance(self._is_train, bool):
            raise TypeError(">> is_train should be boolean value")

        if self._dataset_exists():
            self._npz_exists(npz_name)

            if self.npz_name == 'kp_feature.npz':
                self.data = load_kp_data(self.root, self.npz_name)
                self.maxTime = int(np.ceil(self.data['Time'] * self.scale))
                self.maxNum = int(self.data['neuron_num'])

            elif self.npz_name == 'mfcc_feature.npz':
                self.data = load_mfcc_data(self.root, self.npz_name)
                self.maxTime = 50
                self.maxNum = int(self.data['neuron_num'])

            # 如果npz_name不存在
            else:
                if self._classfile_exists():
                    if self.preprocessing == 'kp':
                        self.npz_name = save_kp_feature(root=self.root, npz_name=self.npz_name, sample_rate=16e3,
                                                        class_labels=RWCP10.classes)
                        self.data = load_kp_data(self.root, self.npz_name)
                        self.maxTime = int(np.ceil(self.data['Time'] * self.scale))
                        self.maxNum = int(self.data['neuron_num'])
                    elif self.preprocessing == 'mfcc':
                        self.npz_name = save_mfcc_feature(root=self.root, npz_name=self.npz_name, sample_rate=16e3,
                                                          class_labels=RWCP10.classes)
                        self.data = load_mfcc_data(self.root, self.npz_name)
                        self.maxTime = 50
                        self.maxNum = int(self.data['neuron_num'])
                    else:
                        print(">> Wrong preprocessing method. Please select kp or mfcc")

        else:
            print(">> Wrong root dir path. Please specify the dir path of the dataset")

    def __getitem__(self, index):
        if self._is_train:
            if self.npz_name == 'kp_feature.npz':
                spiking = [self.data['train_audios'][index] * self.scale, self.data['train_ids'][index]]
            else:
                spiking = (self.data['train_audios'][index]).astype(float)
            label = np.int64(self.data['train_labels'][index])
        else:
            if self.npz_name == 'kp_feature.npz':
                spiking = [self.data['test_audios'][index] * self.scale, self.data['test_ids'][index]]
            else:
                spiking = (self.data['test_audios'][index]).astype(float)
            label = np.int64(self.data['test_labels'][index])
        return spiking, label

    def __len__(self):
        if self._is_train:
            return len(self.data['train_audios'])
        else:
            return len(self.data['test_audios'])

    @property
    def data_dict(self):
        return self.data

    @property
    def is_train(self):
        return self._is_train

    @is_train.setter
    def is_train(self, is_train):
        assert is_train in [True, False], ">> Invalid is_train setting"
        self._is_train = is_train

    def _dataset_exists(self):
        if os.path.exists(self.root):
            return True
        else:
            return False

    # 判断原始数据是否存在
    def _classfile_exists(self):
        for cls_id in RWCP10.classes.keys():
            if not os.path.isdir(os.path.join(self.root, 'test', cls_id)):
                return False
            if not os.path.isdir(os.path.join(self.root, 'train', cls_id)):
                return False
        return True

    def _npz_exists(self, npz_name):
        file_name = os.listdir(self.root)
        if npz_name == ('mfcc_feature', 'kp_feature'):
            if 'mfcc_feature.npz' in file_name:
                self.npz_name = 'mfcc_feature.npz'
            elif 'kp_feature.npz' in file_name:
                self.npz_name = 'kp_feature.npz'
            else:
                self.npz_name = ''
        else:
            if npz_name in file_name:
                self.npz_name = npz_name
            else:
                self.npz_name = ''


class MNISTVoices(Dataset):
    r"""
    Used to load any type of 0-9 audio dataset
    Class number: 10
    """
    class_number = 10
    classes = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9
    }

    def __init__(self, root, is_train=True, **kwargs):
        super().__init__()
        self.scale = kwargs.get('scale', 0.1)
        preprocessing = kwargs.get('preprocessing', 'mfcc')
        self.preprocessing = preprocessing.lower()
        self.root = root
        npz_name = kwargs.get('npz_name', ('mfcc_feature', 'kp_feature'))

        self._is_train = is_train
        self.data = {
            'train_audios': [],
            'test_audios': [],
            'train_ids': [],
            'test_ids': [],
            'train_labels': [],
            'test_labels': [],
            'Time': [],
            'neuron_num': []
        }
        if not isinstance(self._is_train, bool):
            raise TypeError(">> is_train should be boolean value")

        if self._dataset_exists():
            self._npz_exists(npz_name)

            if self.npz_name == 'kp_feature.npz':
                self.data = load_kp_data(self.root, self.npz_name)
                self.maxTime = int(np.ceil(self.data['Time'] * self.scale))
                self.maxNum = int(self.data['neuron_num'])

            elif self.npz_name == 'mfcc_feature.npz':
                self.data = load_mfcc_data(self.root, self.npz_name)
                self.maxTime = 50
                self.maxNum = int(self.data['neuron_num'])

            # 如果npz_name不存在
            else:
                if self._classfile_exists():
                    if self.preprocessing == 'kp':
                        self.npz_name = save_kp_feature(root=self.root, npz_name=self.npz_name, sample_rate=16e3,
                                                   class_labels=MNISTVoices.classes)
                        self.data = load_kp_data(self.root, self.npz_name)
                        self.maxTime = int(np.ceil(self.data['Time'] * self.scale))
                        self.maxNum = int(self.data['neuron_num'])
                    elif self.preprocessing == 'mfcc':
                        self.npz_name = save_mfcc_feature(root=self.root, npz_name=self.npz_name, sample_rate=16e3,
                                                     class_labels=MNISTVoices.classes)
                        self.data = load_mfcc_data(self.root, self.npz_name)
                        self.maxTime = 50
                        self.maxNum = int(self.data['neuron_num'])
                    else:
                        print(">> Wrong preprocessing method. Please select kp or mfcc")

        else:
            print(">> Wrong root dir path. Please specify the dir path of the dataset")

    def __getitem__(self, index):
        if self._is_train:
            if self.npz_name == 'kp_feature.npz':
                spiking = [self.data['train_audios'][index] * self.scale, self.data['train_ids'][index]]
            else:
                spiking = (self.data['train_audios'][index]).astype(float)
            label = np.int64(self.data['train_labels'][index])
        else:
            if self.npz_name == 'kp_feature.npz':
                spiking = [self.data['test_audios'][index] * self.scale, self.data['test_ids'][index]]
            else:
                spiking = (self.data['test_audios'][index]).astype(float)
            label = np.int64(self.data['test_labels'][index])
        return spiking, label

    def __len__(self):
        if self._is_train:
            return len(self.data['train_audios'])
        else:
            return len(self.data['test_audios'])

    @property
    def data_dict(self):
        return self.data

    @property
    def is_train(self):
        return self._is_train

    @is_train.setter
    def is_train(self, is_train):
        assert is_train in [True, False], ">> Invalid is_train setting"
        self._is_train = is_train

    def _dataset_exists(self):
        if os.path.exists(self.root):
            return True
        else:
            return False

    # 判断原始数据是否存在
    def _classfile_exists(self):
        for cls_id in MNISTVoices.classes.keys():
            if not os.path.isdir(os.path.join(self.root, 'test', cls_id)):
                return False
            if not os.path.isdir(os.path.join(self.root, 'train', cls_id)):
                return False
        return True

    def _npz_exists(self, npz_name):
        file_name = os.listdir(self.root)
        if npz_name == ('mfcc_feature', 'kp_feature'):
            if 'mfcc_feature.npz' in file_name:
                self.npz_name = 'mfcc_feature.npz'
            elif 'kp_feature.npz' in file_name:
                self.npz_name = 'kp_feature.npz'
            else:
                self.npz_name = ''
        else:
            if npz_name in file_name:
                self.npz_name = npz_name
            else:
                self.npz_name = ''



class TIDIGITS(Dataset):
    r"""
    Used to load any type of 0-9 and oh audio dataset
    Class number: 11
    """
    class_number = 11
    classes = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "oh": 10
    }

    def __init__(self, root, is_train=True, **kwargs):
        super().__init__()
        self.scale = kwargs.get('scale', 0.1)
        preprocessing = kwargs.get('preprocessing', 'mfcc')
        self.preprocessing = preprocessing.lower()
        self.root = root
        npz_name = kwargs.get('npz_name', ('mfcc_feature', 'kp_feature'))

        self._is_train = is_train
        self.data = {
            'train_audios': [],
            'test_audios': [],
            'train_ids': [],
            'test_ids': [],
            'train_labels': [],
            'test_labels': [],
            'Time': [],
            'neuron_num': []
        }
        if not isinstance(self._is_train, bool):
            raise TypeError(">> is_train should be boolean value")

        if self._dataset_exists():
            self._npz_exists(npz_name)

            if self.npz_name == 'kp_feature.npz':
                self.data = load_kp_data(self.root, self.npz_name)
                self.maxTime = int(np.ceil(self.data['Time'] * self.scale))
                self.maxNum = int(self.data['neuron_num'])

            elif self.npz_name == 'mfcc_feature.npz':
                self.data = load_mfcc_data(self.root, self.npz_name)
                self.maxTime = 40
                self.maxNum = int(self.data['neuron_num'])

            # 如果npz_name不存在
            else:
                if self._classfile_exists():
                    if self.preprocessing == 'kp':
                        self.npz_name = save_kp_feature(root=self.root, npz_name=self.npz_name, sample_rate=16e3,
                                                   class_labels=TIDIGITS.classes)
                        self.data = load_kp_data(self.root, self.npz_name)
                        self.maxTime = int(np.ceil(self.data['Time'] * self.scale))
                        self.maxNum = int(self.data['neuron_num'])
                    elif self.preprocessing == 'mfcc':
                        self.npz_name = save_mfcc_feature(root=self.root, npz_name=self.npz_name, sample_rate=20e3,
                                                          signal_num=20e3, class_labels=TIDIGITS.classes)
                        self.data = load_mfcc_data(self.root, self.npz_name)
                        self.maxTime = 40
                        self.maxNum = int(self.data['neuron_num'])
                    else:
                        print(">> Wrong preprocessing method. Please select kp or mfcc")

        else:
            print(">> Wrong root dir path. Please specify the dir path of the dataset")

    def __getitem__(self, index):
        if self._is_train:
            if self.npz_name == 'kp_feature.npz':
                spiking = [self.data['train_audios'][index] * self.scale, self.data['train_ids'][index]]
            else:
                spiking = (self.data['train_audios'][index]).astype(float)
            label = np.int64(self.data['train_labels'][index])
        else:
            if self.npz_name == 'kp_feature.npz':
                spiking = [self.data['test_audios'][index] * self.scale, self.data['test_ids'][index]]
            else:
                spiking = (self.data['test_audios'][index]).astype(float)
            label = np.int64(self.data['test_labels'][index])
        return spiking, label

    def __len__(self):
        if self._is_train:
            return len(self.data['train_audios'])
        else:
            return len(self.data['test_audios'])

    @property
    def data_dict(self):
        return self.data

    @property
    def is_train(self):
        return self._is_train

    @is_train.setter
    def is_train(self, is_train):
        assert is_train in [True, False], ">> Invalid is_train setting"
        self._is_train = is_train

    def _dataset_exists(self):
        if os.path.exists(self.root):
            return True
        else:
            return False

    # 判断原始数据是否存在
    def _classfile_exists(self):
        for cls_id in TIDIGITS.classes.keys():
            if not os.path.isdir(os.path.join(self.root, 'test', cls_id)):
                return False
            if not os.path.isdir(os.path.join(self.root, 'train', cls_id)):
                return False
        return True

    def _npz_exists(self, npz_name):
        file_name = os.listdir(self.root)
        if npz_name == ('mfcc_feature', 'kp_feature'):
            if 'mfcc_feature.npz' in file_name:
                self.npz_name = 'mfcc_feature.npz'
            elif 'kp_feature.npz' in file_name:
                self.npz_name = 'kp_feature.npz'
            else:
                self.npz_name = ''
        else:
            if npz_name in file_name:
                self.npz_name = npz_name
            else:
                self.npz_name = ''




class SHD(Dataset):
    '''
    Spiking Heidelberg Digits Dataset
    max spiking time: 136.9 ms
    max neuron num: 700
    Class number: 20
    number of train samples: 8156
    number of test samples: 2264
    '''

    class_number = 20
    maxNum = 700
    Time = 1.37
    files = {
        "train_dataset": 'shd_train.h5',
        "test_dataset": 'shd_test.h5'
    }
    def __init__(self, root, is_train=True, **kwargs):
        super().__init__()
        self.scale = kwargs.get('scale', 100)
        self.maxTime = int(SHD.Time * self.scale)
        self.root = root
        self._is_train = is_train
        self.data = {
            'train_spiking': [],
            'test_spiking': [],
            'train_ids': [],
            'test_ids': [],
            'train_labels': [],
            'test_labels': []
        }
        if not isinstance(self._is_train, bool):
            raise TypeError(">> is_train should be boolean value")
        if self._dataset_exists():
            self._to_numpy_format()
        else:
            raise ValueError(">> Failed to load the set, file not exist. You should download the dataset firstly.")

    def __getitem__(self, index):
        if self._is_train:
            spiking = [self.data['train_spiking'][index] * self.scale, self.data['train_ids'][index]]
            label = np.int64(self.data['train_labels'][index])
        else:
            spiking = [self.data['test_spiking'][index] * self.scale, self.data['test_ids'][index]]
            label = np.int64(self.data['test_labels'][index])
        return spiking, label


    def __len__(self):
        if self._is_train:
            return len(self.data['train_spiking'])
        else:
            return len(self.data['test_spiking'])

    @property
    def dataset_folder(self):
        return os.path.join(self.root, self.__class__.__name__)

    @property
    def data_dict(self):
        return self.data

    @property
    def is_train(self):
        return self._is_train

    @is_train.setter
    def is_train(self, is_train):
        assert is_train in [True, False], ">> Invalid is_train setting"
        self._is_train = is_train

    def _to_numpy_format(self):
        # trainset
        # import tables
        import h5py
        if self._is_train:
            # train_fileh = tables.open_file(os.path.join(self.root, SHD.files['train_dataset']), mode='r')
            train_fileh = h5py.File(os.path.join(self.root, SHD.files['train_dataset']), 'r')
            neuron_ids = train_fileh['spikes']['units'][:]
            spike_times = train_fileh['spikes']['times'][:]
            labels = train_fileh['labels'][:]
            # neuron_ids = train_fileh.root.spikes.units
            # spike_times = train_fileh.root.spikes.times
            # labels = np.array(train_fileh.root.labels)

            self.data['train_spiking'] = spike_times
            self.data['train_ids'] = neuron_ids
            self.data['train_labels'] = labels

        else:
            # testset
            # test_fileh = tables.open_file(os.path.join(self.root, SHD.files['test_dataset']), mode='r')
            test_fileh = h5py.File(os.path.join(self.root, SHD.files['test_dataset']), 'r')
            neuron_ids = test_fileh['spikes']['units'][:]
            spike_times = test_fileh['spikes']['times'][:]
            labels = test_fileh['labels'][:]
            # neuron_ids = test_fileh.root.spikes.units
            # spike_times = test_fileh.root.spikes.times
            # labels = np.array(test_fileh.root.labels)

            self.data['test_spiking'] = spike_times
            self.data['test_ids'] = neuron_ids
            self.data['test_labels'] = labels

        print(">> Dataset loaded")

    def _dataset_exists(self):
        if os.path.exists(os.path.join(self.root)):
            for file in SHD.files.values():
                if not os.path.isfile(os.path.join(self.root, file)):
                    return False
            return True
        else:
            return False


class SSC(Dataset):
    '''
    Spiking Speech Command Dataset
    max spiking time: 99.95ms
    max neuron num: 700
    Class number: 35
    number of train samples: 75466
    number of test samples: 30363
    '''

    class_number = 35
    maxNum = 700
    Time = 1
    files = {
        "train_dataset": 'ssc_train.h5',
        "valid_dataset": 'ssc_valid.h5',
        "test_dataset": 'ssc_test.h5'
    }

    def __init__(self, root, is_train=True, **kwargs):
        super().__init__()
        self.scale = kwargs.get('scale', 100)
        self.maxTime = int(SSC.Time * self.scale)
        self.root = root
        self._is_train = is_train
        self.data = {
            'train_spiking': [],
            'test_spiking': [],
            'train_ids': [],
            'test_ids': [],
            'train_labels': [],
            'test_labels': []
        }
        if not isinstance(self._is_train, bool):
            raise TypeError(">> is_train should be boolean value")
        if self._dataset_exists():
            self._to_numpy_format()
        else:
            raise ValueError(">> Failed to load the set, file not exist. You should download the dataset firstly.")


    def __getitem__(self, index):
        if self._is_train:
            spiking = [self.data['train_spiking'][index] * self.scale, self.data['train_ids'][index]]
            label = np.int64(self.data['train_labels'][index])
        else:
            spiking = [self.data['test_spiking'][index] * self.scale, self.data['test_ids'][index]]
            label = np.int64(self.data['test_labels'][index])
        return spiking, label

    def __len__(self):
        if self._is_train:
            return len(self.data['train_spiking'])
        else:
            return len(self.data['test_spiking'])

    @property
    def dataset_folder(self):
        return os.path.join(self.root, self.__class__.__name__)

    @property
    def data_dict(self):
        return self.data

    @property
    def is_train(self):
        return self._is_train

    @is_train.setter
    def is_train(self, is_train):
        assert is_train in [True, False], ">> Invalid is_train setting"
        self._is_train = is_train

    def _to_numpy_format(self):
        # import tables
        import h5py

        # trainset
        if self._is_train:
            # train_fileh = tables.open_file(os.path.join(self.root, SSC.files['train_dataset']), mode='r')
            train_fileh = h5py.File(os.path.join(self.root, SSC.files['train_dataset']), 'r')
            neuron_ids = train_fileh['spikes']['units'][:]
            spike_times = train_fileh['spikes']['times'][:]
            labels = train_fileh['labels'][:]
            # neuron_ids = train_fileh.root.spikes.units
            # spike_times = train_fileh.root.spikes.times
            # labels = np.array(train_fileh.root.labels)

            self.data['train_spiking'] = spike_times
            self.data['train_ids'] = neuron_ids
            self.data['train_labels'] = labels

        else:
            # test_fileh = tables.open_file(os.path.join(self.root, SSC.files['test_dataset']), mode='r')
            test_fileh = h5py.File(os.path.join(self.root, SSC.files['test_dataset']), 'r')
            neuron_ids = test_fileh['spikes']['units'][:]
            spike_times = test_fileh['spikes']['times'][:]
            labels = test_fileh['labels'][:]
            # neuron_ids = test_fileh.root.spikes.units
            # spike_times = test_fileh.root.spikes.times
            # labels = np.array(test_fileh.root.labels)

            self.data['test_spiking'] = spike_times
            self.data['test_ids'] = neuron_ids
            self.data['test_labels'] = labels

        print(">> Dataset loaded")

    def _dataset_exists(self):
        if os.path.exists(os.path.join(self.root)):
            for file in SSC.files.values():
                if not os.path.isfile(os.path.join(self.root, file)):
                    return False
            return True
        else:
            return False

class DVS128Gesture(Dataset):
    resources = 'https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794'

    files = {
        "train_dataset": 'trials_to_train.txt',
        "test_dataset": 'trials_to_test.txt'
    }

    def __init__(self, root, is_train=True):
        super().__init__(root, is_train)
        self.root = root
        self._is_train = is_train
        self.data = {
            'train_spiking': [],
            'test_spiking': [],
            'train_labels': [],
            'test_labels': []
        }
        if not isinstance(self._is_train, bool):
            raise TypeError(">> is_train should be boolean value")
        if self._dataset_exists():
            self._to_numpy_format()
        else:
            raise ValueError(">> Failed to load the set, file not exist. You should download the dataset firstly.")


    def __getitem__(self, index):
        if self._is_train:
            spiking = [self.data['train_spiking'][index], self.data['train_ids'][index]]
            label = np.int64(self.data['train_labels'][index])
        else:
            spiking = [self.data['test_spiking'][index], self.data['test_ids'][index]]
            label = np.int64(self.data['test_labels'][index])
        return spiking, label

    def __len__(self):
        if self._is_train:
            return len(self.data['train_spiking'])
        else:
            return len(self.data['test_spiking'])

    @property
    def dataset_folder(self):
        return os.path.join(self.root, self.__class__.__name__)

    @property
    def data_dict(self):
        return self.data

    @property
    def is_train(self):
        return self._is_train

    @is_train.setter
    def is_train(self, is_train):
        assert is_train in [True, False], ">> Invalid is_train setting"
        self._is_train = is_train

    def _to_numpy_format(self):
        # import tables
        import h5py
        # trainset
        if self._is_train:
            # train_file = tables.open_file(os.path.join(self.root, DVS128Gesture.files['train_dataset']), mode='r')
            train_file = h5py.File(os.path.join(self.root, DVS128Gesture.files['train_dataset']), 'r')

        else:
            # test_file = tables.open_file(os.path.join(self.root, DVS128Gesture.files['test_dataset']), mode='r')
            test_file = h5py.File(os.path.join(self.root, DVS128Gesture.files['test_dataset']), 'r')

        print(">> Dataset loaded")


    @staticmethod
    def split_aedat_files_to_np(fname: str, aedat_file: str, csv_file: str, output_dir: str):
        events = load_aedat_v3(aedat_file)
        print(f'Start to split [{aedat_file}] to samples.')
        # Read csv file and get time stamp and label of each sample. Then split the origin data to samples
        csv_data = np.loadtxt(csv_file, dtype=np.uint32, delimiter=',', skiprows=1)

        # Note that there are some files that many samples have the same label, e.g., user26_fluorescent_labels.csv
        label_file_num = [0] * 11

        for i in range(csv_data.shape[0]):
            # the label of DVS128 Gesture is 1, 2, ..., 11. We set 0 as the first label, rather than 1
            label = csv_data[i][0] - 1
            t_start = csv_data[i][1]
            t_end = csv_data[i][2]
            mask = np.logical_and(events['t'] >= t_start, events['t'] < t_end)
            file_name = os.path.join(output_dir, str(label), f'{fname}_{label_file_num[label]}.npz')
            np.savez(file_name,
                     t=events['t'][mask],
                     x=events['x'][mask],
                     y=events['y'][mask],
                     p=events['p'][mask]
                     )
            print(f'[{file_name}] saved.')
            label_file_num[label] += 1

    def create_events_np_files(self, events_np_root: str):
        '''
        :param events_np_root: Root directory path which saves events files in the ``npz`` format
        :type events_np_root: str

        This function defines how to convert the origin binary data in ``extract_root`` to ``npz`` format and save converted files in ``events_np_root``.
        '''

        train_dir = os.path.join(events_np_root, 'train')
        test_dir = os.path.join(events_np_root, 'test')
        os.mkdir(train_dir)
        os.mkdir(test_dir)
        print(f'Mkdir [{train_dir, test_dir}.')
        for label in range(11):
            os.mkdir(os.path.join(train_dir, str(label)))
            os.mkdir(os.path.join(test_dir, str(label)))
        print(f'Mkdir {os.listdir(train_dir)} in [{train_dir}] and {os.listdir(test_dir)} in [{test_dir}].')

        with open(os.path.join(self.root, DVS128Gesture.files['train_dataset'])) as trials_to_train_txt, open(
                os.path.join(self.root, DVS128Gesture.files['test_dataset'])) as trials_to_test_txt:
            # use multi-thread to accelerate
            # t_ckp = time.time()
            with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 64)) as tpe:
                print(f'Start the ThreadPoolExecutor with max workers = [{tpe._max_workers}].')

                for fname in trials_to_train_txt.readlines():
                    fname = fname.strip()
                    if fname.__len__() > 0:
                        aedat_file = os.path.join(self.root, fname)
                        fname = os.path.splitext(fname)[0]
                        tpe.submit(DVS128Gesture.split_aedat_files_to_np, fname, aedat_file, os.path.join(self.root, fname + '_labels.csv'), train_dir)

                for fname in trials_to_test_txt.readlines():
                    fname = fname.strip()
                    if fname.__len__() > 0:
                        aedat_file = os.path.join(self.root, fname)
                        fname = os.path.splitext(fname)[0]
                        tpe.submit(DVS128Gesture.split_aedat_files_to_np, fname, aedat_file,
                                   os.path.join(self.root, fname + '_labels.csv'), test_dir)

            # print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
        print(f'All aedat files have been split to samples and saved into [{train_dir, test_dir}].')