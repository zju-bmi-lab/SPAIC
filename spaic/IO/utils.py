"""
@author: Yuan Mengwen
@contact: mwyuan94@gmail.com
@project: PyCharm
@filename: utils.py
@time:2021/4/1 14:33
@description:
"""
import os
from random import shuffle
from shutil import copy
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
from scipy.ndimage import maximum_filter
# import soundfile as sf
# import cv2

import struct
from spaic.utils import plot, gtgram
import matplotlib.pyplot as plt
'''
==================audio preprocess method==================
'''
def wav_file_resample(file_path, dest_sample=16e3):
    """
    对WAV文件进行resample的操作
    Args:
        file_path: 需要进行resample操作的wav文件的路径
        dest_sample:目标采样率
    Returns:
        resampled: 降采样后的数据
        dest_sample: 目标采样率
    """
    sample_rate, sound_signal = wav.read(file_path)
    signal_num = int((sound_signal.shape[0]) / sample_rate * dest_sample)
    resampled = signal.resample(sound_signal, signal_num)
    return resampled, dest_sample

def _dataset_exists(root, class_labels):
    if os.path.exists(root):
        for cls_id in class_labels.keys():
            if not os.path.isdir(os.path.join(root, 'test', cls_id)):
                return False
            if not os.path.isdir(os.path.join(root, 'train', cls_id)):
                return False
        return True
    else:
        return False

def save_numpy_format(root, is_train, sample_rate=16e3, class_labels=None, **kwargs):
    # parameters for extracting key points of audio
    window_size = kwargs.get('window_size', 0.016)
    stride = kwargs.get('stride', 0.008)
    kernels_num = kwargs.get('kernels_num', 100)
    freq_min = kwargs.get('freq_min', 20)
    Dr = kwargs.get('Dr', 3)
    Dc = kwargs.get('Dc', 3)
    significance_level = kwargs.get('significance_level', 3)

    # set class labels
    if class_labels is None:
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
    else:
        classes = class_labels

    # whether the data set exists
    if _dataset_exists(root, classes):
        print(">> The dataset exists")
    else:
        print(">> Wrong root dir path")

    subset = 'train' if is_train else 'test'
    data = {
        'train_audios': [],
        'train_labels': [],
        'test_audios': [],
        'test_labels': [],
        'train_ids': [],
        'test_ids': []
    }

    for cls in classes.keys():
        cur_dir = os.path.join(root, subset, cls)
        for file in os.listdir(cur_dir):
            if not file.endswith('wav'):
                continue

            wavform, sr = wav_file_resample(os.path.join(cur_dir, file), sample_rate)
            gmspec = fetchGmSpectrogram(wavform, sample_rate, window_size, stride, kernels_num, freq_min) # gtgram.gtgram(wavform, sample_rate, window_size, stride, kernels_num, freq_min, show=True)
            irow, icol, ival = extractKeyPoints(gmspec, Dr, Dc, significance_level)

            audios_name = "{}_audios".format(subset.lower())
            labels_name = "{}_labels".format(subset.lower())
            id_name = "{}_ids".format(subset.lower())
            data[audios_name].append(ival)
            data[labels_name].append(classes[cls])
            data[id_name].append(irow)

    # 将音频数据存储为.npz文件
    data_root = os.path.join(root, subset.lower())

    for k in data.keys():
        data[k] = np.array(data[k], dtype=object)
    np.savez(data_root, train_audios=data['train_audios'], train_labels=data['train_labels'], train_ids=data['train_ids'], test_audios=data['test_audios'], test_labels=data['test_labels'], test_ids=data['test_ids'])
    print(">> " + subset + "_dataset saved")

def load_audio_data(root, is_train):
    filename = 'train.npz' if is_train else 'test.npz'
    fileroot = os.path.join(root, filename)
    data = np.load(fileroot, allow_pickle=True)
    print(">> " + filename + " loaded")
    return data

def dataset_split(source_root, target_root, ratio, is_shuffle):
    train_root = target_root + "\\" + 'train'
    test_root = target_root + "\\" + 'test'

    if not os.path.isdir(train_root):
        os.makedirs(train_root)
    if not os.path.isdir(test_root):
        os.makedirs(test_root)

    for class_name in os.listdir(source_root):
        class_root = os.path.join(source_root, class_name)

        train_dir = os.path.join(train_root, class_name)
        test_dir = os.path.join(test_root, class_name)
        if not os.path.isdir(train_dir):
            os.makedirs(train_dir)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

        samples = os.listdir(class_root)
        samples_len = len(samples)

        if is_shuffle is True:
            shuffle(samples)

        # i 用来计算文件数量
        i = 0
        to_path = train_dir
        for data_name in samples:
            split_num = ratio*samples_len
            if i == 0:
                to_path = train_dir
            elif ((i % split_num) == 0):
                to_path = test_dir
            from_path = os.path.join(class_root, data_name)
            copy(from_path, to_path)
            i += 1

def reclassification(source_root, target_root, class_num, perperson_perclass_samplenum):
    '''
    将按录音者分类的digit语音数据集重保存为按录的音频的类别分类
    '''
    # 想保存到的根路径
    for i in range(class_num):
        save_dir = os.path.join(target_root, str(i))
        # 如果目录不存在，则创建
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    if os.path.exists(source_root):
        all_files = os.listdir(source_root)
    else:
        raise ValueError('The path'+ source_root +' is not exist')

    for file_name in all_files:
        file_root = os.path.join(source_root, file_name)
        samples = os.listdir(file_root)
        # i 用来计算文件数量，k 用来计算应存放到哪一类中
        i = 0
        k = 0
        for data_name in samples:
            if i == 0:
                k = 0
            elif ((i%perperson_perclass_samplenum)==0):
                k += 1
            from_path = os.path.join(file_root, data_name)
            to_path = target_root + "\\" + str(k)
            copy(from_path, to_path)
            i += 1

def datasetAlignment(source, maxNum):
    """
    Zeros are padded to each sample in the dataset according to the value of maxNum
    Args:
        source (ndarray): samples of dataset
        maxNum (int): The length of longest sample

    Returns:
        The data after padding

    """
    source = source.tolist()
    for n in range(len(source)):
        pad_len = maxNum-len(source[n])
        source[n] = np.pad(source[n], (0, pad_len), 'constant', constant_values=(0, 0))

    source = np.array(source)
    return source

def batchAlignment(source):
    source = source.tolist()
    maxNum = 0
    for n in range(len(source)):
        wav_sig = source[n]
        maxNum = max(maxNum, len(wav_sig))

    for n in range(len(source)):
        pad_len = maxNum-len(source[n])
        source[n] = np.pad(source[n], (0, pad_len), 'constant', constant_values=(0, 0))

    return source

def fetchGmSpectrogram(sig, fs=16e3, window_size=0.016, stride=0.008, kernels_num=32, freq_min=20, log=False, show=False):
    gmspec = gtgram.gtgram(sig, fs, window_size, stride, kernels_num, freq_min)
    if log:
        gmspec = np.log(gmspec)
    if show:
        p1 = plt.figure('spectrum', dpi=500)
        axes = p1.add_axes([0.1, 0.1, 0.9, 0.9])
        plot.gtgram_plot(gtgram.gtgram, axes, sig, fs, window_size, stride, kernels_num, freq_min)
        # plt.show()
        # print('')
    return gmspec

def extractKeyPoints(gmspec, Dr=13, Dc=13, significance_level=3):
    # print('begin extractKeyPoints')
    row_mask = np.ones([1, Dr])
    colum_mask = np.ones([Dc, 1])
    plus_mask = np.zeros([Dc, Dr])
    plus_mask[:, Dr >> 1] = 1
    plus_mask[Dc >> 1, :] = 1
    row_filter_spec = maximum_filter(gmspec, footprint=row_mask, mode='reflect')
    colum_filter_spec = maximum_filter(gmspec, footprint=colum_mask, mode='reflect')

    is_keypoint = np.logical_or((row_filter_spec == gmspec), (colum_filter_spec == gmspec))
    [irow, icol] = np.where(is_keypoint == True)
    ival = gmspec[irow, icol]
    # center_points = np.concatenate([irow,ival,])
    pad_gmspec = np.pad(gmspec, ((Dc >> 1, Dc >> 1), (Dr >> 1, Dr >> 1)), mode='symmetric')

    # todo: L默认填为矩形，其余位置由mask决定
    Lrow = np.empty([0, Dr])
    for i in range(irow.size):
        row_tmp = pad_gmspec[irow[i] + Dc >> 1, icol[i]:icol[i] + Dr].reshape([1, -1])
        Lrow = np.concatenate([Lrow, row_tmp], axis=0)
    Lcol = np.empty([Dc, 0])
    for i in range(icol.size):
        col_tmp = pad_gmspec[irow[i]:irow[i] + Dc, icol[i] + Dr >> 1].reshape([-1, 1])
        Lcol = np.concatenate([Lcol, col_tmp], axis=1)

    avg_Lrow = np.mean(Lrow, axis=1)
    avg_Lcol = np.mean(Lcol, axis=0)
    noise = np.zeros([ival.size])
    noise[avg_Lrow < avg_Lcol] = avg_Lrow[avg_Lrow < avg_Lcol]
    noise[avg_Lcol <= avg_Lrow] = avg_Lcol[avg_Lcol <= avg_Lrow]
    significant = ((ival - noise) > significance_level)

    # 去除噪点
    irow = irow[significant]
    icol = icol[significant]
    ival = ival[significant]
    print('..', sum(significant))
    return irow, icol, ival

def get_Max_Min(data):
    '''
    get the maximum and minimum number of data
    Args:
        data (): can be spiking time or neuron ids

    Returns:

    '''
    maxData = 0
    minData = float('inf')
    for i in range(len(data)):
        tempMax = max(data[i])
        maxData = max(maxData, tempMax)
        tempMin = min(data[i])
        minData = min(minData, tempMin)

    return maxData, minData
'''
==================image preprocess method==================
'''
def RGBtoGray(image):
    """
    Converts RGB image into gray image.

    Args:
        image: RGB image.
    Returns:
        Gray image.
    """
    import cv2
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def GraytoBinary(image):
    """
    Converts input image into black and white (binary)

    Args:
        image: Gray image.
    Returns:
        Binary image.
    """
    import cv2
    return cv2.threshold(image, 0, 1, cv2.THRESH_BINARY)[1]


def reshape(image, x, y):
    """
    Scale the image to (x, y).

    Args：
        image: Image to be rescaled.
        x: x dimension.
        y: y dimension.
    Returns:
        Re-scaled image.
    """
    return cv2.resize(image, (x, y))


'''
==================DVS preprocess method==================
'''
def load_aedat_v3(file_name: str):
    '''
    Args:
        file_name(str): path of the aedat v3 file
    Returns:
        a dict whose keys are ['t', 'x', 'y', 'p'] and values are ``numpy.ndarray``
    This function is written by referring to https://gitlab.com/inivation/dv/dv-python . It can be used for DVS128 Gesture.
    '''
    with open(file_name, 'rb') as bin_f:
        # skip ascii header
        line = bin_f.readline()
        while line.startswith(b'#'):
            if line == b'#!END-HEADER\r\n':
                break
            else:
                line = bin_f.readline()

        txyp = {
            't': [],
            'x': [],
            'y': [],
            'p': []
        }
        while True:
            header = bin_f.read(28)
            if not header or len(header) == 0:
                break

            # read header
            e_type = struct.unpack('H', header[0:2])[0]
            e_size = struct.unpack('I', header[4:8])[0]
            e_tsoverflow = struct.unpack('I', header[12:16])[0]
            e_capacity = struct.unpack('I', header[16:20])[0]

            data_length = e_capacity * e_size
            data = bin_f.read(data_length)
            counter = 0

            if e_type == 1:
                while data[counter:counter + e_size]:
                    aer_data = struct.unpack('I', data[counter:counter + 4])[0]
                    timestamp = struct.unpack('I', data[counter + 4:counter + 8])[0] | e_tsoverflow << 31
                    x = (aer_data >> 17) & 0x00007FFF
                    y = (aer_data >> 2) & 0x00007FFF
                    pol = (aer_data >> 1) & 0x00000001
                    counter = counter + e_size
                    txyp['x'].append(x)
                    txyp['y'].append(y)
                    txyp['t'].append(timestamp)
                    txyp['p'].append(pol)
            else:
                # non-polarity event packet, not implemented
                pass
        txyp['x'] = np.asarray(txyp['x'])
        txyp['y'] = np.asarray(txyp['y'])
        txyp['t'] = np.asarray(txyp['t'])
        txyp['p'] = np.asarray(txyp['p'])
        return txyp


# if __name__ == "__main__":
    # sroot = r'C:\Users\hp\Desktop\SpeechMNIST'
    # troot = r'C:\Users\hp\Desktop\SpeechMNIST1'
    # dataset_split(sroot, troot, 0.7, True)

    # sroot = r'C:\Users\hp\Desktop\AudioMNIST'
    # troot = r'C:\Users\hp\Desktop\SpeechMNIST'
    # reclassification(sroot, troot, 10, 50)

    # time_now = time.time()
    # path = r'F:\GitCode\Python\dataset\AudioMNIST\train\0\0_01_2.wav'
    # sound, sampling_freq = wav_file_resample(path, 16e3)
    # print(time.time() - time_now)
    # time_now1 = time.time()
    # sound1, fs = librosa.load(path, sr=16e3)
    # print(time.time() - time_now1)

    # root = r'F:\GitCode\Python\datasets\AudioMNIST'
    # save_numpy_format(root, True, sample_rate=16e3)
    # save_numpy_format(root, False, sample_rate=16e3)

    # load_audio_data(root, True)
    # load_audio_data(root, False)
    # source = data['train_audios']
    # maxNum = data['maxNum']
    # datasetAlignment(source, maxNum)

    # root = r'F:\GitCode\Python\datasets\AudioMNIST'
    # filename = 'test.npz'
    # fileroot = os.path.join(root, filename)
    # data = np.load(fileroot, allow_pickle=True)
    # trian_audios0 = data['train_ids'][0]
    # trian_audios1 = data['train_ids'][1]
    # trian_audios2 = data['train_ids'][2]
    # trian_audios3 = data['train_ids'][3]
    # maxvalue, minvalue = get_Max_Min(data['train_ids'])
    # filename = 'test.npz'
    # test_audios0 = data['test_audios'][0]
    # test_audios1 = data['test_audios'][1]
    # test_audios2 = data['test_audios'][2]
    # test_audios3 = data['test_audios'][3]
    # test_audios = data['test_audios']
    # maxvalue, minvalue = get_Max_Min(data['test_ids'])
    # print(maxvalue, minvalue)
    # print('')

