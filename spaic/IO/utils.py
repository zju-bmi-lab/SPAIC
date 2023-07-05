"""
@author: Yuan Mengwen
@contact: mwyuan94@gmail.com
@project: PyCharm
@filename: utils.py
@time:2021/4/1 14:33
@description:
"""
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from random import shuffle
from shutil import copy
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
from scipy.ndimage import maximum_filter
# import soundfile as sf
# import cv2
import math

import struct
from ..utils import plot, gtgram
import matplotlib.pyplot as plt

import scipy.io.wavfile as wav

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


def wav_file_cut(file_path, signal_num=16e3):
    """
    对WAV文件进行裁剪操作
    Args:
        file_path: 需要进行resample操作的wav文件的路径
        signal_num:目标数据数量
    Returns:
        cropped_data: 裁剪后的数据

    """
    sample_rate, sound_signal = wav.read(file_path)
    cropped_data = signal.resample(sound_signal, int(signal_num))
    return cropped_data


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


def save_mfcc_feature(root, npz_name, sample_rate=16e3, signal_num=16e3, class_labels=None, **kwargs):
    from python_speech_features import mfcc
    if npz_name == '':
        feature_name = 'mfcc_feature'

    # set class labels
    if class_labels is None:
        raise ValueError('Missing class labels dict')
    else:
        classes = class_labels

    data = {
        'train_audios': [],
        'train_labels': [],
        'test_audios': [],
        'test_labels': [],
        'Time': [],
        'neuron_num': []
    }

    dataset_name = ['train', 'test']  # if is_train else 'test'
    for subset in dataset_name:
        for cls in classes.keys():
            cur_dir = os.path.join(root, subset, cls)
            for file in os.listdir(cur_dir):
                if not file.endswith('wav'):
                    continue
                wavform = wav_file_cut(os.path.join(cur_dir, file), signal_num)
                from python_speech_features import mfcc
                feature_mfcc = mfcc(wavform, samplerate=sample_rate)
                feature_mfcc = feature_mfcc.flatten()
                audios_name = "{}_audios".format(subset.lower())
                labels_name = "{}_labels".format(subset.lower())
                data[audios_name].append(feature_mfcc)
                data[labels_name].append(classes[cls])

    # 将音频数据存储为.npz文件
    data_root = os.path.join(root, feature_name)
    npz_name = feature_name + '.npz'
    data['neuron_num'] = len(data['train_audios'][0])
    trainMaxTime = get_Max(data['train_audios'])
    testMaxTime = get_Max(data['test_audios'])
    data['Time'] = max(trainMaxTime, testMaxTime)

    for k in data.keys():
        data[k] = np.array(data[k], dtype=object)
    np.savez(data_root, train_audios=data['train_audios'], train_labels=data['train_labels'],
             test_audios=data['test_audios'], test_labels=data['test_labels'], Time=data['Time'],
             neuron_num=data['neuron_num'])
    print(">> mfcc_features saved")
    return npz_name


def save_kp_feature(root=None, npz_name=None, sample_rate=16e3, class_labels=None, **kwargs):
    # parameters for extracting key points of audio
    window_size = kwargs.get('window_size', 0.016)
    stride = kwargs.get('stride', 0.008)
    kernels_num = kwargs.get('kernels_num', 100)
    freq_min = kwargs.get('freq_min', 20)
    Dr = kwargs.get('Dr', 3)
    Dc = kwargs.get('Dc', 3)
    significance_level = kwargs.get('significance_level', 3)
    if npz_name == '':
        feature_name = 'kp_feature'

    # set class labels
    if class_labels is None:
        raise ValueError('Missing class labels dict')
    else:
        classes = class_labels

    data = {
        'train_audios': [],
        'train_labels': [],
        'test_audios': [],
        'test_labels': [],
        'train_ids': [],
        'test_ids': [],
        'Time': [],
        'neuron_num': []
    }

    dataset_name = ['train', 'test']  # if is_train else 'test'
    for subset in dataset_name:
        for cls in classes.keys():
            cur_dir = os.path.join(root, subset, cls)
            for file in os.listdir(cur_dir):
                if not file.endswith('wav'):
                    continue

                wavform, sr = wav_file_resample(os.path.join(cur_dir, file), sample_rate)
                gmspec = fetchGmSpectrogram(wavform, sample_rate, window_size, stride, kernels_num,
                                            freq_min)  # gtgram.gtgram(wavform, sample_rate, window_size, stride, kernels_num, freq_min, show=True)
                irow, icol, ival = extractKeyPoints(gmspec, Dr, Dc, significance_level)
                audios_name = "{}_audios".format(subset.lower())
                labels_name = "{}_labels".format(subset.lower())
                id_name = "{}_ids".format(subset.lower())
                data[audios_name].append(ival)
                data[labels_name].append(classes[cls])
                data[id_name].append(irow)

    # 将音频数据存储为.npz文件
    data_root = os.path.join(root, feature_name)
    npz_name = feature_name + '.npz'
    data['neuron_num'] = kernels_num
    trainMaxTime = get_Max(data['train_audios'])
    testMaxTime = get_Max(data['test_audios'])
    data['Time'] = max(trainMaxTime, testMaxTime)

    for k in data.keys():
        data[k] = np.array(data[k], dtype=object)
    np.savez(data_root, train_audios=data['train_audios'], train_labels=data['train_labels'],
             train_ids=data['train_ids'],
             test_audios=data['test_audios'], test_labels=data['test_labels'], test_ids=data['test_ids'],
             Time=data['Time'], neuron_num=data['neuron_num'])
    print(">> kp_feature saved")
    return npz_name


def load_kp_data(root, filename):
    data = {
        'train_audios': [],
        'test_audios': [],
        'train_ids': [],
        'test_ids': [],
        'train_labels': [],
        'test_labels': [],
        'Time': [],
        'neuron_num': []
    }
    fileroot = os.path.join(root, filename)
    data_temp = np.load(fileroot, allow_pickle=True)
    data['train_audios'] = data_temp['train_audios']
    data['test_audios'] = data_temp['test_audios']
    data['train_labels'] = data_temp['train_labels']
    data['test_labels'] = data_temp['test_labels']
    data['train_ids'] = data_temp['train_ids']
    data['test_ids'] = data_temp['test_ids']
    data['Time'] = data_temp['Time']
    data['neuron_num'] = data_temp['neuron_num']
    print(">> " + filename + " loaded")
    return data


def load_mfcc_data(root, filename):
    data = {
        'train_audios': [],
        'test_audios': [],
        'train_labels': [],
        'test_labels': [],
        'Time': [],
        'neuron_num': []
    }
    fileroot = os.path.join(root, filename)
    data_temp = np.load(fileroot, allow_pickle=True)
    data['train_audios'] = data_temp['train_audios']
    data['test_audios'] = data_temp['test_audios']
    data['train_labels'] = data_temp['train_labels']
    data['test_labels'] = data_temp['test_labels']
    data['Time'] = data_temp['Time']
    data['neuron_num'] = data_temp['neuron_num']
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
            split_num = math.ceil(ratio * samples_len)
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
        raise ValueError('The path' + source_root + ' is not exist')

    for file_name in all_files:
        file_root = os.path.join(source_root, file_name)
        samples = os.listdir(file_root)
        # i 用来计算文件数量，k 用来计算应存放到哪一类中
        i = 0
        k = 0
        for data_name in samples:
            if i == 0:
                k = 0
            elif ((i % perperson_perclass_samplenum) == 0):
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
        pad_len = maxNum - len(source[n])
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
        pad_len = maxNum - len(source[n])
        source[n] = np.pad(source[n], (0, pad_len), 'constant', constant_values=(0, 0))

    return source


def fetchGmSpectrogram(sig, fs=16e3, window_size=0.016, stride=0.008, kernels_num=32, freq_min=20, log=False,
                       show=False):
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
    [irow, icol] = np.where(is_keypoint is True)
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


def get_Max(data):
    '''
    get the maximum number of data
    Args:
        data (): can be spiking time or neuron ids

    Returns:

    '''
    maxData = 0
    for i in range(len(data)):
        tempMax = max(data[i])
        maxData = max(maxData, tempMax)
    return maxData


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


def reshape(image, shape):
    """
    Scale the image to (x, y).

    Args：
        image: Image to be rescaled.
        shape: Changed shape
    Returns:
        Re-scaled image.
    """
    import cv2
    return cv2.resize(image, shape)


def im2col(img, kh, kw, stride, padding='same'):
    '''
    :param img: 4D array
    :param kh: kernel_height
    :param kw: kernel_width
    :param stride:
    :param padding:
    :return:
    '''
    if padding == 'same':
        p1 = kh // 2
        p2 = kw // 2
        img = np.pad(img, ((0, 0), (0, 0), (p1, p1), (p2, p2),), 'constant')
    N, C, H, W = img.shape
    out_h = (H - kh) // stride[0] + 1
    out_w = (W - kw) // stride[1] + 1
    outsize = out_w * out_h
    col = np.empty((N, C, kw * kh, outsize,))
    for y in range(out_h):
        y_start = y * stride[0]
        y_end = y_start + kh
        for x in range(out_w):
            x_start = x * stride[1]
            x_end = x_start + kw
            col[:, :, 0:, y * out_w + x] = img[:, :, y_start:y_end, x_start:x_end].reshape(N, C, kh * kw)
    return col.reshape(N, -1, outsize)


def un_tar(file_name, output_root):
    # untar zip file to folder whose name is same as tar file
    import tarfile
    tar = tarfile.open(file_name)
    names = tar.getnames()

    file_name = os.path.basename(file_name)
    extract_dir = os.path.join(output_root, file_name.split('.')[0])

    # create folder if nessessary
    if os.path.isdir(extract_dir):
        pass
    else:
        os.makedirs(extract_dir)

    file_list = os.listdir(extract_dir)
    if len(file_list) == len(names):
        pass
    else:
        for name in names:
            tar.extract(name, extract_dir)
    tar.close()


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


def create_same_directory_structure(source_dir: str, target_dir: str) -> None:
    '''
    :param source_dir: Path of the directory that be copied from
    :type source_dir: str
    :param target_dir: Path of the directory that be copied to
    :type target_dir: str
    :return: None
    Create the same directory structure in ``target_dir`` with that of ``source_dir``.
    '''
    for sub_dir_name in os.listdir(source_dir):
        source_sub_dir = os.path.join(source_dir, sub_dir_name)
        if os.path.isdir(source_sub_dir):
            target_sub_dir = os.path.join(target_dir, sub_dir_name)
            os.mkdir(target_sub_dir)
            print(f'Mkdir [{target_sub_dir}].')
            create_same_directory_structure(source_sub_dir, target_sub_dir)


def integrate_events_file_to_frames_file_by_fixed_frames_number(events_np_file: str, output_dir: str, split_by: str,
                                                                frames_num: int, H: int, W: int,
                                                                print_save: bool = False) -> None:
    '''
    :param events_np_file: path of the events np file
    :type events_np_file: str
    :param output_dir: output directory for saving the frames
    :type output_dir: str
    :param split_by: 'time' or 'number'
    :type split_by: str
    :param frames_num: the number of frames
    :type frames_num: int
    :param H: the height of frame
    :type H: int
    :param W: the weight of frame
    :type W: int
    :param print_save: If ``True``, this function will print saved files' paths.
    :type print_save: bool
    :return: None
    Integrate a events file to frames by fixed frames number and save it. See ``cal_fixed_frames_number_segment_index`` and ``integrate_events_segment_to_frame`` for more details.
    '''
    fname = os.path.join(output_dir, os.path.basename(events_np_file))
    np.savez(fname, frames=integrate_events_by_fixed_frames_number(np.load(events_np_file), split_by, frames_num, H, W))
    if print_save:
        print(f'Frames [{fname}] saved.')


def integrate_events_by_fixed_frames_number(events: dict, split_by: str, frames_num: int, H: int, W: int) -> np.ndarray:
    '''
    :param events: a dict whose keys are ['t', 'x', 'y', 'p'] and values are ``numpy.ndarray``
    :type events: Dict
    :param split_by: 'time' or 'number'
    :type split_by: str
    :param frames_num: the number of frames
    :type frames_num: int
    :param H: the height of frame
    :type H: int
    :param W: the weight of frame
    :type W: int
    :return: frames
    :rtype: np.ndarray
    Integrate events to frames by fixed frames number. See ``cal_fixed_frames_number_segment_index`` and ``integrate_events_segment_to_frame`` for more details.
    '''
    j_l, j_r = cal_fixed_frames_number_segment_index(events['t'], split_by, frames_num)
    frames = np.zeros([frames_num, 2, H, W])
    for i in range(frames_num):
        frames[i] = integrate_events_segment_to_frame(events, H, W, j_l[i], j_r[i])
    return frames


def cal_fixed_frames_number_segment_index(events_t: np.ndarray, split_by: str, frames_num: int) -> tuple:
    '''
    :param events_t: events' t
    :type events_t: numpy.ndarray
    :param split_by: 'time' or 'number'
    :type split_by: str
    :param frames_num: the number of frames
    :type frames_num: int
    :return: a tuple ``(j_l, j_r)``
    :rtype: tuple
    Denote ``frames_num`` as :math:`M`, if ``split_by`` is ``'time'``, then
    .. math::
        \\Delta T & = [\\frac{t_{N-1} - t_{0}}{M}] \\\\
        j_{l} & = \\mathop{\\arg\\min}\\limits_{k} \\{t_{k} | t_{k} \\geq t_{0} + \\Delta T \\cdot j\\} \\\\
        j_{r} & = \\begin{cases} \\mathop{\\arg\\max}\\limits_{k} \\{t_{k} | t_{k} < t_{0} + \\Delta T \\cdot (j + 1)\\} + 1, & j <  M - 1 \\cr N, & j = M - 1 \\end{cases}
    If ``split_by`` is ``'number'``, then
    .. math::
        j_{l} & = [\\frac{N}{M}] \\cdot j \\\\
        j_{r} & = \\begin{cases} [\\frac{N}{M}] \\cdot (j + 1), & j <  M - 1 \\cr N, & j = M - 1 \\end{cases}
    '''
    j_l = np.zeros(shape=[frames_num], dtype=int)
    j_r = np.zeros(shape=[frames_num], dtype=int)
    N = events_t.size

    if split_by == 'number':
        di = N // frames_num
        for i in range(frames_num):
            j_l[i] = i * di
            j_r[i] = j_l[i] + di
        j_r[-1] = N

    elif split_by == 'time':
        dt = (events_t[-1] - events_t[0]) // frames_num
        idx = np.arange(N)
        for i in range(frames_num):
            t_l = dt * i + events_t[0]
            t_r = t_l + dt
            mask = np.logical_and(events_t >= t_l, events_t < t_r)
            idx_masked = idx[mask]
            j_l[i] = idx_masked[0]
            j_r[i] = idx_masked[-1] + 1

        j_r[-1] = N
    else:
        raise NotImplementedError

    return j_l, j_r


def integrate_events_segment_to_frame(events: dict, H: int, W: int, j_l: int = 0, j_r: int = -1) -> np.ndarray:
    '''
    :param events: a dict whose keys are ['t', 'x', 'y', 'p'] and values are ``numpy.ndarray``
    :type events: Dict
    :param H: height of the frame
    :type H: int
    :param W: weight of the frame
    :type W: int
    :param j_l: the start index of the integral interval, which is included
    :type j_l: int
    :param j_r: the right index of the integral interval, which is not included
    :type j_r:
    :return: frames
    :rtype: np.ndarray
    Denote a two channels frame as :math:`F` and a pixel at :math:`(p, x, y)` as :math:`F(p, x, y)`, the pixel value is integrated from the events data whose indices are in :math:`[j_{l}, j_{r})`:
.. math::
    F(p, x, y) &= \sum_{i = j_{l}}^{j_{r} - 1} \mathcal{I}_{p, x, y}(p_{i}, x_{i}, y_{i})
where :math:`\lfloor \cdot \rfloor` is the floor operation, :math:`\mathcal{I}_{p, x, y}(p_{i}, x_{i}, y_{i})` is an indicator function and it equals 1 only when :math:`(p, x, y) = (p_{i}, x_{i}, y_{i})`.
    '''
    # 累计脉冲需要用bitcount而不能直接相加，原因可参考下面的示例代码，以及
    # https://stackoverflow.com/questions/15973827/handling-of-duplicate-indices-in-numpy-assignments
    # We must use ``bincount`` rather than simply ``+``. See the following reference:
    # https://stackoverflow.com/questions/15973827/handling-of-duplicate-indices-in-numpy-assignments

    # Here is an example:

    # height = 3
    # width = 3
    # frames = np.zeros(shape=[2, height, width])
    # events = {
    #     'x': np.asarray([1, 2, 1, 1]),
    #     'y': np.asarray([1, 1, 1, 2]),
    #     'p': np.asarray([0, 1, 0, 1])
    # }
    #
    # frames[0, events['y'], events['x']] += (1 - events['p'])
    # frames[1, events['y'], events['x']] += events['p']
    # print('wrong accumulation\n', frames)
    #
    # frames = np.zeros(shape=[2, height, width])
    # for i in range(events['p'].__len__()):
    #     frames[events['p'][i], events['y'][i], events['x'][i]] += 1
    # print('correct accumulation\n', frames)
    #
    # frames = np.zeros(shape=[2, height, width])
    # frames = frames.reshape(2, -1)
    #
    # mask = [events['p'] == 0]
    # mask.append(np.logical_not(mask[0]))
    # for i in range(2):
    #     position = events['y'][mask[i]] * width + events['x'][mask[i]]
    #     events_number_per_pos = np.bincount(position)
    #     idx = np.arange(events_number_per_pos.size)
    #     frames[i][idx] += events_number_per_pos
    # frames = frames.reshape(2, height, width)
    # print('correct accumulation by bincount\n', frames)

    frame = np.zeros(shape=[2, H * W])
    x = events['x'][j_l: j_r].astype(int)  # avoid overflow
    y = events['y'][j_l: j_r].astype(int)
    p = events['p'][j_l: j_r]
    mask = []
    mask.append(p == 0)
    mask.append(np.logical_not(mask[0]))
    for c in range(2):
        position = y[mask[c]] * W + x[mask[c]]
        events_number_per_pos = np.bincount(position)
        frame[c][np.arange(events_number_per_pos.size)] += events_number_per_pos
    return frame.reshape((2, H, W))


def integrate_events_file_to_frames_file_by_fixed_duration(events_np_file: str, output_dir: str, duration: int, H: int,
                                                           W: int, print_save: bool = False) -> None:
    '''
    :param events_np_file: path of the events np file
    :type events_np_file: str
    :param output_dir: output directory for saving the frames
    :type output_dir: str
    :param duration: the time duration of each frame
    :type duration: int
    :param H: the height of frame
    :type H: int
    :param W: the weight of frame
    :type W: int
    :param print_save: If ``True``, this function will print saved files' paths.
    :type print_save: bool
    :return: None
    Integrate events to frames by fixed time duration of each frame.
    '''
    frames = integrate_events_by_fixed_duration(np.load(events_np_file), duration, H, W)
    fname, _ = os.path.splitext(os.path.basename(events_np_file))
    fname = os.path.join(output_dir, f'{fname}_{frames.shape[0]}.npz')
    np.savez(fname, frames=frames)
    if print_save:
        print(f'Frames [{fname}] saved.')
    return frames.shape[0]


def integrate_events_by_fixed_duration(events: dict, duration: int, H: int, W: int) -> np.ndarray:
    '''
    :param events: a dict whose keys are ['t', 'x', 'y', 'p'] and values are ``numpy.ndarray``
    :type events: Dict
    :param duration: the time duration of each frame
    :type duration: int
    :param H: the height of frame
    :type H: int
    :param W: the weight of frame
    :type W: int
    :return: frames
    :rtype: np.ndarray
    Integrate events to frames by fixed time duration of each frame.
    '''
    t = events['t']
    N = t.size

    frames = []
    left = 0
    right = 0
    while True:
        t_l = t[left]
        while True:
            if right == N or t[right] - t_l > duration:
                break
            else:
                right += 1
        # integrate from index [left, right)
        frames.append(np.expand_dims(integrate_events_segment_to_frame(events, H, W, left, right), 0))

        left = right

        if right == N:
            return np.concatenate(frames)

# if __name__ == "__main__":
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import os
#     import wave
#
#     # 读入音频。
#     path = r"F:\GitCode\Python\datasets\TidigitsWAV1\train\zero"
#     name = '1.wav'
#     # 我音频的路径为E:\SpeechWarehouse\zmkm\zmkm0.wav
#     filename = os.path.join(path, name)
#
#     # 打开语音文件。
#     f = wave.open(filename, 'rb')
#     # 得到语音参数
#     params = f.getparams()
#     nchannels, sampwidth, framerate, nframes = params[:4]
#     # ---------------------------------------------------------------#
#     # 将字符串格式的数据转成int型
#     print("reading wav file......")
#     strData = f.readframes(nframes)
#     waveData = np.fromstring(strData, dtype=np.short)
#     # 归一化
#     waveData = waveData * 1.0 / max(abs(waveData))
#     # 将音频信号规整乘每行一路通道信号的格式，即该矩阵一行为一个通道的采样点，共nchannels行
#     waveData = np.reshape(waveData, [nframes, nchannels]).T  # .T 表示转置
#     f.close()  # 关闭文件
#     print("file is closed!")
#     # ----------------------------------------------------------------#
#     '''绘制语音波形'''
#     print("plotting signal wave...")
#     time = np.arange(0, nframes) * (1.0 / framerate)  # 计算时间
#     time = np.reshape(time, [nframes, 1]).T
#     plt.plot(time[0, :nframes], waveData[0, :nframes], c="b")
#     plt.axis('off')  # no axis
#     plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
#     # plt.xlabel("time")
#     # plt.ylabel("amplitude")
#     # plt.title("Original wave")
#     plt.show()
#     print('end')


#     sroot = r'F:\GitCode\Python\datasets\TidigitsWAV'
#     troot = r'F:\GitCode\Python\datasets\TidigitsWAV1'
# classes = {
#     "zero": 0,
#     "one": 1,
#     "two": 2,
#     "three": 3,
#     "four": 4,
#     "five": 5,
#     "six": 6,
#     "seven": 7,
#     "eight": 8,
#     "nine": 9,
#     "oh": 10
# }
# save_mfcc_feature(root=troot, npz_name='mfcc_test.npz', class_labels=classes)
#     dataset_split(sroot, troot, 0.7, True)
#     print('end')


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
# filenameTr = 'train.npz'
# root = r'F:\GitCode\Python\datasets\DigitsVoices'
# filenameTr = 'train_kernels_num50significance_level10.npz'
# filerootTr = os.path.join(root, filenameTr)
# dataTr = np.load(filerootTr, allow_pickle=True)
# trian_audios0 = dataTr['train_ids'][0]
# trian_audios1 = dataTr['train_ids'][1]
# trian_audios2 = dataTr['train_ids'][2]
# trian_audios3 = dataTr['train_ids'][3]
# train_ids = dataTr['train_ids']
# maxvalue = get_Max(dataTr['train_audios'])
# filenameTe = 'test_kernels_num50significance_level10.npz'
# filerootTe = os.path.join(root, filenameTe)
# dataTe = np.load(filerootTe, allow_pickle=True)
# test_audios0 = dataTe['test_audios'][0]
# test_audios1 = dataTe['test_audios'][1]
# test_audios2 = dataTe['test_audios'][2]
# test_audios3 = dataTe['test_audios'][3]
# test_audios = dataTe['test_audios']
# test_ids = dataTe['test_ids']
# maxvalue = get_Max(dataTe['test_audios'])
# print(maxvalue)
# print('')
