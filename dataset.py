import numpy as np
import torch
import skimage
from skimage import transform
import matplotlib.pyplot as plt
import os
import copy


class Dataset(torch.utils.data.Dataset):
    """
    dataset of image files of the form 
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data_dir, data_type='float32', transform=None, sgm=25, ratio=0.9, size_data=(256, 256, 3), size_window=(5, 5)):
        self.data_dir = data_dir
        self.transform = transform
        self.data_type = data_type
        self.sgm = sgm

        self.ratio = ratio
        self.size_data = size_data
        self.size_window = size_window

        lst_data = os.listdir(data_dir)

        # lst_input = [f for f in lst_data if f.startswith('input')]
        # lst_label = [f for f in lst_data if f.startswith('label')]
        #
        # lst_input.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        # lst_label.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        #
        # self.lst_input = lst_input
        # self.lst_label = lst_label

        lst_data.sort(key=lambda f: (''.join(filter(str.isdigit, f))))
        # lst_data.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        self.lst_data = lst_data
        self.noise = self.sgm / 255.0 * np.random.randn(len(self.lst_data), self.size_data[0], self.size_data[1], self.size_data[2])

    def __getitem__(self, index):
        # label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        # input = np.load(os.path.join(self.data_dir, self.lst_input[index]))
        #
        # if label.dtype == np.uint8:
        #     label = label / 255.0
        # if input.dtype == np.uint8:
        #     input = input / 255.0
        #
        # if label.ndim == 2:
        #     label = np.expand_dims(label, axis=2)
        # if input.ndim == 2:
        #     input = np.expand_dims(input, axis=2)
        #
        # if self.ny != label.shape[0]:
        #     label = label.transpose((1, 0, 2))
        # if self.ny != input.shape[0]:
        #     input = input.transpose((1, 0, 2))
        #
        # data = {'input': input, 'label': label}

        data = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))

        if data.dtype == np.uint8:
            data = data / 255.0

        if data.ndim == 2:
            data = np.expand_dims(data, axis=2)

        if data.shape[0] > data.shape[1]:
            data = data.transpose((1, 0, 2))

        label = data + self.noise[index]
        input, mask = self.generate_mask(copy.deepcopy(label))

        data = {'label': label, 'input': input, 'mask': mask}

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.lst_data)

    def generate_mask(self, input):

        ratio = self.ratio
        size_window = self.size_window
        size_data = self.size_data
        num_sample = int(size_data[0] * size_data[1] * (1 - ratio))

        mask = np.ones(size_data)
        output = input

        for ich in range(size_data[2]):
            idy_msk = np.random.randint(0, size_data[0], num_sample)
            idx_msk = np.random.randint(0, size_data[1], num_sample)

            idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2, size_window[0] // 2 + size_window[0] % 2, num_sample)
            idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2, size_window[1] // 2 + size_window[1] % 2, num_sample)

            idy_msk_neigh = idy_msk + idy_neigh
            idx_msk_neigh = idx_msk + idx_neigh

            idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * size_data[0] - (idy_msk_neigh >= size_data[0]) * size_data[0]
            idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * size_data[1] - (idx_msk_neigh >= size_data[1]) * size_data[1]

            id_msk = (idy_msk, idx_msk, ich)
            id_msk_neigh = (idy_msk_neigh, idx_msk_neigh, ich)

            output[id_msk] = input[id_msk_neigh]
            mask[id_msk] = 0.0

        return output, mask

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = torch.from_numpy(value.transpose((2, 0, 1)))
        #
        # return data

        input, label, mask = data['input'], data['label'], data['mask']

        input = input.transpose((2, 0, 1)).astype(np.float32)
        label = label.transpose((2, 0, 1)).astype(np.float32)
        mask = mask.transpose((2, 0, 1)).astype(np.float32)
        return {'input': torch.from_numpy(input), 'label': torch.from_numpy(label), 'mask': torch.from_numpy(mask)}


class Normalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        input, label, mask = data['input'], data['label'], data['mask']

        input = (input - self.mean) / self.std
        label = (label - self.mean) / self.std

        data = {'input': input, 'label': label, 'mask': mask}
        return data


class RandomFlip(object):
    def __call__(self, data):
        # Random Left or Right Flip

        # for key, value in data:
        #     data[key] = 2 * (value / 255) - 1
        #
        # return data
        input, label, mask = data['input'], data['label'], data['mask']

        if np.random.rand() > 0.5:
            input = np.fliplr(input)
            label = np.fliplr(label)
            mask = np.fliplr(mask)

        if np.random.rand() > 0.5:
            input = np.flipud(input)
            label = np.flipud(label)
            mask = np.flipud(mask)

        return {'input': input, 'label': label, 'mask': mask}


class Rescale(object):
  """Rescale the image in a sample to a given size

  Args:
    output_size (tuple or int): Desired output size.
                                If tuple, output is matched to output_size.
                                If int, smaller of image edges is matched
                                to output_size keeping aspect ratio the same.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size

  def __call__(self, data):
    input, label = data['input'], data['label']

    h, w = input.shape[:2]

    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)

    input = transform.resize(input, (new_h, new_w))
    label = transform.resize(label, (new_h, new_w))

    return {'input': input, 'label': label}


class RandomCrop(object):
  """Crop randomly the image in a sample

  Args:
    output_size (tuple or int): Desired output size.
                                If int, square crop is made.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size)
    else:
      assert len(output_size) == 2
      self.output_size = output_size

  def __call__(self, data):
    input, label, mask = data['input'], data['label'], data['mask']

    h, w = input.shape[:2]
    new_h, new_w = self.output_size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    id_y = np.arange(top, top + new_h, 1)[:, np.newaxis].astype(np.int32)
    id_x = np.arange(left, left + new_w, 1).astype(np.int32)

    # input = input[top: top + new_h, left: left + new_w]
    # label = label[top: top + new_h, left: left + new_w]

    input = input[id_y, id_x]
    label = label[id_y, id_x]
    mask = mask[id_y, id_x]

    return {'input': input, 'label': label, 'mask': mask}


class UnifromSample(object):
  """Crop randomly the image in a sample

  Args:
    output_size (tuple or int): Desired output size.
                                If int, square crop is made.
  """

  def __init__(self, stride):
    assert isinstance(stride, (int, tuple))
    if isinstance(stride, int):
      self.stride = (stride, stride)
    else:
      assert len(stride) == 2
      self.stride = stride

  def __call__(self, data):
    input, label, mask = data['input'], data['label'], data['mask']

    h, w = input.shape[:2]
    stride_h, stride_w = self.stride
    new_h = h//stride_h
    new_w = w//stride_w

    top = np.random.randint(0, stride_h + (h - new_h * stride_h))
    left = np.random.randint(0, stride_w + (w - new_w * stride_w))

    id_h = np.arange(top, h, stride_h)[:, np.newaxis]
    id_w = np.arange(left, w, stride_w)

    input = input[id_h, id_w]
    label = label[id_h, id_w]
    mask = mask[id_h, id_w]

    return {'input': input, 'label': label, 'mask': mask}


class ZeroPad(object):
  """Rescale the image in a sample to a given size

  Args:
    output_size (tuple or int): Desired output size.
                                If tuple, output is matched to output_size.
                                If int, smaller of image edges is matched
                                to output_size keeping aspect ratio the same.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size

  def __call__(self, data):
    input, label, mask = data['input'], data['label'], data['mask']

    h, w = input.shape[:2]

    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)

    l = (new_w - w)//2
    r = (new_w - w) - l

    u = (new_h - h)//2
    b = (new_h - h) - u

    input = np.pad(input, pad_width=((u, b), (l, r), (0, 0)))
    label = np.pad(label, pad_width=((u, b), (l, r), (0, 0)))
    mask = np.pad(mask, pad_width=((u, b), (l, r), (0, 0)))

    return {'input': input, 'label': label, 'mask': mask}

class ToNumpy(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = value.transpose((2, 0, 1)).numpy()
        #
        # return data

        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

        # input, label = data['input'], data['label']
        # input = input.transpose((2, 0, 1))
        # label = label.transpose((2, 0, 1))
        # return {'input': input.detach().numpy(), 'label': label.detach().numpy()}

class Denormalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data = self.std * data + self.mean
        return data
