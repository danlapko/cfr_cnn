import numpy as np
import torch
from skimage import transform


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, dtype="fp32"):
        self.npdtype = np.float32
        self.torchdtype = torch.float32
        if dtype == "fp16":
            self.npdtype = np.float16
            self.torchdtype = torch.float16

    def __call__(self, sample):
        video, still = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        still = still.transpose((2, 0, 1))
        video = np.array([frame.transpose((2, 0, 1)) for frame in video], dtype=self.npdtype)

        video = torch.from_numpy(video).type(self.torchdtype)
        still = torch.from_numpy(still).type(self.torchdtype)
        return video, still


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Should be applied to numpy array (not torch tensor)
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        video, still = sample

        h, w = still.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        still = transform.resize(still, (new_h, new_w))
        for i in range(len(video)):
            video[i] = transform.resize(video[i], (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively

        return video, still


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        video, still = sample

        h, w = still.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        for i in range(len(video)):
            video[i] = video[i][top: top + new_h,
                       left: left + new_w]
            video[i] = transform.resize(video[i], (h, w))

        return video, still


class CenterCrop(object):
    """

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        video, still = sample

        h, w = video[0].shape[:2]
        new_h, new_w = self.output_size

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        for i in range(len(video)):
            video[i] = video[i][top: top + new_h,
                       left: left + new_w]
        new_h1 = 334
        new_w1 = 260
        h1, w1 = still.shape[:2]
        top1 = (h1 - new_h1) // 2
        left1 = (w1 - new_w1) // 2
        still = still[top1: top1 + new_h1,
                left1: left1 + new_w1]
        still = transform.resize(still, (new_h, new_w))

        return video, still
