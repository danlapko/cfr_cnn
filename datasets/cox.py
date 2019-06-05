import os
import random

import torch
from skimage import io
from torch.utils.data import Dataset
import numpy as np

from datasets.transformers import ToTensor


class COX(Dataset):

    def __init__(self, video_dirpath, still_dirpath, min_video_length, cam="cam1", transform=None):
        self.video_dirpath = video_dirpath
        self.still_dirpath = still_dirpath
        self.min_video_length = min_video_length
        self.cam = cam

        self.still_names = sorted(os.listdir(still_dirpath))

        self.transform = transform

        self.video_cache = [None] * self.__len__()
        self.still_cache = [None] * self.__len__()

    def load_sample(self, idx):
        still_name = self.still_names[idx]

        still = io.imread(os.path.join(self.still_dirpath, still_name)) / 255
        if len(still.shape) == 2:
            still = np.stack((still,) * 3, axis=-1)

        still_num = still_name.split("_")[0]
        frames_dir = os.path.join(self.video_dirpath, self.cam, still_num)
        frames_names = sorted(os.listdir(frames_dir))

        video_length = len(frames_names)
        while video_length < 5 * self.min_video_length:
            # print(still_num, self.min_video_length, video_length)
            for i in range(video_length):
                frames_names.append(frames_names[video_length - i - 1])
            video_length = len(frames_names)

        video = []
        for frame_name in frames_names:
            frame = io.imread(os.path.join(frames_dir, frame_name)) / 255
            if len(frame.shape) == 2:
                frame = np.stack((frame,) * 3, axis=-1)
            video.append(frame)

        sample = video, still
        if self.transform:
            video, still = self.transform(sample)

        self.still_cache[idx] = still
        self.video_cache[idx] = video

    def __len__(self):
        return len(self.still_names)

    def __getitem__(self, idx):
        if self.still_cache[idx] is None:
            self.load_sample(idx)

        still = self.still_cache[idx]
        video = self.video_cache[idx]

        start_frame_idx = np.random.randint(0, len(video) - 5 * self.min_video_length + 1)
        video = video[start_frame_idx:start_frame_idx + 5 * self.min_video_length:5]

        flip = random.random() > 0.5
        if flip:
            video = video.flip((-1,))
            still = still.flip((-1,))
        sample = video, still

        return sample


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    dataset = COX("/home/danila/masters/datasets/face_48_60/video",
                  "/home/danila/masters/datasets/face_48_60/still",
                  21,
                  "cam1", transform=ToTensor("fp32"))

    for i in range(2):
        video, still = dataset[i]
        print(video.shape, still.shape, still.dtype)
        plt.imshow(video[0].transpose(0, 1).transpose(1, 2))
        plt.show()

        plt.imshow(still.transpose(0, 1).transpose(1, 2))
        plt.show()
