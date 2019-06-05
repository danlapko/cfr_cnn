import torch
from torch import nn
import torch.nn.functional as F


class CONV_BN_RELU(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3):
        super(CONV_BN_RELU, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return (out)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = CONV_BN_RELU(3, 64, 3, 1, 1)
        self.conv2 = CONV_BN_RELU(64, 32, 3, 1, 1)
        self.conv3 = CONV_BN_RELU(32, 16, 3, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=(1, 0))

        self.fc1 = nn.Sequential(nn.Linear(768, 512),
                                 nn.LeakyReLU())

        self.fc2 = nn.Sequential(nn.Linear(512, 256),
                                 nn.LeakyReLU())

    def forward(self, x):
        B, C, W, H = x.shape
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool3(x)

        x = x.view((B, -1))

        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(256, 512),
                                 nn.LeakyReLU())

        self.fc2 = nn.Sequential(nn.Linear(512, 768),
                                 nn.LeakyReLU())

        self.conv1 = CONV_BN_RELU(16, 32, 3, 1, 1)
        self.conv2 = CONV_BN_RELU(32, 64, 3, 1, 1)
        self.conv3 = CONV_BN_RELU(64, 3, 3, 1, 1)

        self.uppool1 = nn.Upsample(mode='bilinear', size=(15, 12))
        self.uppool = nn.Upsample(mode='bilinear', scale_factor=2)

    def forward(self, x):
        B, E = x.shape

        x = self.fc1(x)
        x = self.fc2(x)

        x = x.view((B, -1, 8, 6))
        x = self.uppool1(x)
        x = self.conv1(x)
        x = self.uppool(x)
        x = self.conv2(x)
        x = self.uppool(x)
        x = self.conv3(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        emb = self.encoder(x)
        img = self.decoder(emb)
        return emb, img


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(512, 128),
                                 nn.LeakyReLU())
        self.fc2 = nn.Sequential(nn.Linear(128, 32),
                                 nn.LeakyReLU())

        self.fc3 = nn.Sequential(nn.Linear(32, 1),
                                 nn.Sigmoid())

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
