import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from torch import nn
from torch.nn.modules.distance import PairwiseDistance
from torch.utils.tensorboard import SummaryWriter

from models.cfr import Classifier, Decoder, Encoder
from utils import roc_curve, rank1, weighted_mse_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

l2_dist = PairwiseDistance(2)


class Trainer:
    def __init__(self, color_channels,
                 encoder_path, decoder_path, clf_path):

        self.clf_path = clf_path
        self.decoder_path = decoder_path
        self.encoder_path = encoder_path
        self.color_channels = color_channels

        self.enc = Encoder().to(device)
        self.dec = Decoder().to(device)
        self.clf = Classifier().to(device)

        if os.path.isfile(encoder_path):
            self.enc.load_state_dict(torch.load(encoder_path))
            print("enc loaded")
        if os.path.isfile(decoder_path):
            self.dec.load_state_dict(torch.load(decoder_path))
            print("dec loaded")
        if os.path.isfile(clf_path):
            self.clf.load_state_dict(torch.load(clf_path))
            print("clf loaded")

        self.enc_optimizer = torch.optim.Adam(self.enc.parameters(), betas=(0.9, 0.999), lr=1e-4)
        self.dec_optimizer = torch.optim.Adam(self.dec.parameters(), betas=(0.9, 0.999), lr=1e-4)
        self.clf_optimizer = torch.optim.Adam(self.clf.parameters(), betas=(0.9, 0.999), lr=1e-4)

        self.mse_loss_function = weighted_mse_loss
        self.clf_loss_function = nn.BCELoss().to(device)

        self.writer = SummaryWriter(log_dir="logs")

    def test(self, test_loader, epoch):
        self.enc.eval()
        self.dec.eval()
        self.clf.eval()
        X, y = next(iter(test_loader))
        B, D, C, W, H = X.shape

        n = len(y)

        with torch.no_grad():
            loss_mse = self.get_mse_loss(X.to(device), y.to(device))
            loss_clf = self.get_clf_loss(X.to(device), y.to(device))

        self.writer.add_scalar("test loss_mse", loss_mse.item(), global_step=epoch)
        self.writer.add_scalar("test loss_clf", loss_clf.item(), global_step=epoch)

        embeddings_anc = self.enc(X.view(B * D, C, W, H).to(device))
        embeddings_pos = self.enc(y.to(device))

        trshs, fprs, tprs = roc_curve(embeddings_anc.detach(), embeddings_pos.detach(), self.clf)
        rnk1 = rank1(embeddings_anc.detach(), embeddings_pos.detach(), self.clf)
        plt.step(fprs, tprs)
        plt.yticks(np.arange(0, 1, 0.05))
        plt.xticks(np.arange(min(fprs), max(fprs), 10))
        plt.xscale('log')
        plt.title(f"ROC auc={auc(fprs, tprs)} rnk1={rnk1}")

        self.writer.add_figure("ROC test", plt.gcf(), global_step=epoch)
        self.writer.add_scalar("auc", auc(fprs, tprs), global_step=epoch)
        self.writer.add_scalar("rank1", rnk1, global_step=epoch)

        print(f"\n###### {epoch} TEST loss_mse {loss_mse.item():.5} loss_clf {loss_clf.item():.5} "
              f"auc={auc(fprs, tprs)} rank1 = {rnk1}  #######")

        x = X.view(B * D, C, W, H)[0:1]
        emb = self.enc(x.to(device))
        front = self.dec(emb).detach().cpu()
        self.writer.add_image("cfr", np.hstack((x[0], y[0], front[0])), global_step=epoch)
        torch.cuda.empty_cache()

    def get_mse_loss(self, X, y):
        B, D, C, W, H = X.shape
        embeddings_anc = self.enc(X.view(B * D, C, W, H))
        frontals = self.dec(embeddings_anc)
        loss_mse = self.mse_loss_function(frontals, y)
        return loss_mse

    def get_clf_loss(self, X, y):
        B, D, C, W, H = X.shape
        embeddings_anc = self.enc(X.view(B * D, C, W, H)).detach()

        embeddings_pos = self.enc(y).detach()

        clf_in = torch.cat((embeddings_anc, embeddings_pos), dim=-1)
        decisions = self.clf(clf_in)
        loss_clf = self.clf_loss_function(decisions, torch.full_like(decisions, 0.0))
        for shift in range(1, 2):
            embeddings_neg = torch.roll(embeddings_pos, shift, 0)

            clf_in = torch.cat((embeddings_anc, embeddings_neg), dim=-1)
            decisions = self.clf(clf_in)
            loss_clf += self.clf_loss_function(decisions, torch.full_like(decisions, 1.0))

        return loss_clf

    def train_aec(self, aec_loader, test_loader, batch_size, epochs):

        print("\nSTART AEC TRAINING\n")

        for epoch in range(epochs):
            self.test(test_loader, epoch)

            self.enc.train()
            self.dec.train()
            self.clf.train()
            # train by batches
            for idx, (btch_X, btch_y) in enumerate(aec_loader):
                btch_X = btch_X.to(device)
                btch_y = btch_y.to(device)

                self.enc.zero_grad()
                self.dec.zero_grad()

                loss_mse = self.get_mse_loss(btch_X, btch_y)
                loss_mse.backward()

                self.enc_optimizer.step()
                self.dec_optimizer.step()

                print(f"btch {idx * batch_size} loss_mse={loss_mse.item():.4} ")

                global_step = epoch * len(aec_loader.dataset) / batch_size + idx
                self.writer.add_scalar("train loss_mse", loss_mse.item(), global_step)

            torch.save(self.enc.state_dict(), self.encoder_path)
            torch.save(self.dec.state_dict(), self.decoder_path)

    def train_clf(self, clf_loader, test_loader, batch_size, epochs):
        print("\nSTART CLF TRAINING\n")

        for epoch in range(3003,epochs):
            self.test(test_loader, epoch)

            self.enc.train()
            self.dec.train()
            self.clf.train()
            # train by batches
            for idx, (btch_X, btch_y) in enumerate(clf_loader):
                btch_X = btch_X.to(device)
                btch_y = btch_y.to(device)

                self.clf.zero_grad()

                loss_clf = self.get_clf_loss(btch_X, btch_y)
                loss_clf.backward()

                self.clf_optimizer.step()

                print(f"btch {idx * batch_size} loss_clf={loss_clf.item():.4}")

                global_step = epoch * len(clf_loader.dataset) / batch_size + idx
                self.writer.add_scalar("train loss_clf", loss_clf.item(), global_step)

            torch.save(self.clf.state_dict(), self.clf_path)
