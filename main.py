import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from datasets.cox import COX
from datasets.transformers import ToTensor
from trainer import Trainer


def prepare_dataloaders(sequence_length=7, batch_size=32, use_rgb=False, cam="cam1", k_for_autoenc=0.2, k_for_clf=0.1,
                        k_for_test=0.7):
    dataset = COX(f"/home/danila/masters/datasets/{'rgb_scaled' if use_rgb else 'gray'}/video",
                  f"/home/danila/masters/datasets/{'rgb_cropped' if use_rgb else 'gray'}/still",
                  sequence_length,
                  cam,
                  # transform=Compose([Rescale((80, 64)), RandomCrop((75, 60)), ToTensor("fp32")])
                  # transform=Compose([CenterCrop((96, 80)), ToTensor("fp32")])
                  transform=Compose([ToTensor("fp32")])
                  )
    # dist_h, dist_w = 60, 48  # (160, 128) (60,48)

    aec_size = int(k_for_autoenc * len(dataset))
    clf_size = int(k_for_clf * len(dataset))
    test_size = int(k_for_test * len(dataset))

    torch.manual_seed(0)

    aec_dataset, clf_dataset, test_dataset = torch.utils.data.random_split(dataset, [aec_size, clf_size, test_size])
    torch.manual_seed(torch.initial_seed())

    aec_loader = DataLoader(aec_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)

    clf_loader = DataLoader(clf_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)

    test_loader = DataLoader(test_dataset,
                             batch_size=test_size,
                             shuffle=True,
                             num_workers=0)

    print(f"aec_size={aec_size} clf_size={clf_size} test_size={test_size}")
    return aec_loader, clf_loader, test_loader


def main_():
    # ===== Dataset params =======
    # dist_h, dist_w = 60, 48  # (160, 128) (60,48)
    k_for_autoenc = 0.9
    k_for_clf = 0.9
    k_for_test = 0.1
    seq_len = 1  # 21 30
    use_rgb = False

    # ===== Training params =======

    cam = "cam1"

    epochs_aec = 10000
    epochs_clf = 10000
    batch_size = 50

    aec_loader, clf_loader, test_loader = prepare_dataloaders(seq_len, batch_size, use_rgb, cam,
                                                              k_for_autoenc=k_for_autoenc,
                                                              k_for_clf=k_for_clf,
                                                              k_for_test=k_for_test)

    trainer = Trainer(color_channels=3,
                      encoder_path=f"pretrained/{'rgb' if use_rgb else 'gray'}/enc.mdl",
                      decoder_path=f"pretrained/{'rgb' if use_rgb else 'gray'}/dec.mdl",
                      clf_path=f"pretrained/{'rgb' if use_rgb else 'gray'}/clf.mdl")

    # trainer.train_aec(aec_loader, test_loader, batch_size, epochs_aec)
    # trainer.train_clf(clf_loader, test_loader, batch_size, epochs_clf)

    print("\n ######## FINAL TEST ########")
    for i in range(10):
        trainer.test(test_loader, epochs_clf)


if __name__ == "__main__":
    main_()
