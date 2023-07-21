import os
from dvc_vgg import DVG_classifier, train_model
from torch.optim import Adam
import pytorch_modular.data_loaders as loaders
import torchvision.transforms as T
from pathlib import Path
import torch


if __name__ == '__main__':
    HOME = os.getcwd()
    print(HOME)
    # HOME = os.path.join(HOME), 'Machine_Learning', 'Papers', 'VGG')
    DVC_DIR = os.path.join(HOME, 'dogs-vs-cats')
    TRAIN_DIR = os.path.join(DVC_DIR, 'train')
    TEST_DIR = os.path.join(DVC_DIR, 'test')

    TEST_DIR_RENAMED = os.path.join(Path(TEST_DIR).parent, 'test_original')
    print(len(os.listdir(TEST_DIR_RENAMED)))

    # time to create the dataloader

    RESIZE = (256, 256)

    trans = T.Compose([T.Resize(size=(256, 256)),
                       # T.GaussianBlur(kernel_size=(5, 5), sigma=(0.2, 0.5)),
                       T.ToTensor(),
                       ])

    # since the problem is known to be binary, we create a collate function
    # that adds an extra dimension to avoid the shape mismatch problem

    # now we have a small test dataset, and we can proceed
    # let's limit the number of files extracted to a mere 100 for faster training
    train_loader, test_loader, indices_to_names = loaders.create_dataloaders(TRAIN_DIR,
                                                                             batch_size=20,
                                                                             train_transform=trans,
                                                                             test_dir=TEST_DIR,
                                                                             )

    classifier = DVG_classifier(input_shape=(256, 256, 3))
    optimizer = Adam(classifier.parameters(), lr=10 ** -3)
    train_model(classifier,
                train_loader,
                test_loader,
                optimizer=optimizer,
                epochs=3,
                print_progress=True
                )
