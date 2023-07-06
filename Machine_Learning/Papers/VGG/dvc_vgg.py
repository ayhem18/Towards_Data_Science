"""
This script contains all the functionalities needed to train the dog VS cats classifier based on my_vgg architecture
"""
from torch import nn
from my_vgg import my_vgg
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
from pytorch_modular.pytorch_utilities import get_default_device, input_shape_from_dataloader
from pytorch_modular.engine_classification import train_per_epoch, test_per_epoch, create_summary_writer
from tqdm import tqdm


class DVG_classifier(my_vgg):

    num_classes = 2

    def __init__(self, input_shape: tuple[int, int, int], *args, **kwargs):
        super().__init__(input_shape, num_classes=self.num_classes, *args, **kwargs)

    # the forward function is the same as the base model
    def forward(self, x: torch.tensor):
        return super().forward(x)


def train_model(model: DVG_classifier,
                train_dataloader: DataLoader,
                test_dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                epochs: int = 5,
                writer: SummaryWriter = None,  # used to save the results and track them by TensorBoard
                device: str = None,
                print_progress=False,
                ):
    # set the device
    if device is None:
        device = get_default_device()

    # 2. Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    # set the model to the current device
    model.to(device)
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_per_epoch(model=model,
                                                dataloader=train_dataloader,
                                                loss_fn=nn.BCEWithLogitsLoss,
                                                optimizer=optimizer,
                                                output_layer=nn.Sigmoid,
                                                device=device)

        test_loss, test_acc = test_per_epoch(model=model,
                                             dataloader=test_dataloader,
                                             loss_fn=nn.BCEWithLogitsLoss,
                                             output_layer=nn.Sigmoid,
                                             device=device)

        if print_progress:
            # 4. Print out what's happening
            print(
                f"Epoch: {epoch + 1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if writer is not None:
            # track loss results
            writer.add_scalars(main_tag='Loss',
                               tag_scalar_dict={"train_loss": train_loss, 'test_loss': test_loss},
                               global_step=epoch)

            writer.add_scalars(main_tag='Accuracy',
                               tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc},
                               global_step=epoch)

            # to get the shape of the model's input, we can use the train_dataloader
            writer.add_graph(model=model,
                             input_to_model=torch.randn(size=input_shape_from_dataloader(train_dataloader)).to(device))

            writer.close()

            # 6. Return the filled results at the end of the epochs
    return results
