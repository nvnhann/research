import torch
import torch.nn as nn


# Define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()

        # Define the feature extraction part of the network
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Define the classification part of the network
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=dropout),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass the input tensor through each of the defined layers

        # Pass through the feature extractor
        x = self.features(x)

        # Flatten the tensor to feed into the classifier
        x = torch.flatten(x, 1)

        # Pass through the classifier
        x = self.classifier(x)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders
    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):
    model = MyModel(num_classes=23, dropout=0.3)

    # Get a batch from train dataloader
    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    # Get the output of the model
    out = model(images)

    # Verify if model's output is a tensor
    assert isinstance(out,
                      torch.Tensor), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    # Verify the shape of the model's output
    assert out.shape == torch.Size([2, 23]), f"Expected an output tensor of size (2, 23), got {out.shape}"


def test_model_feature_extractor():
    model = MyModel(num_classes=23, dropout=0.3)
    images = torch.randn(2, 3, 224, 224)

    # Extract features
    features = model.features(images)

    # Check the shape of the features
    assert features.shape == torch.Size(
        [2, 512, 14, 14]), f"Expected feature map shape to be (2, 512, 14, 14), got {features.shape}"


def test_model_classifier():
    model = MyModel(num_classes=23, dropout=0.3)
    features = torch.randn(2, 512, 14, 14)

    # Flatten Features for Classifier Input
    features = torch.flatten(features, 1)

    # Classifier output
    output = model.classifier(features)

    # Verify the shape of classifier's output
    assert output.shape == torch.Size([2, 23]), f"Expected classifier output shape to be (2, 23), got {output.shape}"
