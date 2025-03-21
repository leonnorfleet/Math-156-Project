import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64  # Features for intermediate layers
        layers = []

        # First convolutional layer
        layers.append(nn.Conv2d(
            in_channels=channels, out_channels=features, kernel_size=kernel_size,
            padding=padding, bias=False
        ))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(
                in_channels=features, out_channels=features, kernel_size=kernel_size,
                padding=padding, bias=False
            ))
            layers.append(nn.BatchNorm2d(features))  # Use the same features for batchnorm
            layers.append(nn.ReLU(inplace=True))

        # Final convolutional layer (output layer)
        layers.append(nn.Conv2d(
            in_channels=features, out_channels=channels, kernel_size=kernel_size,
            padding=padding, bias=False
        ))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out
    
# Test that model works for (512x512) sized images
# if __name__ == "__main__":
#   x = torch.randn((3, 3, 512, 512))
#   model = DnCNN(channels=3, num_of_layers=12)
#   preds = model(x)
#   print(preds.shape)  # Should match input shape