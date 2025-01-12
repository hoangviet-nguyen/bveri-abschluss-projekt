from torch import nn
import torch

class RSB(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: list[int],
        kernel_size=(3, 3),
        stride=1,
        padding=1,
        skip=True,
    ):
        super().__init__()
        self.skip = skip

        f1, f2, f3, f4 = out_channels

        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=(1, 1), stride=stride)
        self.bn1 = nn.BatchNorm2d(f1)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(f1, f2, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(f2)

        self.conv3 = nn.Conv2d(f2, f3, kernel_size=(1, 1))
        self.bn3 = nn.BatchNorm2d(f3)

        self.conv4 = nn.Conv2d(in_channels, f4, kernel_size=(1, 1),stride=stride)
        self.bn4 = nn.BatchNorm2d(f4)

    def forward(self, input_tensor):
        x = input_tensor

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.skip:
            shortcut = input_tensor
        else:
            shortcut = self.conv4(input_tensor)
            shortcut = self.bn4(shortcut)
        x += shortcut
        x = self.relu(x)
        return x
    def __repr__(self):
        return f"EncoderDecoder model with encoder and decoder components."
    
class RSBEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.initial_conv = nn.Conv2d(3, 64, kernel_size=(5, 5), stride=1, padding=2)
        self.initial_bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.block2 = nn.ModuleList(
            [
                RSB(64,  [64, 64, 128, 128], skip=False, stride=2),
                RSB(128, [64, 64, 128, 128], skip=True),
                RSB(128, [64, 64, 128, 128], skip=True),

            ]
        )

        self.block3 = nn.ModuleList(
            [
                RSB(128, [128, 128, 256, 256], skip=False,stride=2),
                RSB(256, [128, 128, 256, 256], skip=True),
                RSB(256, [128, 128, 256, 256], skip=True),
                RSB(256, [128, 128, 256, 256], skip=True),
            ]
        )

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):

        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.relu(x)

        encoded1 = x
        x = self.pool1(x)

        for encoder_block in self.block2:
            x = encoder_block(x)

        encoded2 = x

        for encoder_block in self.block3:
            x = encoder_block(x)

        encoded3 = x

        return [encoded1, encoded2, encoded3]
    def __repr__(self):
        return f"EncoderDecoder model with encoder and decoder components."

class RSBDecoder(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        self.deconv1 = nn.Upsample(scale_factor=2)
        self.deconv2 = nn.Upsample(scale_factor=2)
        self.deconv3 = nn.Upsample(scale_factor=2)

        # Convolutions for feature refinement
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv2a = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2b = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # Final output convolution
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

        # Skip connection layers
        self.concat_conv1 = nn.Conv2d(128, 256, kernel_size=1)
        self.concat_bn1 = nn.BatchNorm2d(256)
        self.concat_conv2 = nn.Conv2d(64, 128, kernel_size=1)
        self.concat_bn2 = nn.BatchNorm2d(128)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(64)

    def concat_skip(self, layer_input, skip_input, conv, bn):
        u = conv(layer_input)
        u = bn(u)
        concatenated = torch.cat((u, skip_input), dim=1)
        return concatenated

    def forward(self, encoded_inputs):
        encoded1, encoded2, encoded3 = encoded_inputs

        decoded1 = self.conv1(encoded3)
        decoded1 = self.bn1(decoded1)
        decoded1 = self.deconv1(decoded1)
        decoded1_final = self.concat_skip(encoded2, decoded1, self.concat_conv1, self.concat_bn1)

        decoded2_a = self.conv2a(decoded1_final)
        decoded2_a = self.bn2(decoded2_a)
        decoded2_a = self.deconv2(decoded2_a)

        decoded2_b = self.conv2b(decoded2_a)
        decoded2_b = self.bn3(decoded2_b)
        decoded2_b = self.deconv3(decoded2_b)
        decoded2_b = self.concat_skip(encoded1, decoded2_b, self.concat_conv2, self.concat_bn2)

        decoded3 = self.conv3(decoded2_b)
        decoded3 = self.bn4(decoded3)
        decoded3 = self.conv4(decoded3)
        decoded3 = self.bn5(decoded3)

        out = self.final_conv(decoded3)

        return out
    
class SuimNet(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded_inputs = self.encoder(x)
        out = self.decoder(encoded_inputs)
        return out

