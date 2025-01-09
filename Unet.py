import torch
from torch import nn
from torch.nn import functional as F

class EncoderBlock(nn.Module):
    """A basic encoder block that performs convolution, normalization, and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channel: int,
        kernel_size=(3, 3),
        padding=1,
    ):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.bn3 = nn.BatchNorm2d(out_channel)

        self.conv1 = nn.Conv2d(in_channels, out_channel, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        return x
    
class Encoder(nn.Module):
    """Encodes an image to a low-dimensional representation.

    Args:
        num_channels_in (int): Number of input channels (e.g., 3 for RGB images).
        num_channels (list[int]): Number of output channels for each block.
            Each block reduces spatial dimensionality by half.
    Input:
        image batch of shape (N, C, H, W)

    Output:
        image batch of shape (N, C2, H / S, W / S), where S is the global stride.
    """

    def __init__(self, start=4, num_blocks=8):
        super().__init__()
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(in_channels=start * 2**i, out_channel=start * 2 ** (i + 1))
                for i in range(0, num_blocks)
            ]
        )

    def forward(self, x):

        encoded_blocks = list()
        for block in self.encoder_blocks:
            x = block(x)
            encoded_blocks.append(x)

        return encoded_blocks


class DecoderBlock(nn.Module):
    """Decodes a low-dimensional representation back to an image.

    Args:
        num_channels_in (int): Number of input channels (output of encoder)
        num_channels (list[int]): Number of channels for each block, reversed from the encoder configuration.
    Input:
        feature map of shape (N, C, H, W)
    Output:
        image batch of shape (N, C_out, H_out, W_out), where C_out is typically the original input channels.
    """

    def __init__(
        self,
        in_channels,
        out_channel,
        kernel_size=(3, 3),
        padding=1,
        output_padding=1,
        skip=False,
    ):

        super().__init__()
        self.relu = nn.ReLU()
        self.skip = skip

        if skip:
            self.concat_conv = nn.Conv2d(
                in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1
            )
            self.concat_bn = nn.BatchNorm2d(in_channels)

        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.bn3 = nn.BatchNorm2d(out_channel)

        self.conv1 = nn.Conv2d(in_channels, out_channel, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.deconv = nn.ConvTranspose2d(
            out_channel,
            out_channel,
            kernel_size,
            padding=padding,
            output_padding=output_padding,
            stride=2,
        )

    def forward(self, x, skip_layer=None):
        if self.skip:
            x = self.concat_skip(x, skip_layer)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.deconv(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x

    def concat_skip(self, current_layer, skip_layer):
        current_layer = F.interpolate(
            current_layer,
            size=skip_layer.shape[2:],
            mode="nearest",
        )
        concatenated = torch.cat((current_layer, skip_layer), dim=1)
        concatenated = self.concat_conv(concatenated)
        concatenated = self.concat_bn(concatenated)
        return concatenated
    
class Decoder(nn.Module):
    def __init__(self, start=4, num_blocks=8):
        super().__init__()

        end = int(start * 2 ** num_blocks)
        self.init_block = DecoderBlock(in_channels=end, out_channel=end // 2)

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    in_channels=end // 2**i, out_channel=end // 2 ** (i + 1), skip=True
                )
                for i in range(1, num_blocks)
            ]
        )

    def forward(self, encoded_inputs):
        x = encoded_inputs[-1]
        skip_connections = encoded_inputs[:-1]
        skip_connections = skip_connections[::-1]

        x = self.init_block(x)

        for skip, block in zip(skip_connections, self.decoder_blocks):
            x = block(x, skip_layer=skip)

        return x
    

class UNet(nn.Module):

    def __init__(self, encoder, decoder, n_classes, start=5):
        super().__init__()

        self.inital_conv = nn.Conv2d(3, start, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(
            in_channels=start, out_channels=n_classes, kernel_size=1
        )
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.inital_conv(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_conv(x)
        return x