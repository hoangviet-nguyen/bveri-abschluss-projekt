from torch import nn
from torch.nn import functional as F
import torch

'''
===========================================================================================================================================
RSB Encoder Block
===========================================================================================================================================
''' 
class RSB(nn.Module):
    """A basic encoder block that performs convolution, normalization, and activation."""
    def __init__(
        self,
        in_channels: int,
        out_channels: list[int],
        kernel_size=(3, 3),
        padding=1,
        skip=True,
    ):
        super().__init__()
        self.skip = skip

        f1, f2, f3, f4 = out_channels

        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=(1, 1), stride=1)
        self.bn1 = nn.BatchNorm2d(f1)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(f1, f2, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(f2)

        self.conv3 = nn.Conv2d(f2, f3, kernel_size=(1, 1))
        self.bn3 = nn.BatchNorm2d(f3)

        self.conv4 = nn.Conv2d(in_channels, f4, kernel_size=(1, 1))
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

class RSBEncoder(nn.Module):
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

    def __init__(self):
        super().__init__()

        self.initial_conv = nn.Conv2d(3, 64, kernel_size=(5, 5), stride=1, padding=2)
        self.initial_bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.block2 = nn.ModuleList(
            [
                RSB(64,  [64, 64, 128, 128], skip=False),
                RSB(128, [64, 64, 128, 128], skip=True),
                RSB(128, [64, 64, 128, 128], skip=True),
            ]
        )

        self.block3 = nn.ModuleList(
            [
                RSB(128, [128, 128, 256, 256], skip=False),
                RSB(256, [128, 128, 256, 256], skip=True),
                RSB(256, [128, 128, 256, 256], skip=True),
                RSB(256, [128, 128, 256, 256], skip=True),
            ]
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.relu(x)

        encoded1 = x
        x = self.pool1(x)

        for encoder_block in self.block2:
            x = encoder_block(x)

        x = self.pool1(x)
        encoded2 = x

        for encoder_block in self.block3:
            x = encoder_block(x)

        x= self.pool1(x)
        encoded3 = x

        return [encoded1, encoded2, encoded3]

'''
===========================================================================================================================================
RSB Decoder Block
===========================================================================================================================================
'''   
class RSBDecoder(nn.Module):
    """Decodes a low-dimensional representation back to an image.

    Args:
        num_channels_in (int): Number of input channels (output of encoder)
        num_channels (list[int]): Number of channels for each block, reversed from the encoder configuration.
    Input:
        feature map of shape (N, C, H, W)
    Output:
        image batch of shape (N, C_out, H_out, W_out), where C_out is typically the original input channels.
    """

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        self.deconv1 = nn.ConvTranspose2d(
            256, 256, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        self.deconv2 = nn.ConvTranspose2d(
            512, 256, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        self.deconv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        # Convolutions for feature refinement
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

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

        decoded1 = self.deconv1(encoded3)
        decoded1 = self.bn1(decoded1)
        decoded1_final = self.concat_skip(encoded2, decoded1, self.concat_conv1, self.concat_bn1)

        decoded2_a = self.deconv2(decoded1_final)
        decoded2_a = self.bn2(decoded2_a)

        decoded2_b = self.deconv3(decoded2_a)
        decoded2_b = self.bn3(decoded2_b)
        decoded2_b = self.concat_skip(encoded1, decoded2_b, self.concat_conv2, self.concat_bn2)

        decoded3 = self.conv1(decoded2_b)
        decoded3 = self.bn4(decoded3)
        decoded3 = self.conv2(decoded3)
        decoded3 = self.bn5(decoded3)

        out = self.final_conv(decoded3)

        return out

'''
===========================================================================================================================================
Suim Net Model
===========================================================================================================================================
''' 

class SuimNet(nn.Module):
    """Encoder-Decoder architecture for image-to-image tasks.
    Args:
        encoder (nn.Module): The encoder network that reduces the
            spatial dimensions and extracts features from the input.
        decoder (nn.Module): The decoder network that upsamples the
            features and reconstructs the output image.
        num_in_channels (int): Number of input channels (e.g., 3 for RGB images).
        num_in_encoder_channels (int): Number of channels for the input
            convolution layer, used to match the encoder's initial channel size.
        num_out_decoder_channels (int): Number of channels for the final
            output from the decoder before the output layer.
        num_output_channels (int): Number of output channels (e.g., 1 for grayscale or
            3 for RGB).

    Input:
        x (torch.Tensor): Image batch of shape (N, num_in_channels, H, W).

    Output:
        torch.Tensor: Processed image batch of shape (N, num_output_channels, H, W).
    """

    def __init__(self, encoder:RSBEncoder, decoder:RSBDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded_inputs = self.encoder(x)
        out = self.decoder(encoded_inputs)
        return out