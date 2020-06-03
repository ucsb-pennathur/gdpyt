import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class GdpytNet(nn.Module):

    __name__ = 'GdpytNet'

    def __init__(self, inp_shape, n_conv_layers=3, n_linear_layers=1, kernel_size=3, n_filters_init=32,
                 max_pool_params=None, batch_norm=None):
        super(GdpytNet, self).__init__()

        assert len(inp_shape) == 3
        assert isinstance(n_conv_layers, int)
        assert isinstance(n_linear_layers, int)
        assert isinstance(kernel_size, int)
        assert isinstance(n_filters_init, int)
        if max_pool_params is not None:
            assert isinstance(max_pool_params, dict)
        if batch_norm is not None:
            assert isinstance(batch_norm, list)

        self.blocks = nn.ModuleList()
        self.n_conv_layers = n_conv_layers
        self.n_linear_layers = n_linear_layers
        self.kernel_size = kernel_size
        self.n_filters_init = n_filters_init
        self._max_pool_params = max_pool_params
        self.batch_norm = batch_norm
        self.input_shape = inp_shape

        self._create_layers()

    def _create_layers(self):
        inp_shape = self.input_shape

        # Convolutional layers
        for i in range(self.n_conv_layers):
            out_channels = self.n_filters_init * 2**i
            conv_block = ConvBlock(inp_shape, out_channels, kernel_size=self.kernel_size, stride=1, padding=0,
                                   max_pool_params=self._max_pool_params)
            inp_shape = conv_block.outp_shape
            if inp_shape[1] <= 0 or inp_shape[2] <= 0:
                raise ValueError("Too many convolutional or pooling layers for images with shape {}".format(self.input_shape))
            self.blocks.append(conv_block)

            if i in self.batch_norm:
                self.blocks.append(nn.BatchNorm2d(out_channels))

        # Linear layers
        for i in range(self.n_linear_layers):
            # One quarter the input nodes, except the last layer (1 output)
            if i == self.n_linear_layers - 1:
                linear_block = LinearBlock(inp_shape, 10)
            else:
                linear_block = LinearBlock(inp_shape, 0.25)
            inp_shape = linear_block.outp_shape
            self.blocks.append(linear_block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ConvBlock(nn.Module):
    """
    Convolution block consisting of a convolutional layer and and an optional max pooling layer
    Args:
        in_size:        Size of the input in the format C x H x W. Needs to be
                        of type torch.Size
        out_channels:   Number of channels of the output of the block
        activation:     Specifies activation function to be used (type: str). Possibilities are listed in module global
                        variable ACTIV
        padding:        Number of padding pixels
    Attributes:
        outp_dim:       torch.Tensor that specifies the shape of the output of the block.
                        (Format C x H x W) This attribute can be used to automatically infer
                        the shape that a subsequent fully-connected
                        linear layer needs to have
        block:
    """

    def __init__(self, inp_shape, out_channels, kernel_size=3, stride=1, padding=0, max_pool_params=None):
        super(ConvBlock, self).__init__()

        layer_content = []
        # In_size needs to be list, tuple or torch.Size to be indexable and return int
        # Conv layer
        layer_content.append(nn.Conv2d(inp_shape[0], out_channels, kernel_size=kernel_size,
                                       padding=int(padding), stride=stride))
        # ReLU activation
        layer_content.append(nn.ReLU())
        outp_shape = conv2d_out_shape(inp_shape, out_channels, kernel_size=kernel_size,
                                      padding=padding, stride=stride, dilation=1)

        # Max pool
        if max_pool_params is not None:
            layer_content.append(nn.MaxPool2d(**max_pool_params))
            # Output shape after max pool layer. Define as new input shape for next iteration
            outp_shape = conv2d_out_shape(outp_shape, out_channels, **max_pool_params)

        self.outp_shape = outp_shape
        self.layers = nn.Sequential(*layer_content)

    def forward(self, x):
        out = self.layers(x)
        return out

class LinearBlock(nn.Module):

    def __init__(self, inp_shape, out_features, bias=True):
        super(LinearBlock, self).__init__()

        if isinstance(inp_shape, int):
            in_features = inp_shape
        else:
            assert isinstance(inp_shape, list)
            in_features = 1
            for e in inp_shape:
                in_features *= e
            in_features = int(in_features)

        self.inp_shape = in_features
        if isinstance(out_features, float) and out_features < 1:
            out_features = int(out_features * in_features)
        self.outp_shape = out_features

        layer_content = []
        layer_content.append(nn.Linear(in_features, out_features, bias=bias))
        layer_content.append(nn.ReLU())

        self.layers = nn.Sequential(*layer_content)

    def forward(self, x):
        out = self.layers(x.view(-1, self.inp_shape))
        return out

def init_weights(module):
    """
        Initialize the weights of module.
        Xavier initialization for Conv and Linear layers
    """
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight.data)
    elif isinstance(module, nn.BatchNorm2d):
        module.reset_parameters()

def conv2d_out_shape(shape, out_channels, kernel_size=3, padding=0, stride=1, dilation=1):
    """
    Calculates the shape of the output after a convolutional layer, eg. Conv2d
    or a max pooling layer
    Args:
        shape       Shape of the input to the layer in format C x H x W. Can be
                    a tuple, list, torch.Tensor or torch.Size
        out_channels    Number of channels of the ouput
    Returns:
        out_shape   List with three elements [C_out, H_out, W_out]
    """
    if not isinstance(kernel_size, torch.Tensor):
        kernel_size = torch.Tensor((kernel_size, kernel_size))
    if not isinstance(stride, torch.Tensor):
        stride = torch.Tensor((stride, stride))

    # Handle different input types
    if isinstance(shape, torch.Size):
        chw = torch.Tensor([s for s in shape])
    elif isinstance(shape, torch.Tensor):
        chw = shape
    else:
        chw = torch.Tensor(shape)

    out_shape = chw
    out_shape[1:3] = torch.floor((chw[1:3] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    out_shape[0] = out_channels

    # return as list

    return [int(s.item()) for s in out_shape]

def train_net(model, device, optimizer, criterion, dataloader,
              epochs=10, lambda_=1e-3, reg_type=None):
    # Initialize model weights
    model.apply(init_weights)
    model.train()
    model.to(device)

    avg_epoch_loss_train = []

    for e in range(epochs):
        start = time.time()
        logger.info("Epoch {}: Start".format(e))
        loss_accum = 0
        correct_train = 0

        model.train()  # Set model to training mode

        for i, (batch_x, batch_y) in enumerate(dataloader):
            logger.info("Epoch {}, Batch {}".format(e, i))
            # Data in minibatch format N x C x H x H
            # X = batch['input'].to(device)
            # y = batch['target'].to(device)
            X = Variable(batch_x)
            y = Variable(batch_y)
            prediction = model(X)  # [N, 2, H, W]
            loss = criterion(prediction, y)

            if reg_type:
                assert reg_type in ['l2', 'l1']
                if reg_type == 'l2':
                    for p in model.parameters():
                        loss += lambda_ * p.pow(2).sum()

            loss_accum += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if reg_type == 'l1':
                with torch.no_grad():
                    for p in model.parameters():
                        p.sub_(p.sign() * p.abs().clamp(max=lambda_))

        avg_epoch_loss_train.append(loss_accum / (len(dataloader) * len(X)))

        end = time.time() - start
        logger.info(
            "Epoch {}: Duration: {:.02f}s, Train Loss: {:.02e}".format(
                e, end, avg_epoch_loss_train[e]))

    return avg_epoch_loss_train, model


class GdpytTensorDataset(Dataset):
    def __init__(self, transforms_=None, normalize=False):

        """
        dataloader class
        """
        self.normalize = normalize
        self._source = None
        transform = []
        if transforms_ is not None:
            raise NotImplementedError
            # for transf in transforms:
            #     transforms.append(transf)
        transform.append(ToTensor())
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.stack)

    def __getitem__(self, idx):
        target, image = self._source[idx]

        if self.normalize:
            image = (image - self.stack.stats['mean']) / self.stack.stats['std']

        image = np.nan_to_num(image[:, :, np.newaxis])
        target = np.array([target])

        sample = {'input': image, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def from_calib_stack(self, stack):
        self._source = stack

    def return_dataloader(self, batch_size=4, shuffle=True, num_workers=4):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def infer(self, model, idx):
        """
        Infere a sample in the dataset with a trained model
        """

        x = self.__getitem__(idx)['input']
        target = self.__getitem__(idx)['target']
        # Force mini-batch shape
        x.unsqueeze_(0)

        # Evaluation mode
        model.eval()
        y = model(x)

        logger.info("Predicted: {}, Target: {}".format(y.item(), target.item()))
        return y.item(), target.item()

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, target = sample['input'], sample['target']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'input': torch.from_numpy(image).float(),
                'target': torch.from_numpy(target).float()}
