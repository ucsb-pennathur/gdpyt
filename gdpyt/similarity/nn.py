import torch
import torch.nn as nn
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
            assert isinstance(batch_norm, tuple)

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

            if self.batch_norm is not None:
                if i in self.batch_norm:
                    self.blocks.append(nn.BatchNorm2d(out_channels))

        # Linear layers
        for i in range(self.n_linear_layers):
            # One quarter the input nodes, except the last layer (1 output)
            if i == self.n_linear_layers - 1:
                linear_block = LinearBlock(inp_shape, 1, activation=None)
            else:
                linear_block = LinearBlock(inp_shape, 0.25, activation='relu')
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

    def __init__(self, inp_shape, out_features, bias=True, activation=None):
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
        if activation is not None:
            if activation == 'tanh':
                layer_content.append(nn.Tanh())
            elif activation == 'relu':
                layer_content.append(nn.ReLU())
            elif activation == 'lrelu':
                layer_content.append(nn.LeakyReLU())
            else:
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
        nn.init.kaiming_normal_(module.weight.data, nonlinearity='relu')
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
              epochs=10):

    # Initialize model weights
    model.train()
    model.float()
    model.to(device)

    avg_epoch_loss_train = []
    avg_epoch_aux_loss_train = []
    std_epoch_loss_train = []

    aux_criterion = nn.CrossEntropyLoss()

    for e in range(epochs):
        start = time.time()
        logger.info("Epoch {}: Start".format(e))
        loss_batch = []
        aux_loss_batch = []

        model.train()  # Set model to training mode

        for i, batch in enumerate(dataloader):
            logger.info("Epoch {}, Batch {}".format(e, i))
            # Data in minibatch format N x C x H x H
            X = batch['input'].float().to(device)
            y = batch['target'].float().to(device)

            if 'aux_target' in batch.keys():
                y_aux = batch['aux_target'].to(device)
            else:
                y_aux = None

            prediction = model(X)  # [N, 1]
            if y_aux is None:
                loss = criterion(prediction, y)
                aux_loss = None
            else:
                loss = criterion(prediction.target, y)
                aux_loss = aux_criterion(prediction.aux_logits, y_aux)
                aux_loss_batch.append(aux_loss.item())

            loss_batch.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            if aux_loss is not None:
                aux_loss.backward()
            optimizer.step()

        avg_epoch_loss_train.append(np.array(loss_batch).sum() / (len(dataloader) * len(X)))
        std_epoch_loss_train.append(np.array(loss_batch).std())
        if len(aux_loss_batch) > 0:
            avg_epoch_aux_loss_train.append(np.array(aux_loss_batch).sum() / (len(dataloader) * len(X)))

        end = time.time() - start
        logger.info(
            "Epoch {}: Duration: {:.02f}s, Train Loss: {:.02e}".format(
                e, end, avg_epoch_loss_train[e]))

    return avg_epoch_loss_train, std_epoch_loss_train, model


class WeightedMSELoss(nn.MSELoss):

    def __init__(self, weight_z, **kwargs):
        super(WeightedMSELoss, self).__init__(**kwargs)
        self.weight_func = weight_z

    def forward(self, y, target):
        weight = self.weight_func(target)
        ret = weight * (y - target) ** 2
        ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)
        return ret
