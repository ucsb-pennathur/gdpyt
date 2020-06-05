import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
from gdpyt.plotting import plot_tensor_dset
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
              epochs=10, lambda_=1e-3, reg_type=None):

    # Initialize model weights
    model.apply(init_weights)
    model.train()
    model.to(device)

    avg_epoch_loss_train = []
    std_epoch_loss_train = []

    for e in range(epochs):
        start = time.time()
        logger.info("Epoch {}: Start".format(e))
        loss_batch = []

        model.train()  # Set model to training mode

        for i, batch in enumerate(dataloader):
            logger.info("Epoch {}, Batch {}".format(e, i))
            # Data in minibatch format N x C x H x H
            X = batch['input'].to(device)
            y = batch['target'].to(device)
            prediction = model(X)  # [N, 1]
            #logger.info("Prediction: {}, Target: {}".format(prediction, y))
            loss = criterion(prediction, y)

            # L1 or L2 regularization
            if reg_type is not None:
                assert reg_type in ['l2', 'l1']
                if reg_type == 'l2':
                    for p in model.parameters():
                        loss += lambda_ * p.pow(2).sum()
                elif reg_type == 'l1':
                    with torch.no_grad():
                        for p in model.parameters():
                            p.sub_(p.sign() * p.abs().clamp(max=lambda_))
                else:
                    raise NotImplementedError

            loss_batch.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_epoch_loss_train.append(np.array(loss_batch).sum() / (len(dataloader) * len(X)))
        std_epoch_loss_train.append(np.array(loss_batch).std())

        end = time.time() - start
        logger.info(
            "Epoch {}: Duration: {:.02f}s, Train Loss: {:.02e}".format(
                e, end, avg_epoch_loss_train[e]))

    return avg_epoch_loss_train, std_epoch_loss_train, model


class GdpytTensorDataset(Dataset):

    __name__ = 'GdpytTensorDataset'

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
        self._shape = None
        self.stats = None
        self._mode = None

    def __len__(self):
        return len(self._source)

    def __getitem__(self, idx):
        source_particle = self._source[idx]

        target = source_particle.z
        image = source_particle.get_template(resize=self._shape)

        if self.normalize:
            image = (image - self.stats['mean']) / self.stats['std']

        # Add channel dimension if array is only a 2D image
        if len(image.shape) == 2:
            image = np.nan_to_num(image[:, :, np.newaxis])
        else:
            image = np.nan_to_num(image)
        target = np.array([target])

        if self._mode in ['train', 'test']:
            sample = {'input': image, 'target': target}
        else:
            sample = {'input': image}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _compute_stats(self):
        imgs = []
        for particle in self._source:
            imgs.append(particle.get_template(resize=self.shape))
        imgs = np.array(imgs)
        self.stats = {'mean': imgs.mean(), 'std': imgs.std()}

    def _load_calib_stack(self, stack, skip_na=True):
        all_ = []
        for particle in stack.particles:
            if stack.shape == self.shape:
                template = particle.get_template()
            else:
                template = particle.get_template(resize=self.shape)
            if skip_na and np.isnan(template).sum() != 0:
                continue
            all_.append(particle)
        return all_

    def from_calib_set(self, calib_set, max_size=None, skip_na=True, min_stack_len=10):
        # Identify largest template in calibration set
        w_max, h_max = (0, 0)
        skip_stacks = []
        for stack_id, stack in calib_set.calibration_stacks.items():
            w, h = stack.shape
            if min_stack_len is not None:
                if len(stack) < min_stack_len:
                    skip_stacks.append(stack_id)
                    continue
            if max_size is not None:
                if w > max_size or h > max_size:
                    skip_stacks.append(stack_id)
                    continue
            if w > w_max:
                w_max = w
            if h > h_max:
                h_max = h
        logger.info("Max. size specified: {}. Shape of calibration set: {}".format(max_size, (w_max, h_max)))
        self._shape = (w_max, h_max)

        # Load all calibration stacks in this calibration set
        all_ = []
        for stack_id, stack in calib_set.calibration_stacks.items():
            if stack_id not in skip_stacks:
                all_ += self._load_calib_stack(stack, skip_na=skip_na)
        self._source = all_
        logger.info("Created a {} as a training set using {} particles from calibration set".format(self.__name__, len(all_)))

        # When loading from calibration set or stack it's always a training set
        self._mode = 'train'
        self._compute_stats()

    def from_image_collection(self, collection, ref_shape=None, max_size=None, skip_na=True):
        if ref_shape is None:
            logger.error("A shape as a 2 element tuple when loading a test set from an image collection. "
                         "This should be the shape of a sample from the training set")
            raise TypeError
        else:
            assert isinstance(ref_shape, tuple)
            assert len(ref_shape) == 2
            self._shape = ref_shape

        all_ = []
        for image in collection.images.values():
            for particle in image.particles:
                if max_size is not None:
                    w, h = particle.bbox[2:]
                    if w > max_size or h > max_size:
                        continue
                template = particle.get_template(resize=ref_shape)
                if skip_na and np.isnan(template).sum() != 0:
                    continue
                all_.append(particle)

        self._source = all_

        if collection.is_infered():
            logger.info(
            "Created a {} as a test set using {} particles from "
            "GdpytImageCollection in {}".format(self.__name__, len(all_), collection.folder))
            self._mode = 'test'
        else:
            logger.info(
                "Created a {} as a prediction set (unknown targets) using {} particles from "
                "GdpytImageCollection in {}".format(self.__name__, len(all_), collection.folder))
            self._mode = 'predict'
        self._compute_stats()

    def infer(self, model, idx):
        """
        Infere a sample in the dataset with a trained model
        """
        x = self.__getitem__(idx)['input']
        # Force mini-batch shape
        x.unsqueeze_(0)

        # Evaluation mode
        model.eval()
        y = model(x)

        if self._mode in ['train', 'test']:
            target = self.__getitem__(idx)['target']
            logger.info("Predicted: {}, Target: {}".format(y.item(), target.item()))
            return y.item(), target.item()
        else:
            logger.info("Predicted: {}".format(y.item()))
            return y.item()


    def plot(self, N):
        assert isinstance(N, int) and N > 0
        fig = plot_tensor_dset(self, N)
        return fig

    def return_dataloader(self, batch_size=4, shuffle=True, num_workers=4):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def set_sample_z(self, idx, z):
        if isinstance(z, torch.Tensor):
            z = z.item()
        self._source[idx].set_z(z)

    @property
    def input_shape(self):
        return
    @property
    def shape(self):
        return self._shape

    @property
    def mode(self):
        return self._mode

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        outp_sample = {}
        image = sample['input']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        outp_sample.update({'input': torch.from_numpy(image).float()})

        if 'target' in sample.keys():
            target = sample['target']
            outp_sample.update({'target': torch.from_numpy(target).float()})

        return outp_sample

class RotateN90(object):
    """ Rotate an image by a multiple of 90 degrees"""

    def __init__(self, n):
        assert isinstance(n, int)
        self.n = n

    def __call__(self, sample):
        image = sample['input']
        sample.update({'input': np.rot90(image, k=self.n, axes=(0,1))})

        return sample