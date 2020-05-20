import keras.backend as K
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras import Input, utils
from keras.initializers import Constant
from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv2D, Cropping2D, GlobalAveragePooling2D,
                          GlobalMaxPooling2D, Input, Lambda, Layer, LeakyReLU,
                          MaxPooling2D, ReLU, Reshape, Softmax, Subtract,
                          UpSampling2D, ZeroPadding2D, add)
from keras.models import Model
from keras.activations import sigmoid

from special_paddings import *


class GaussianLayer(Layer):
    """ Computes noise std. dev. for Gaussian noise model. """
    def __init__(self, **kwargs):
        super(GaussianLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if not input_shape:
            # global parameter
            self.b = self.add_weight(name='b', 
                                        shape=(),
                                        initializer=Constant(0),
                                        trainable=True)
        super(GaussianLayer, self).build(input_shape)

    def call(self, x):
        noise_std = K.softplus(self.b-4)+1e-3
        return noise_std

    def compute_output_shape(self, input_shape):
        if not input_shape:
            return ()
        else:
            return input_shape

class PoissonLayer(Layer):
    """ Computes input-dependent noise std. dev. for Poisson noise model. """
    def __init__(self, **kwargs):
        super(PoissonLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a = self.add_weight(name='a', 
                                      shape=(),
                                      initializer=Constant(0),
                                      trainable=True)
        super(PoissonLayer, self).build(input_shape)

    def call(self, x):
        noise_est = K.softplus(self.a-4) + 1e-3
        noise_std = (K.maximum(x, 1e-3) * noise_est) ** 0.5
        return noise_std

    def compute_output_shape(self, input_shape):
        return input_shape

class PoissonGaussianLayer(Layer):
    """ Computes input-dependent noise std. dev. for Poisson-Gaussian noise model. """
    def __init__(self, **kwargs):
        super(PoissonGaussianLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a = self.add_weight(name='a', 
                                      shape=(),
                                      initializer=Constant(0),
                                      trainable=True)
        self.b = self.add_weight(name='b', 
                                      shape=(),
                                      initializer=Constant(0),
                                      trainable=True)
        super(PoissonGaussianLayer, self).build(input_shape)

    def call(self, x):
        poisson_noise_est = K.softplus(self.a-4) + 1e-3
        poisson_noise_var = K.maximum(x, 1e-3) * poisson_noise_est
        noise_var = K.maximum(poisson_noise_var + self.b,1e-3)
        noise_std = noise_var**0.5
        return noise_std

    def compute_output_shape(self, input_shape):
        return input_shape

def mse_loss(y,loc):
    """ Mean squared error loss function
        Use mean-squared error to regress to the expected value
        Parameters:
            loc: mean
    """
    loss = (y-loc)**2
    return K.mean(loss)

def uncalib_gaussian_loss(y,loc,std):
    """ Uncalibrated Gaussian loss function
        Model noisy data using a Gaussian parameterized by mean and std. dev.
        Parameters:
            loc: mean
            std: std. dev.
    """
    var = std**2
    total_var = var+1e-3
    loss = (y-loc)**2 / total_var + tf.log(total_var)
    return K.mean(loss)

def uncalib_gaussian_mixture_loss(y,loc,std,a):
    """ Negative log likelihood from mixture of Gaussians
        Parameters:
            y: inputs
            loc: means
            std: standard devs
            a: mixture coefficients
    """
    mixture = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(probs=a, validate_args=True),
        components_distribution=tfp.distributions.Normal(
            loc=loc,
            scale=std,
            validate_args=True,
            allow_nan_stats=False
        ),
        name="mixture"
    )
    y = K.squeeze(y, axis=-1)
    log_likelihood = mixture.log_prob(y, name="log_prob")
    return -K.mean(log_likelihood)

def gaussian_loss(y,loc,std,noise_std,reg_weight):
    """ Gaussian loss function
        Model noisy data using a Gaussian prior and Gaussian noise model
        Parameters:
            y: noisy input image
            loc: prior mean
            std: prior std. dev.
            noise_std: noise std. dev.
            reg_weight: strength of regularization on prior std. dev.
    """
    var = std**2
    noise_var = noise_std**2
    total_var = var+noise_var
    loss = (y-loc)**2 / total_var + tf.log(total_var)
    reg = reg_weight * K.abs(std)
    return K.mean(loss+reg)

def gaussian_posterior_mean(y,loc,std,noise_std):
    """ Gaussian posterior mean
        Given noisy observation (y), compute optimal estimate for denoised image 
            y: noisy input image
            loc: prior mean
            std: prior std. dev.
            noise_std: noise std. dev.
    """
    var = std**2
    noise_var = noise_std**2
    total_var = var+noise_var
    return (loc*noise_var + var*y)/total_var
  
def _conv(x, num_filters, name):
  """ 2d convolution """
  filter_size = [3,3]

  x = Conv2D(filters=num_filters, kernel_size=filter_size, padding='same', kernel_initializer='he_normal', name=name)(x)
  x = LeakyReLU(0.1)(x)

  return x

def _vshifted_conv(x, num_filters, name, activate=True):
    """ Vertically shifted convolution """
    filter_size = [3,3]
    k = filter_size[0]//2

    x = ZeroPadding2D([[k,0],[0,0]])(x)
    x = Conv2D(filters=num_filters, kernel_size=filter_size, padding='same', kernel_initializer='he_normal', name=name)(x)
    x = Cropping2D([[0,k],[0,0]])(x)
    if activate:
        x = LeakyReLU(0.1)(x)

    return x

def _pool(x):
  """ max pooling"""
  x = MaxPooling2D(pool_size=2,strides=2,padding='same')(x)

  return x

def _vshifted_pool(x):
  """ Vertically shifted max pooling"""
  x = ZeroPadding2D([[1,0],[0,0]])(x)
  x = Cropping2D([[0,1],[0,0]])(x)

  x = MaxPooling2D(pool_size=2,strides=2,padding='same')(x)

  return x


"""
keras resnet50
https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py
"""

def identity_block(input_tensor, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    # x = Activation('relu')(x)
    x = LeakyReLU(0.1)(x)

    # x = Conv2D(filters2, kernel_size,
    #                   padding='same',
    #                   kernel_initializer='he_normal',
    #                   name=conv_name_base + '2b')(x)
    x = _vshifted_conv(x, filters2, conv_name_base + '2b', activate=False)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    # x = Activation('relu')(x)
    x = LeakyReLU(0.1)(x)

    x = Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor,
               filters,
               stage,
               block):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=1,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    # x = Activation('relu')(x)
    x = LeakyReLU(0.1)(x)

    # x = Conv2D(filters2, kernel_size, padding='same',
    #                   kernel_initializer='he_normal',
    #                   name=conv_name_base + '2b')(x)
    x = _vshifted_conv(x, filters2, conv_name_base + '2b', activate=False)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    # x = Activation('relu')(x)
    x = LeakyReLU(0.1)(x)

    x = Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=1,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(input_tensor,
             input_shape,
             pooling=None):
    """Instantiates the ResNet50 architecture.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
    # Returns
        A Keras model instance.
    """

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    # x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    # x = Conv2D(64, (7, 7),
    #                   strides=(2, 2),
    #                   padding='valid',
    #                   kernel_initializer='he_normal',
    #                   name='conv1')(x)
    # x = Activation('relu')(x)
    x = _vshifted_conv(img_input, 48, 'conv1a', activate=False)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1a')(x)
    x = LeakyReLU(0.1)(x)
    # x = _vshifted_conv(x, 48, 'conv1b')
    # x = BatchNormalization(axis=bn_axis, name='bn_conv1b')(x)
    # x = LeakyReLU(0.1)(x)

    # x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    # x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, [48, 48, 96], stage=2, block='a')
    x = identity_block(x, [48, 48, 96], stage=2, block='b')
    x = identity_block(x, [48, 48, 96], stage=2, block='c')

    x = conv_block(x, [96, 96, 192], stage=3, block='a')
    x = identity_block(x, [96, 96, 192], stage=3, block='b')
    x = identity_block(x, [96, 96, 192], stage=3, block='c')
    x = identity_block(x, [96, 96, 192], stage=3, block='d')

    x = conv_block(x, [96, 96, 192], stage=4, block='a')
    x = identity_block(x, [96, 96, 192], stage=4, block='b')
    x = identity_block(x, [96, 96, 192], stage=4, block='c')
    x = identity_block(x, [96, 96, 192], stage=4, block='d')
    x = identity_block(x, [96, 96, 192], stage=4, block='e')
    x = identity_block(x, [96, 96, 192], stage=4, block='f')

    x = conv_block(x, [192, 192, 384], stage=5, block='a')
    x = identity_block(x, [192, 192, 384], stage=5, block='b')
    x = identity_block(x, [192, 192, 384], stage=5, block='c')

    # final pad and crop for blind spot
    x = ZeroPadding2D([[1,0],[0,0]])(x)
    x = Cropping2D([[0,1],[0,0]])(x)

    if pooling is not None:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    inputs = utils.get_source_inputs(input_tensor)
    
    # Create model
    model = Model(inputs, x, name='resnet50')

    return model


def _vertical_blindspot_network(x):
  """ Blind-spot network; adapted from noise2noise GitHub
    Each row of output only sees input pixels above that row
  """
  skips = [x]

  n = x
  n = _vshifted_conv(n, 48, 'enc_conv0')
  n = _vshifted_conv(n, 48, 'enc_conv1')
  n = _vshifted_pool(n)
  skips.append(n)

  n = _vshifted_conv(n, 48, 'enc_conv2')
  n = _vshifted_pool(n)
  skips.append(n)

  n = _vshifted_conv(n, 48, 'enc_conv3')
  n = _vshifted_pool(n)
  skips.append(n)

  n = _vshifted_conv(n, 48, 'enc_conv4')
  n = _vshifted_pool(n)
  skips.append(n)

  n = _vshifted_conv(n, 48, 'enc_conv5')
  n = _vshifted_pool(n)
  n = _vshifted_conv(n, 48, 'enc_conv6')

  #-----------------------------------------------
  n = UpSampling2D(2)(n)
  n = Concatenate(axis=3)([n, skips.pop()])
  n = _vshifted_conv(n, 96, 'dec_conv5')
  n = _vshifted_conv(n, 96, 'dec_conv5b')

  n = UpSampling2D(2)(n)
  n = Concatenate(axis=3)([n, skips.pop()])
  n = _vshifted_conv(n, 96, 'dec_conv4')
  n = _vshifted_conv(n, 96, 'dec_conv4b')

  n = UpSampling2D(2)(n)
  n = Concatenate(axis=3)([n, skips.pop()])
  n = _vshifted_conv(n, 96, 'dec_conv3')
  n = _vshifted_conv(n, 96, 'dec_conv3b')

  n = UpSampling2D(2)(n)
  n = Concatenate(axis=3)([n, skips.pop()])
  n = _vshifted_conv(n, 96, 'dec_conv2')
  n = _vshifted_conv(n, 96, 'dec_conv2b')

  n = UpSampling2D(2)(n)
  n = Concatenate(axis=3)([n, skips.pop()])
  n = _vshifted_conv(n, 96, 'dec_conv1a')
  n = _vshifted_conv(n, 96, 'dec_conv1b')

  # final pad and crop for blind spot
  n = ZeroPadding2D([[1,0],[0,0]])(n)
  n = Cropping2D([[0,1],[0,0]])(n)

  return n



def vshift_conv_2(x, channels_out, name, kernel_size=3, strides=1, bias=True, pad="zero"):
    k = kernel_size//2

    if pad == "reflection":
        x = ReflectionPadding2D([k,0,0,0], name="refpad-"+name)(x)
    else:
        x = ZeroPadding2D([[k,0],[0,0]], name="zeropad-"+name)(x)

    x = Conv2D(filters=channels_out, kernel_size=kernel_size, strides=strides, padding="same",
            kernel_initializer='he_normal', use_bias=bias, name="conv-"+name)(x)
    x = Cropping2D([[0,k],[0,0]], name="crop-"+name)(x)

    return x


def resnet_v2(inputs, num_blocks=10, num_channels=48, need_sigmoid=False):

    x = vshift_conv_2(inputs, num_channels, name="initial")
    x = LeakyReLU(0.2, name="relu-initial")(x)

    for i in range(num_blocks):
        num = str(i)
        bypass = x

        x = vshift_conv_2(x, num_channels, name=num+"a")
        x = BatchNormalization(name="norm-"+num+"a")(x)
        x = LeakyReLU(0.2, name="relu-"+num)(x)
        x = vshift_conv_2(x, num_channels, name=num+"b")
        x = BatchNormalization(name="norm-"+num+"b")(x)

        x = Concatenate(axis=3)([x, bypass])
        # x = Add(name="add-"+num)([x, bypass])

    x = vshift_conv_2(x, num_channels, name="final-1")
    x = BatchNormalization(name="norm-final")(x)

    x = vshift_conv_2(x, num_channels, name="final-2")
    if need_sigmoid:
        x = Activation("sigmoid", name="final-sigmoid")(x)

    return x


def blindspot_network(inputs):
    b,h,w,c = K.int_shape(inputs)
    #if h != w:
        #raise ValueError('input shape must be square')
    if h % 32 != 0 or w % 32 != 0:
        raise ValueError('input shape (%d x %d) must be divisible by 32'%(h,w))

    # make vertical blindspot network
    vert_input = Input([h,w,c])
    # vert_output = _vertical_blindspot_network(vert_input)
    vert_output = resnet_v2(vert_input)
    vert_model = Model(inputs=vert_input,outputs=vert_output)

    # run vertical blindspot network on rotated inputs
    stacks = []
    for i in range(4):
        rotated = Lambda(lambda x: tf.image.rot90(x,i))(inputs)
        if i == 0 or i == 2:
            rotated = Reshape([h,w,c])(rotated)
        else:
            rotated = Reshape([w,h,c])(rotated)
        out = vert_model(rotated)
        out = Lambda(lambda x:tf.image.rot90(x,4-i))(out)
        stacks.append(out)

    # concatenate outputs
    x = Concatenate(axis=3)(stacks)

    # final 1x1 convolutional layers
    x = Conv2D(384, 1, kernel_initializer='he_normal', name='conv1x1_1')(x)
    x = LeakyReLU(0.1)(x)

    x = Conv2D(96, 1, kernel_initializer='he_normal', name='conv1x1_2')(x)
    x = LeakyReLU(0.1)(x)

    return x


def gaussian_blindspot_network(input_shape,mode,reg_weight=0,components=1):
    """ Create a variant of the Gaussian blindspot newtork.
        input_shape: Shape of input image
        mode: mse, uncalib, global, perpixel, poisson
              mse              -- regress to expected value using mean squared error loss 
              uncalib          -- model prior and noise together with single Gaussian at each pixel
              gaussian         -- Gaussian noise
              poisson          -- Poisson noise
              poissongaussian  -- Poisson-Gaussian noise
        reg_weight: strength of regularization on prior std. dev.
        components: number of mixture components (distributions) for each pixel
    """ 
    # create input layer
    inputs = Input(input_shape)
  
    # run blindspot network
    x = blindspot_network(inputs)

    # get prior parameters
    loc = Conv2D(components, 1, kernel_initializer='he_normal', name='loc')(x)
    if mode != 'mse':
        # standard deviation
        std = Conv2D(components, 1, kernel_initializer='he_normal', name='std')(x)
        if components != 1:
            # std cannot be negative or zero for mixture
            std = Lambda(lambda x: K.softplus(x-4) + 1e-3, name="std-softplus")(std)
            # mixture coefficient
            a = Conv2D(components, 1, kernel_initializer="he_normal", name="a")(x)
            a = Softmax(name="a-softmax")(a)

    # get noise variance
    if mode == 'mse':
        pass
    elif mode == 'uncalib':
        pass
    elif mode == 'gaussian':
        noise_std = GaussianLayer()([])
    elif mode == 'poisson':
        noise_std = PoissonLayer()(loc)
    elif mode == 'poissongaussian':
        noise_std = PoissonGaussianLayer()(loc)
    else:
        raise ValueError('unknown mode %s'%mode)
    
    # get outputs
    if mode == 'mse':
        outputs = loc
    elif mode == 'uncalib':
        if components == 1:
            outputs = [loc,std]
        else:
            outputs = [loc,std,a]
    else:
        outputs = Lambda(lambda x:gaussian_posterior_mean(*x))([inputs,loc,std,noise_std])

    # create model
    model = Model(inputs=inputs,outputs=outputs)

    # create loss function
    # input is evaluated against output distribution
    if mode == 'mse':
        loss = mse_loss(inputs,loc)
    elif mode == 'uncalib':
        if components == 1:
            loss = uncalib_gaussian_loss(inputs, loc, std)
        else:
            loss = uncalib_gaussian_mixture_loss(inputs,loc,std,a)
    else:
        loss = gaussian_loss(inputs,loc,std,noise_std,reg_weight)
    model.add_loss(loss)
  
    return model
