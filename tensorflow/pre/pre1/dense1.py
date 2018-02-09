import warnings

from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, multiply, Lambda
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import Layer
import keras.backend as K
from keras.backend import tf as ktf

import numpy as np
import tensorflow as tf

import sys
sys.path.append('/home/albert/github/DenseNet/')
import densenet

DENSENET_121_WEIGHTS_PATH = r'https://github.com/titu1994/DenseNet/releases/download/v3.0/DenseNet-BC-121-32.h5'
DENSENET_161_WEIGHTS_PATH = r'https://github.com/titu1994/DenseNet/releases/download/v3.0/DenseNet-BC-161-48.h5'
DENSENET_169_WEIGHTS_PATH = r'https://github.com/titu1994/DenseNet/releases/download/v3.0/DenseNet-BC-169-32.h5'
DENSENET_121_WEIGHTS_PATH_NO_TOP = r'https://github.com/titu1994/DenseNet/releases/download/v3.0/DenseNet-BC-121-32-no-top.h5'
DENSENET_161_WEIGHTS_PATH_NO_TOP = r'https://github.com/titu1994/DenseNet/releases/download/v3.0/DenseNet-BC-161-48-no-top.h5'
DENSENET_169_WEIGHTS_PATH_NO_TOP = r'https://github.com/titu1994/DenseNet/releases/download/v3.0/DenseNet-BC-169-32-no-top.h5'

# Model construction
                
def Resize(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, im):
        im_reshaped = K.reshape(im, (-1, im.get_shape()[1].value, im.get_shape()[2].value, 1))
        r = ktf.image.resize_images(im_reshaped, (self.output_dim[0], self.output_dim[1]))
        tile = K.tile(r, [1, 1, 1, self.output_dim[2]])
        return tile

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def __create_dense_net(output_dim, img_input, cam_input, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1,
                       nb_layers_per_block=-1, bottleneck=False, reduction=0.0, dropout_rate=None, weight_decay=1e-4,
                       subsample_initial_block=False, activation='softmax',
                       cam_placement=None):

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, 'reduction value must lie between 0.0 and 1.0'

    # layers in each dense block
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == (nb_dense_block), 'If list, nb_layer is used as provided. ' \
                                                   'Note that list size must be (nb_dense_block)'
        final_nb_layer = nb_layers[-1]
        nb_layers = nb_layers[:-1]
    else:
        if nb_layers_per_block == -1:
            assert (depth - 4) % 3 == 0, 'Depth must be 3 N + 4 if nb_layers_per_block == -1'
            count = int((depth - 4) / 3)
            nb_layers = [count for _ in range(nb_dense_block)]
            final_nb_layer = count
        else:
            final_nb_layer = nb_layers_per_block
            nb_layers = [nb_layers_per_block] * nb_dense_block

    # compute initial nb_filter if -1, else accept users initial nb_filter
    if nb_filter <= 0:
        nb_filter = 2 * growth_rate

    # compute compression factor
    compression = 1.0 - reduction

    # Initial convolution
    if subsample_initial_block:
        initial_kernel = (7, 7)
        initial_strides = (2, 2)
    else:
        initial_kernel = (3, 3)
        initial_strides = (1, 1)

    x = Conv2D(nb_filter, initial_kernel, kernel_initializer='he_normal', padding='same',
               strides=initial_strides, use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)

    if subsample_initial_block:
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = densenet.__dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, bottleneck=bottleneck,
                                     dropout_rate=dropout_rate, weight_decay=weight_decay)
        # add transition_block
        x = densenet.__transition_block(x, nb_filter, compression=compression, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

        if cam_placement != None:
            if np.any(block_idx == np.array(cam_placement)):
                out = Lambda(lambda im : K.reshape(im, (-1, im.get_shape()[1].value, im.get_shape()[2].value, 1)))(cam_input)
                # Must declare s and n beforehand, or else leads to errors
                s_x, s_y = int(x.get_shape()[1].value), int(x.get_shape()[2].value)
                print s_x, s_y
                out = Lambda(lambda im : tf.image.resize_images(im, (s_x, s_y)))(out)
                n = int(x.get_shape()[3].value)
                out = Lambda(lambda i : K.tile(i, [1, 1, 1, n]))(out)
                x = multiply([x, out])

    # The last dense_block does not have a transition_block
    x, nb_filter = densenet.__dense_block(x, final_nb_layer, nb_filter, growth_rate, bottleneck=bottleneck,
                                 dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    
    if np.any(nb_dense_block - 1 == np.array(cam_placement)):
        out = Lambda(lambda im : K.reshape(im, (-1, im.get_shape()[1].value, im.get_shape()[2].value, 1)))(cam_input)
        # Must declare s and n beforehand, or else leads to errors
        s_x, s_y = int(x.get_shape()[1].value), int(x.get_shape()[2].value)
        print s_x, s_y
        out = Lambda(lambda im : tf.image.resize_images(im, (s_x, s_y)))(out)
        n = int(x.get_shape()[3].value)
        out = Lambda(lambda i : K.tile(i, [1, 1, 1, n]))(out)
        x = multiply([x, out])
        
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, name='dense_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    predictions = Dense(output_dim, name='dense_3')(x)

    return predictions

def DenseNet(input_shape=None, cam_input_shape=None, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1, nb_layers_per_block=-1,
             bottleneck=False, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, subsample_initial_block=False,
             weights=None, output_dim=128, activation='softmax', cam_placement=None):

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `cifar10` '
                         '(pre-training on CIFAR-10).')

    if activation not in ['softmax', 'sigmoid']:
        raise ValueError('activation must be one of "softmax" or "sigmoid"')

    if activation == 'sigmoid' and classes != 1:
        raise ValueError('sigmoid activation can only be used when classes = 1')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=8,
                                      data_format=K.image_data_format(),
                                      require_flatten=False)
    cam_input_shape = (128,64)#(input_shape[0], input_shape[1])

    if cam_placement is None:
        img_input = Input(shape=input_shape, name='input_1')
        x = __create_dense_net(output_dim, img_input, None, depth, nb_dense_block,
                               growth_rate, nb_filter, nb_layers_per_block, bottleneck, reduction,
                               dropout_rate, weight_decay, subsample_initial_block, activation,
                               cam_placement)
        inputs = img_input
    else:
        img_input = Input(shape=input_shape, name='input_1')
        cam_input = Input(shape=cam_input_shape, name='input_cam')
        x = __create_dense_net(output_dim, img_input, cam_input, depth, nb_dense_block,
                               growth_rate, nb_filter, nb_layers_per_block, bottleneck, reduction,
                               dropout_rate, weight_decay, subsample_initial_block, activation,
                               cam_placement)
        inputs = [img_input, cam_input]

    # Create model.
    model = Model(inputs, x, name='densenet')

    # load weights
    if weights == 'imagenet':
        weights_loaded = False

        if (depth == 121) and (nb_dense_block == 4) and (growth_rate == 32) and (nb_filter == 64) and \
                (bottleneck is True) and (reduction == 0.5) and (dropout_rate == 0.0) and (subsample_initial_block):
            weights_path = get_file('DenseNet-BC-121-32-no-top.h5',
                                    DENSENET_121_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='55e62a6358af8a0af0eedf399b5aea99')
            model.load_weights(weights_path, by_name=True)
            weights_loaded = True

        if (depth == 161) and (nb_dense_block == 4) and (growth_rate == 48) and (nb_filter == 96) and \
                (bottleneck is True) and (reduction == 0.5) and (dropout_rate == 0.0) and (subsample_initial_block):
            weights_path = get_file('DenseNet-BC-161-48-no-top.h5',
                                    DENSENET_161_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='1a9476b79f6b7673acaa2769e6427b92')
            model.load_weights(weights_path, by_name=True)
            weights_loaded = True

        if (depth == 169) and (nb_dense_block == 4) and (growth_rate == 32) and (nb_filter == 64) and \
                (bottleneck is True) and (reduction == 0.5) and (dropout_rate == 0.0) and (subsample_initial_block):
            weights_path = get_file('DenseNet-BC-169-32-no-top.h5',
                                    DENSENET_169_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='89c19e8276cfd10585d5fadc1df6859e')
            model.load_weights(weights_path, by_name=True)
            weights_loaded = True

        if weights_loaded:
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)

            if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')

            print("Weights for the model were loaded successfully")

    return model

def DenseNetImageNet121(input_shape=None,
                        bottleneck=True,
                        reduction=0.5,
                        dropout_rate=0.0,
                        weight_decay=1e-4,
                        include_top=True,
                        weights='imagenet',
                        output_dim=128,
                        activation='softmax',
                        cam_placement=None):
    return DenseNet(input_shape, depth=121, nb_dense_block=4, growth_rate=32, nb_filter=64,
                    nb_layers_per_block=[6, 12, 24, 16], bottleneck=bottleneck, reduction=reduction,
                    dropout_rate=dropout_rate, weight_decay=weight_decay, subsample_initial_block=True,
                    weights=weights, output_dim=output_dim, activation=activation, cam_placement=cam_placement)
