import numpy as np
import tensorflow as tf
import json


class FullyConvNetwork:

    def __init__(self, num_classes=None):
        self.counters = {
            'conv_unit': 0,
            'conv.count': 0,
            'maxpool.count': 0,
            'maxout.count': 0
        }
        self.num_classes = num_classes
        self.reg = None

    def get(self):

        X_global = tf.placeholder(tf.float32, [None, 53, 53, 53, 1], name="input_global")
        X_local = tf.placeholder(tf.float32, [None, 33, 33, 33, 1], name="input_local")

        # model = self.conv_unit(X_local, injected_input=self.conv_unit(X_global, class_count=self.num_classes), class_count=self.num_classes)
        model = self.conv_unit(X_local, class_count=self.num_classes)
        return {
            "X": {
                'global': X_global,
                'local': X_local
            },
            "Y" : tf.placeholder(tf.float32, [None, self.num_classes], name="y"),
            "reg": self.reg,
            "model": model
        }


    def conv(self, input_tensor, filter_size=None, filter_stride=None, filter_count=None, name=None):

        output_shape = None
        input_shape = input_tensor.get_shape().as_list()

        if input_shape[0] is None:
            input_shape = input_shape[1:]
        input_dim = input_shape[:-1]
        input_chan = input_shape[-1]

        shape_dim = np.repeat(filter_size, len(input_dim))
        shape_chan = filter_count

        output_shape = np.array(input_dim) - np.array(shape_dim)
        output_shape = np.float16(output_shape)
        output_shape /= filter_stride
        output_shape += 1

        output_shape = np.append(output_shape, filter_count)

        w_shape = list(shape_dim)
        w_shape.append(input_chan)
        w_shape.append(shape_chan)

        b_shape = [filter_count]

        stride_list = [1, filter_stride, filter_stride, filter_stride, 1]

        output = None
        with tf.variable_scope(name):
            w = tf.get_variable("Weights", dtype=tf.float32, shape=w_shape,initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("Biases", dtype=tf.float32, shape=b_shape, initializer=tf.constant_initializer(0.0))
            output = tf.nn.conv3d(input_tensor, w, strides=stride_list, padding='VALID', name="OpConv")
            output = tf.nn.bias_add(output, b, name="OpBias")
            if self.reg is None:
                self.reg= tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
            else:
                self.reg += tf.nn.l2_loss(w) + tf.nn.l2_loss(b)

        return output

    def maxpool(self, input_tensor, pooling_size=None, pooling_stride=None, name=None):
        strides_list = [1, pooling_stride, pooling_stride, pooling_stride, 1]
        k_list = [1, pooling_size, pooling_size, pooling_size, 1]
        output = None
        with tf.variable_scope(name):
            output = tf.nn.max_pool3d(input_tensor, ksize=k_list, strides=strides_list, padding='VALID', name="OpPool")
        return output

    def maxout(self, inputs, num_units, axis=None, name=None):
        shape = inputs.get_shape().as_list()
        if shape[0] is None:
            shape[0] = -1
        if axis is None:  # Assume that channel is the last dimension
            axis = -1
        num_channels = shape[axis]
        if num_channels % num_units:
            raise ValueError('number of features({}) is not '
                             'a multiple of num_units({})'.format(num_channels, num_units))
        shape[axis] = num_units
        shape += [num_channels // num_units]
        with tf.variable_scope(name):
            outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
        return outputs

    def softmax(self, target, axis, name=None):
        with tf.name_scope(name, 'softmax', values=[target]):
            max_axis = tf.reduce_max(target, axis, keep_dims=True)
            target_exp = tf.exp(target - max_axis)
            normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
            softmax = target_exp / normalize
            return softmax

    def conv_unit(self, input_tensor, injected_input=None, class_count=None):
        output = None
        with tf.variable_scope("ConvUnit" + str(self.counters['conv_unit'])):
            # Conv 7x7
            path_1 = self.conv(input_tensor, filter_size=7, filter_stride=1, filter_count=48, name="OpConv0")
            # Relu
            path_1 = tf.nn.relu(path_1)
            # Pooling 4x4
            path_1 = self.maxpool(path_1, pooling_size=4, pooling_stride=1, name="OpPool0")
            # Conv 3x3
            path_1 = self.conv(path_1, filter_size=3, filter_stride=1, filter_count=48, name="OpConv1")
            # Relu
            path_1 = tf.nn.relu(path_1)
            # Pooling 2x2
            path_1 = self.maxpool(path_1, pooling_size=2, pooling_stride=1, name="OpPool1")

            path_2 = self.conv(input_tensor, 13, 1, 80, "OpConv2")
            path_2 = tf.nn.relu(path_2)

            concat_list = [path_1, path_2]
            if injected_input is not None:
                concat_list.append(injected_input)
                print(path_1.shape)
                print(path_2.shape)
                print(injected_input)
            output = tf.concat(concat_list, 4)

            #print(path_1)
            #print(path_2)
            #print(output)

            # Conv 21x21
            output = self.conv(output, filter_size=21, filter_stride=1, filter_count=class_count, name="OpConv3")
            # output = tf.squeeze(output)
            # output = self.softmax(output, 1, name="OpSoftmax0")

            print(output)

        self.counters['conv_unit'] += 1
        return output


