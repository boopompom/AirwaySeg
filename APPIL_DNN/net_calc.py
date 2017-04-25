import numpy as np
import tensorflow as tf
import json


class NetCalc:

    def getX(self):
        if self.X is None:
            input_tf_shape = [None]
            input_tf_shape.extend(self.input_shape)
            self.X = tf.placeholder(self.data_format, input_tf_shape)
        return self.X

    def get_tensor_size(self, shape, placeholders=None):

        total_size = 1
        placeholder_idx = 0

        for i in shape:
            if i == -1 or i is None:
                i = placeholders[placeholder_idx]
                placeholder_idx += 1

            total_size *= i
        return total_size * 4

    def __init__(self, input_shape, n_classes, batch_size, print_only=False):

        self.input_shape = input_shape
        self.n_classes = n_classes
        self.batch_size = batch_size

        self.conv_dim = len(self.input_shape) - 1

        self.parts = []
        self.X = None
        self.conv_counter = 0
        self.memory_footprint = 0
        self.data_format = tf.float32
        self.regularizers = None

        self.print_only = print_only


    def build_from_config(self, json_cfg):
        arch = json_cfg['arch']
        for i in arch:
            if i['type'] == 'conv':
                s = [i['filter_size']] * self.conv_dim
                self.conv(s, i['filter_count'], i['stride'], maintain_spatial=i['maintain_spatial'])

            if i['type'] == 'pool':
                s = [i['size']] * self.conv_dim
                self.pool(s, i['stride'])

            if i['type'] == 'fc':
                self.fc(i['count'])

        self.fc()
        return self.build()

    # Copied from http://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
    @staticmethod
    def sizeof_fmt(n, suffix='B'):
        num = n
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, 'Yi', suffix)

    def get_reg(self):
        return  self.regularizers

    def fc_b(self, x, w, b, d, name, is_out=False):

        # Fully connected layer
        net = tf.reshape(x, [-1, w.get_shape().as_list()[0]])
        net = tf.add(tf.matmul(net, w), b)
        if not is_out:
            net = tf.nn.relu(net)
            net = tf.nn.dropout(net, 0.5)

        if self.regularizers is None:
            self.regularizers = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
        else :
            self.regularizers += tf.nn.l2_loss(w) + tf.nn.l2_loss(b)

        return net


    def conv_b(self, x, W, b, name, stride=1, padding='VALID'):

        conv_fn = tf.nn.conv2d if self.conv_dim == 2 else tf.nn.conv3d
        strides_list = [1, stride, stride, 1] if self.conv_dim == 2 else [1, stride, stride, stride, 1]
        net = conv_fn(x, W, strides=strides_list, padding=padding, name=name)
        net = tf.nn.bias_add(net, b)
        net = tf.nn.relu(net)
        return net

    def pool_b(self, x, name, stride=2, k=2):

        maxpool_fn = tf.nn.max_pool if self.conv_dim == 2 else tf.nn.max_pool3d
        strides_list = [1, stride, stride, 1] if self.conv_dim == 2 else [1, stride, stride, stride, 1]
        k_list = [1, k, k, 1] if self.conv_dim == 2 else [1, k, k, k, 1]

        return maxpool_fn(x, ksize=k_list, strides=strides_list, padding='VALID', name=name)

    def build(self):

        print("\n\n")
        input_shape = np.array(self.input_shape)
        input_data = self.X
        for idx, part in enumerate(self.parts):


            memory_size = {}
            params = {}
            output_shape = None

            if part['type'] == 'pool':
                input_dim = input_shape[:-1]
                input_chan = input_shape[-1]

                shape_dim = part['shape']

                output_shape = np.array(input_dim) - np.array(shape_dim)
                output_shape = np.float16(output_shape)
                output_shape /= part['stride']
                output_shape += 1

                params['stride'] = part['stride']
                params['size'] = part['shape'][0]

                output_shape = np.append(output_shape, input_chan)
                if self.print_only is not True:
                    input_data = self.pool_b(input_data, part['name'], stride=part['stride'], k=shape_dim[0])

            if part['type'] == 'fc':
                input_dim = input_shape[:-1]
                input_chan = input_shape[-1]
                o = part['n']

                if o is None:
                    o = self.n_classes

                sz = 1
                for i in input_dim:
                    sz *= i
                sz *= input_chan
                w_shape = [sz, o]
                b_shape = [o]

                memory_size['W'] = self.get_tensor_size(w_shape, [])
                memory_size['b'] = self.get_tensor_size(b_shape, [])
                self.memory_footprint += memory_size['W'] + memory_size['b']
                params['count'] = o

                if self.print_only is not True:
                    w = tf.get_variable("W_" + part['name'], shape=w_shape,
                                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.random_normal(b_shape))
                    input_data = self.fc_b(input_data, w, b, None, part['name'], is_out=(idx ==len(self.parts) -1))

                output_shape = [o]

            if part['type'] == 'conv':
                input_dim = input_shape[:-1]
                input_chan = input_shape[-1]

                shape_dim = part['shape']
                shape_chan = part['count']

                if part['maintain_spatial']:
                    output_shape = list(input_dim)
                else :
                    output_shape = np.array(input_dim) - np.array(shape_dim)
                    output_shape = np.float16(output_shape)
                    output_shape /= part['stride']
                    output_shape += 1

                output_shape = np.append(output_shape, part['count'])

                w_shape = list(shape_dim)
                w_shape.append(input_chan)
                w_shape.append(shape_chan)

                b_shape = [part['count']]

                memory_size['W'] = self.get_tensor_size(w_shape, [])
                memory_size['b'] = self.get_tensor_size(b_shape, [])
                self.memory_footprint += memory_size['W'] + memory_size['b']
                params['count'] = part['count']
                params['size'] = shape_dim[0]
                params['stride'] = part['stride']

                if self.print_only is not True:
                    w = tf.get_variable("W_" + part['name'], shape=w_shape,
                                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.random_normal(b_shape))
                    p = 'SAME' if part['maintain_spatial'] else 'VALID'
                    input_data = self.conv_b(input_data, w, b, part['name'], stride=part['stride'], padding=p)

            input_memory_size = self.get_tensor_size(input_shape) * self.batch_size
            self.memory_footprint += input_memory_size
            print("\t" * 3 + str(input_shape) + " - " + self.sizeof_fmt(input_memory_size))
            print("\t" * 3 + "↓")
            print("\t" * 3 + part['name'] + " - " + str(params) + " - " + str(memory_size))
            print("\t" * 3 + "↓")
            input_shape = output_shape

        print("\t" * 3 + "Output")
        print("\n\n")
        print("Total size of the model " + self.sizeof_fmt(self.memory_footprint))
        if self.print_only is not True:
            return input_data
        return self


    def conv(self, shape, count, stride, maintain_spatial=False):
        self.conv_counter += 1
        self.parts.append({
            "name": "Conv" + str(self.conv_counter),
            "shape": shape,
            "count": count,
            "stride": stride,
            "maintain_spatial": maintain_spatial,
            "type": "conv"
        })
        return self

    def pool(self, shape, stride):
        # shape.append(0)
        self.parts.append({
            "name": "Pool" + str(self.conv_counter),
            "shape": shape,
            "stride": stride,
            "type": "pool"
        })
        return self

    def fc(self, n=None):
        self.conv_counter += 1

        self.parts.append({
            "name": "FC" + str(self.conv_counter),
            "n": n,
            "type": "fc"
        })
        return self


