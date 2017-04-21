import numpy as np
import tensorflow as tf
import json


class NetCalc:

    def get_X(self):
        for idx in self.input_shape:
            if self.X[idx] is None:
                input_tf_shape = [None]
                input_tf_shape.extend(self.input_shape[idx])
                self.X[idx] = tf.placeholder(self.data_format, input_tf_shape, name="X_" + idx)
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

    def __init__(self, input_shape, n_classes, batch_size, mode='train', print_only=False):

        self.input_shape = input_shape
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.mode = mode

        self.active_chain = None
        self.chains = {
            'out': []
        }
        self.X = {}
        self.conv_dim = {}
        self.chain_order = []
        for idx in self.input_shape:
            if idx != 'out':
                self.chain_order.append(idx)
            self.chains[idx] = []
            self.conv_dim[idx] = None
            self.X[idx] = None
            if len(self.input_shape[idx]) > 2:
                self.conv_dim[idx] = len(self.input_shape[idx]) - 1

        self.chain_order.append('out')
        self.conv_counter = 0
        self.memory_footprint = 0
        self.data_format = tf.float32
        self.regularizers = None

        self.print_only = print_only


    def build_from_config(self, json_cfg):
        chains = json_cfg
        for chain in chains:
            idx = self.active_chain = chain['id']
            for i in chain['arch']:
                if i['type'] == 'conv':
                    s = [i['filter_size']] * self.conv_dim[idx]
                    self.conv(s, i['filter_count'], i['stride'], maintain_spatial=i['maintain_spatial'])

                if i['type'] == 'pool':
                    s = [i['size']] * self.conv_dim[idx]
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
        return self.regularizers

    def fc_b(self, x, w, b, d, name, is_out=False):

        # Fully connected layer
        net = tf.reshape(x, [-1, w.get_shape().as_list()[0]])
        net = tf.add(tf.matmul(net, w), b)
        if not is_out:
            net = tf.nn.relu(net)
            if self.mode == 'train':
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

        if self.regularizers is None:
            self.regularizers = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
        else :
            self.regularizers += tf.nn.l2_loss(W) + tf.nn.l2_loss(b)

        net = tf.nn.bias_add(net, b)
        net = tf.nn.relu(net)
        return net

    def pool_b(self, x, name, stride=2, k=2):

        maxpool_fn = tf.nn.max_pool if self.conv_dim == 2 else tf.nn.max_pool3d
        strides_list = [1, stride, stride, 1] if self.conv_dim == 2 else [1, stride, stride, stride, 1]
        k_list = [1, k, k, 1] if self.conv_dim == 2 else [1, k, k, k, 1]

        return maxpool_fn(x, ksize=k_list, strides=strides_list, padding='VALID', name=name)

    def madroob(self, lst):
        res = 1
        for i in lst:
            res *= i
        return res

    def build(self):

        if self.print_only is False:
            self.get_X()

        # print("\n\n")
        output_shape = {}
        output_data = {}

        for idx in self.chain_order:
            chain = self.chains[idx]
            parts = chain
            chain_idx = chain_id = idx

            input_shape = None
            input_data = None
            if idx == 'out':
                flat_data = []
                total_input_size = 0
                if len(output_data) > 1:
                    for k in output_data:
                        sz = int(self.madroob(output_shape[k]))
                        total_input_size += sz
                        flat_data.append(tf.reshape(output_data[k], [-1, sz]))
                    input_data = tf.concat(1, flat_data)
                    input_shape = np.array([total_input_size])
                else :
                    k = list(output_data.keys()).pop()
                    input_shape = output_shape[k]
                    input_data = output_data[k]

            else:
                input_shape = np.array(self.input_shape[chain_idx])
                input_data = self.X[chain_idx]


            output_shape[chain_idx] = None
            output_data[chain_idx] = None

            for idx, part in enumerate(parts):

                memory_size = {}
                params = {}

                if part['type'] == 'pool':
                    input_dim = input_shape[:-1]
                    input_chan = input_shape[-1]

                    shape_dim = part['shape']

                    output_shape[chain_idx] = np.array(input_dim) - np.array(shape_dim)
                    output_shape[chain_idx] = np.float16(output_shape[chain_idx])
                    output_shape[chain_idx] /= part['stride']
                    output_shape[chain_idx] += 1

                    params['stride'] = part['stride']
                    params['size'] = part['shape'][0]

                    output_shape[chain_idx] = np.append(output_shape[chain_idx], input_chan)
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
                        input_data = self.fc_b(input_data, w, b, None, part['name'], is_out=(idx ==len(parts) -1))

                    output_shape[chain_idx] = [o]

                if part['type'] == 'conv':
                    input_dim = input_shape[:-1]
                    input_chan = input_shape[-1]

                    shape_dim = part['shape']
                    shape_chan = part['count']

                    if part['maintain_spatial'] is True:
                        output_shape[chain_idx] = list(input_dim)
                    else :
                        output_shape[chain_idx] = np.array(input_dim) - np.array(shape_dim)
                        output_shape[chain_idx] = np.float16(output_shape[chain_idx])
                        output_shape[chain_idx] /= part['stride']
                        output_shape[chain_idx] += 1

                    output_shape[chain_idx] = np.append(output_shape[chain_idx], part['count'])

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

                output_data[chain_idx] = input_data
                input_memory_size = self.get_tensor_size(input_shape) * self.batch_size
                self.memory_footprint += input_memory_size
                # print("\t" * 3 + str(input_shape) + " - " + self.sizeof_fmt(input_memory_size))
                # print("\t" * 3 + "↓")
                # print("\t" * 3 + part['name'] + " - " + str(params) + " - " + str(memory_size))
                # print("\t" * 3 + "↓")
                input_shape = output_shape[chain_idx]

        # print("\t" * 3 + "Output")
        # print("\n\n")
        # print("Total size of the model " + self.sizeof_fmt(self.memory_footprint))
        if self.print_only is not True:
            return output_data['out']
        return self


    def conv(self, shape, count, stride, maintain_spatial=False):
        self.conv_counter += 1
        self.chains[self.active_chain].append({
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
        self.chains[self.active_chain].append({
            "name": "Pool" + str(self.conv_counter),
            "shape": shape,
            "stride": stride,
            "type": "pool"
        })
        return self

    def fc(self, n=None):
        self.conv_counter += 1

        self.chains[self.active_chain].append({
            "name": "FC" + str(self.conv_counter),
            "n": n,
            "type": "fc"
        })
        return self


