import json
import time
import pickle
import os
import random
import string

import numpy as np
import tensorflow as tf
from voi_queue import ROIQueue
from fc_calc import FCNetwork
from stat_helper import *

class NNModel:

    @staticmethod
    def gen_id(size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

    def __init__(self, network_arch, input_shape, num_classes, model_state=None):


        self.network_arch = json.loads(network_arch)
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.graph = None
        self.graph_scope = NNModel.gen_id()

        self.model = None
        self.X = None
        self.Y = None
        self.model_state_path = model_state

        self.network_builder = NetCalc(input_shape, num_classes, 32, False)

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            self.session = tf.Session(graph=self.graph)
            self.model = self.network_builder.build_from_config(self.network_arch)
            self.X = self.network_builder.get_X()
            self.Y = tf.placeholder(tf.float32, [None, self.num_classes], name="y")

    @staticmethod
    def from_json(self, dataset_filename, network_structure_json):
        pass

    @staticmethod
    def from_directory(self, directory):
        pass

    def test(self, input_data):
        with self.graph.as_default() as g:
            self._test(input_data)

    def _test(self, input_data):

        sess = self.session
        saver = tf.train.Saver()
        if self.model_state_path is not None:
            saver.restore(sess, self.model_state_path + "/model.ckpt")

        input_dict = {}
        for idx in input_data:
            input_dict[idx] = input_data[idx]

        result = sess.run([self.model], feed_dict=input_dict)


    def train(self, dataset_filename, angles=None, iterations=8e6, batch_size=32,
              drop_out=0.5, reg_power=5e-4, learning_rate=1e-5):
        with self.graph.as_default() as g:
            self._train(dataset_filename, angles=angles, iterations=iterations, batch_size=batch_size,
                drop_out=drop_out, reg_power=reg_power, learning_rate=learning_rate
            )

    def _train(self, dataset_filename, angles=None, iterations=8e6, batch_size=32,
              drop_out=0.5, reg_power=5e-4, learning_rate=1e-5):

        if angles is None:
            angles = {
                'train': None,
                'validate': None,
                'test': None
            }

        roi_queue = ROIQueue(
            dataset_filename,
            batch_size=batch_size,
            train_angles=angles['train'],
            validate_angles=angles['validate'],
            test_angles=angles['test']
        )
        roi_queue.start()

        stat_train_acc = []
        stat_test_acc = []
        stat_data_loss = []
        stat_learning_rate = []

        sm_stat_train_acc = []
        sm_stat_test_acc = []
        sm_stat_data_loss = []

        full_input_shape = {}
        for subset_id in self.input_shape:
            full_input_shape[subset_id] = [None]
            full_input_shape[subset_id].extend(self.input_shape[subset_id][:])

        sess = self.session
        saver = tf.train.Saver()

        if self.model_state_path is not None:
            saver.restore(sess, self.model_state_path)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.model, self.Y))
        cost += reg_power * self.network_builder.get_reg()
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        correct_pred = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        init = tf.initialize_all_variables()
        sess.run(init)

        model_meta_information = {
            "type": "convnet",
            "arch": self.network_arch,
            "date": time.time(),
            "iterations_req": iterations,
            "time_start": time.time(),
            "time_end": 0,
            "iterations_done": 0,
            "regularization_strength": reg_power,
            "drop_out_probability": drop_out,
            "batch_size": batch_size,
            "initial_learning_rate": 1e-5,
            "stats": {
                "train_accurecy": None,
                "test_accurecy": None,
                "train_loss": None
            },
            "result": None
        }
        directory_format = "{0}/convnet_{1}_{2}/"
        filename_format = "model.{0}"

        stat_step = 20
        display_step = 2000
        train_loss = 0
        step = 1
        try:
            while step * batch_size < iterations:
                batch_x, batch_y = roi_queue.get_next_batch('train', batch_size, flattened=False)
                train_dict = {
                    self.Y: batch_y
                }
                for idx in batch_x:
                    full_input_shape[idx][0] = batch_size
                    batch_x[idx].shape = full_input_shape[idx]
                    train_dict[self.X[idx]] = batch_x[idx]

                # Run optimization op (backprop)
                _ = sess.run([optimizer], feed_dict=train_dict)

                if step % stat_step == 0:
                    test_batch_x, test_batch_y = roi_queue.get_next_batch('test', 32, flattened=False)
                    test_dict = {
                        self.Y: batch_y
                    }
                    for idx in batch_x:
                        full_input_shape[idx][0] = 32
                        test_batch_x[idx].shape = full_input_shape[idx]
                        test_dict[self.X[idx]] = test_batch_x[idx]

                    train_acc, train_loss = sess.run([accuracy, cost], feed_dict=train_dict)
                    summary, test_acc = sess.run([accuracy], feed_dict=test_dict)
                    stat_data_loss.append([step * batch_size, train_loss])
                    stat_train_acc.append([step * batch_size, train_acc])
                    stat_test_acc.append([step * batch_size, test_acc])
                    stat_learning_rate.append([step * batch_size, learning_rate])

                if step % display_step == 0:
                    steps_count = int(display_step / stat_step)
                    tr_acc = int(get_mean(stat_train_acc, steps=steps_count) * 100)
                    ts_acc = int(get_mean(stat_test_acc, steps=steps_count) * 100)
                    sm_stat_train_acc.append([step * batch_size, tr_acc])
                    sm_stat_test_acc.append([step * batch_size, ts_acc])

                    progress = get_mean_diff(stat_train_acc, steps_count)

                    update_graph(stat_data_loss, sm_stat_train_acc, sm_stat_test_acc, stat_learning_rate)

                    prg = 0 if progress is None else progress
                    print("Iter " + str(step * batch_size) +
                          ", Loss = {:.6f}".format(train_loss) +
                          ", Training Accuracy = {:.3f}".format(tr_acc) +
                          ", Test Accuracy = {:.3f}".format(ts_acc)
                          )

                model_meta_information["stats"]["train_accurecy"] = sm_stat_train_acc
                model_meta_information["stats"]["test_accurecy"] = sm_stat_test_acc
                model_meta_information["stats"]["test_accurecy"] = stat_data_loss
                model_meta_information["result"] = None if len(sm_stat_test_acc) == 0 else sm_stat_test_acc[-1]
                model_meta_information["time_end"] = time.time()
                model_meta_information["iterations_done"] = (step * batch_size)
                step += 1

            if len(sm_stat_test_acc) != 0:
                timestr = time.strftime("%Y%m%d-%H%M%S")
                test_batch_x, test_batch_y = roi_queue.get_next_batch('test', 32, flattened=False)
                directory_name = directory_format.format('./models/', sm_stat_test_acc[-1], timestr)
                meta_filename = directory_name + filename_format.format("p")
                ckpt_filename = directory_name + filename_format.format("ckpt")
                test_X_filename = directory_name + "test_data_X.npy"
                test_Y_filename = directory_name + "test_data_Y.npy"
                os.makedirs(directory_name, mode=0o777, exist_ok=True)
                np.save(test_X_filename, test_batch_x)
                np.save(test_Y_filename, test_batch_y)
                saver.save(sess, ckpt_filename)
                pickle.dump(model_meta_information, open(meta_filename, "wb"))
            roi_queue.stop()

            print("Optimization Finished!")
        except KeyboardInterrupt:

            if len(sm_stat_test_acc) != 0:
                model_meta_information["time_end"] = time.time()
                model_meta_information["iterations_done"] = (step * batch_size)
                timestr = time.strftime("%Y%m%d-%H%M%S")
                test_batch_x, test_batch_y = roi_queue.get_next_batch('test', 32, flattened=False)
                directory_name = directory_format.format('./models/', sm_stat_test_acc[-1], timestr)
                meta_filename = directory_name + filename_format.format("p")
                ckpt_filename = directory_name + filename_format.format("ckpt")
                test_X_filename = directory_name + "test_data_X.npy"
                test_Y_filename = directory_name + "test_data_Y.npy"
                os.makedirs(directory_name, mode=0o777, exist_ok=True)
                np.save(test_X_filename, test_batch_x)
                np.save(test_Y_filename, test_batch_y)
                saver.save(sess, ckpt_filename)
                pickle.dump(model_meta_information, open(meta_filename, "wb"))

            roi_queue.stop()

            print("Optimization Terminated!")
