import sys
import json
import time
import pickle
import os
import random
import string

import numpy as np
import tensorflow as tf
from voi_queue import VOIQueue
from net_calc import NetCalc
from fc_net import FullyConvNetwork
from stat_helper import *

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score



class NNModel:

    @staticmethod
    def gen_id(size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))


    def print_confusion_matrix(plabels, tlabels):
        """
            functions print the confusion matrix for the different classes
            to find the error...

            Input:
            -----------
            plabels: predicted labels for the classes...
            tlabels: true labels for the classes

            code from: http://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix-in-python
        """
        plabels = pd.Series(plabels)
        tlabels = pd.Series(tlabels)

        # draw a cross tabulation...
        df_confusion = pd.crosstab(tlabels, plabels, rownames=['Actual'], colnames=['Predicted'], margins=True)

        # print df_confusion
        return df_confusion

    @staticmethod
    def performance_metrics(labels, y_pred, y_org):
        pass

    @staticmethod
    def confusion_matrix(class_labels_a, class_labels_p, y_actual, y_predicted, class_map_a=None, class_map_p=None):
        class_count_a = len(class_labels_a)
        class_count_p = len(class_labels_p)
        total_examples = len(y_predicted)
        row_count = class_count_a + 1
        col_count = class_count_p + 1


        matrix = [[0 for i in range(row_count)] for j in range(col_count)]
        for idx, val in enumerate(y_actual):
            a_i = int(y_actual[idx]) if class_map_a is None else int(class_map_a[int(y_actual[idx])])
            p_i = int(y_predicted[idx]) if class_map_p is None else int(class_map_p[int(y_predicted[idx])])
            matrix[a_i][p_i] += 1
            matrix[a_i][-1] += 1
            matrix[-1][p_i] += 1
            matrix[-1][-1] += 1

        return matrix

    def get_raw_predictions(self, x, y, roi):

        feed_dict = {
            self.Y: y
        }
        for idx in x:
            feed_dict[self.X[idx]] = x[idx]

        prediction = tf.argmax(self.model, 1)
        labels = prediction.eval(feed_dict, session=self.session)


        y_actual = list(np.where(y[:] == 1)[1])
        y_predicted = list(labels)

        # Legacy roi objects don't have a class_map object
        class_map = None

        if not hasattr(roi[0], 'class_map'):
            # Assume that in case of a missing class map, if it is a 2 class problem we use this map
            if max(y_actual) + 1 == 2:
                class_map = {
                    '_Normal_ipf': 0,
                    '_Emphysema_ipf': 1,
                    '_Bronchovascular_ipf': 2,
                    '_Ground Glass_ipf': 3,
                    '_Ground Glass - Reticular_ipf': 4,
                    '_Honeycomb_ipf': 5,
                    '_Mix_ipf': 6
                }
        else:
            class_map = roi[0].class_map

        y_original = None
        if class_map is not None:
            y_original = [class_map[i.roi_org_class] for i in roi]

        # print("Stuff", y_actual, y_predicted, y_original)

        y_actual_labels = list(range(max(y_actual) + 1))

        y_original_labels = list(range(len(class_map))) if class_map is not None else y_actual_labels

        return {
            'actual': y_actual,
            'predicted': y_predicted,
            'original': y_original,
            'actual_labels': y_actual_labels,
            'original_labels': y_original_labels,
            'class_map': class_map
        }


    def evaluate_model(self, x, y, roi):

        # Legacy roi objects don't have a class_map object
        class_map = None
        if not hasattr(roi[0], 'class_map'):
            class_map = {
                '_Normal_ipf': 0,
                '_Emphysema_ipf': 1,
                '_Bronchovascular_ipf': 2,
                '_Ground Glass_ipf': 3,
                '_Ground Glass - Reticular_ipf': 4,
                '_Honeycomb_ipf': 5,
                '_Mix_ipf': 6
            }
        else:
            class_map = roi[0].class_map


        class_count = len(y[0])
        feed_dict = {
            self.Y: y
        }
        for idx in x:
            feed_dict[self.X[idx]] = x[idx]

        predictions = self.session.run([self.correct_pred], feed_dict=feed_dict)
        prediction = tf.argmax(self.model, 1)
        labels = prediction.eval(feed_dict, session=self.session)
        acc = self.accuracy.eval(feed_dict, session=self.session)

        y_actual = np.where(y[:] == 1)[1]
        y_predicted = labels
        y_original = None
        if class_map is not None:
            y_original = [class_map[i.roi_org_class] for i in roi]

        # print("Stuff", y_actual, y_predicted, y_original)

        y_actual_labels = list(range(max(y_actual) + 1))
        print(y_actual_labels)
        y_original_labels = list(range(len(class_map))) if class_map is not None else y_actual_labels

        # print("Error Checking:")
        # print(NNModel.confusion_matrix(class_map, y_actual_labels, y_actual_labels, y_actual, y_actual))
        # print(NNModel.confusion_matrix(class_map, y_actual_labels, y_original_labels, y_original, y_actual))

        return {
            'confusion_matrix': NNModel.confusion_matrix(y_actual_labels, y_actual_labels,   y_actual, y_predicted),
            'confusion_matrix_original': NNModel.confusion_matrix(y_actual_labels, y_original_labels, y_original, y_predicted)
        }



        return
        false_negative = {}
        original_classes = []
        if class_count == 2:
            # Counts false positives
            for idx, label in enumerate(labels):
                if labels[idx] == False:
                    if y[idx][0] == 0:
                        # false negative
                        if roi[idx].roi_org_class not in false_negative:
                            false_negative[roi[idx].roi_org_class] = 0
                        false_negative[roi[idx].roi_org_class] += 1

            original_class_count = {}
            for idx, label in enumerate(labels):
                if roi[idx].roi_org_class not in original_class_count:
                    original_class_count[roi[idx].roi_org_class] = 0
                original_class_count[roi[idx].roi_org_class] += 1

            original_classes = list(original_class_count.keys())


        org_confusion_matrix = [[0 for i in range(class_count)] for j in range(len(original_classes))]

        print("Test set accuracy", acc, "\n")
        # NNModel.confusionMatrix("Partial Confusion matrix", y, predictions[0], False)  # Partial confusion Matrix
        scores = NNModel.confusion_matrix(y_actual, y_predicted, y_original)  # complete confusion Matrix

        #Fixme: Mustn't be hard coded here
        predicted = scores['predicted']
        for idx, val in enumerate(predicted):
            a_i = original_y[idx]
            p_i = predicted[idx]
            org_confusion_matrix[a_i][p_i] += 1
            org_confusion_matrix[-1][p_i] += 1
            org_confusion_matrix[a_i][-1] += 1
            org_confusion_matrix[-1][-1] += 1

        scores['org_confusion_matrix'] = None
        if class_count == 2:
            scores['org_confusion_matrix'] = org_confusion_matrix
            print(org_confusion_matrix)
        return scores

    def build_from_saved_state(self, model_state_path):
        model_metadata_path = os.path.join(model_state_path, 'model.p')
        model_meta_data = pickle.load(open(model_metadata_path, 'rb'))
        # print('input_shape' not in model_meta_data)
        self.network_arch = model_meta_data['arch']
        self.mean = model_meta_data['mean']
        self.std = model_meta_data['std']
        self.model_path = model_state_path
        self.model_state_path = os.path.join(model_state_path, 'model.ckpt')
        self.model_meta_information = model_meta_data

        if 'name' not in model_meta_data:
            self.name = "Unknown"
        else:
            self.name = model_meta_data['name']

        if 'input_shape' not in model_meta_data:
            self.input_shape = {'image': [15, 15, 23, 1]}
        else:
            self.input_shape = model_meta_data['input_shape']

        if 'num_classes' not in model_meta_data:
            self.num_classes = 6
        else:
            self.num_classes = model_meta_data['num_classes']


    def __init__(self, name=None, mode='train', model_state_path=None):

        self.name = name
        self.model_path = None
        self.mean = 0
        self.std = 1

        self.model_meta_information = None

        self.graph = None
        self.graph_scope = NNModel.gen_id()

        self.model = None
        self.X = None
        self.Y = None
        self.reg = None
        self.correct_pred = None
        self.accuracy = None

        self.session = None
        self.train_params = None

        self.graph = tf.Graph()


    @staticmethod
    def from_directory(self, directory):
        pass

    def get_raw_eval(self):
        with self.graph.as_default() as g:
            with tf.Session(graph=self.graph) as self.session:
                return self._get_raw_eval()

    def _get_raw_eval(self):
        self.model = self.network_builder.build_from_config(self.network_arch)
        self.X = self.network_builder.get_X()
        self.Y = tf.placeholder(tf.float32, [None, self.num_classes], name="y")
        self.correct_pred = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred , tf.float32))

        saver = tf.train.Saver()
        saver.restore(self.session, self.model_state_path)

        test_data_filename = os.path.join(self.model_path, 'test_rois.p')

        # These ROI templates for the test set only, so we build one dataset with not splits and then use the train data
        roi_queue = ROIQueue(
            test_data_filename,
            batch_size=32,
            train_angles=[0],
            validate_angles=[0],
            test_angles=[0],
            validate_pct=0.0,
            test_pct=0.0
        )

        # This ROIQueue is constructed from pure test data, so we just put all examples into "train" and retrieve them
        test_batch_x, test_batch_y, roi, rotation = roi_queue.get_all('train')
        return self.get_raw_predictions(test_batch_x, test_batch_y, roi)

    def eval(self):
        with self.graph.as_default() as g:
            with tf.Session(graph=self.graph) as self.session:
                return self._eval()

    def _eval(self):
        self.model = self.network_builder.build_from_config(self.network_arch)
        self.X = self.network_builder.get_X()
        self.Y = tf.placeholder(tf.float32, [None, self.num_classes], name="y")
        self.correct_pred = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred , tf.float32))

        saver = tf.train.Saver()
        saver.restore(self.session, self.model_state_path)

        test_data_filename = os.path.join(self.model_path, 'test_rois.p')

        # These ROI templates for the test set only, so we build one dataset with not splits and then use the train data
        roi_queue = ROIQueue(
            test_data_filename,
            batch_size=32,
            train_angles=[0],
            validate_angles=[0],
            test_angles=[0],
            validate_pct=0.0,
            test_pct=0.0
        )
        test_batch_x, test_batch_y, roi, rotation = roi_queue.get_all('train')
        return self.evaluate_model(test_batch_x, test_batch_y, roi)

    def test(self, input_data):
        with self.graph.as_default() as g:
            with tf.Session(graph=self.graph) as self.session:
                self.model = self.network_builder.build_from_config(self.network_arch)
                self.X = self.network_builder.get_X()
                self.Y = tf.placeholder(tf.float32, [None, self.num_classes], name="y")
                self.correct_pred = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.Y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred , tf.float32))
                return self._test(input_data)

    def _test(self, input_data):

        sess = self.session

        if self.model_state_path is not None:
            saver = tf.train.Saver()
            saver.restore(sess, self.model_state_path)

        input_dict = {}
        for idx in input_data:
            X = np.float32(input_data[idx])
            if idx == 'image':
                X -= self.mean
                X /= self.std

            input_dict[self.X[idx]] = X

        r = sess.run([self.model], feed_dict=input_dict)
        r = r[0]
        # normalization, Thanks Stanford C231n
        m = np.max(r, axis=1)
        m.shape = (m.shape[0], 1)
        r -= m
        s = np.sum(np.exp(r), axis=1)
        s.shape = (s.shape[0], 1)
        r = np.exp(r) / s
        r = np.int16(r * 100)
        return r


    def _save(self, saver, sess, roi_q):

        directory_format = "{0}/{1}_{2}_{3}/"
        filename_format = "model.{0}"

        if len(self.model_meta_information["stats"]["validation_accurecy"]) == 0:
            return

        latest_acc = self.model_meta_information["stats"]["validation_accurecy"][-1]
        timestr = time.strftime("%Y%m%d-%H%M%S")
        directory_name = os.path.abspath(directory_format.format('./models/', self.name, latest_acc, timestr)) + '/'
        os.makedirs(directory_name, mode=0o777, exist_ok=True)

        meta_filename = directory_name + filename_format.format("p")
        ckpt_filename = directory_name + filename_format.format("ckpt")
        test_data_filename = directory_name + "test_rois.p"

        roi_q.export('test', test_data_filename)
        saver.save(sess, ckpt_filename)
        pickle.dump(self.model_meta_information, open(meta_filename, "wb"))

    def train(self, dataset_filename, cv_folds=10, angles=None, iterations=8e6, batch_size=32, steps_to_downscale=2,
              drop_out=0.5, reg_power=5e-4, learning_rate=1e-5):

        with self.graph.as_default() as g:
            with tf.Session(graph=self.graph) as self.session:
                m = FullyConvNetwork().get()
                self.X = m['X']
                self.Y = m['Y']
                self.model = m['model']
                self.reg = m['reg']

                if angles is None:
                    angles = {
                        'train': None,
                        'validate': None,
                        'test': None
                    }

                voi_queue = VOIQueue(
                    dataset_filename,
                    random_seed=1992,
                    batch_size=batch_size,
                )

                self._train(
                    voi_queue,
                    iterations=iterations,
                    batch_size=batch_size,
                    steps_to_downscale=steps_to_downscale,
                    drop_out=drop_out,
                    reg_power=reg_power,
                    learning_rate=learning_rate,
                )

    def _train(self, voi_queue, iterations=8e6, batch_size=32, steps_to_downscale=2,
               drop_out=0.5, reg_power=5e-4, learning_rate=1e-5):

        no_improvement_limit = steps_to_downscale
        validation_batch_size = 16

        steps_with_no_improvement = 0
        highest_validation_acc = 0

        stat_train_acc = []
        stat_vd_acc = []
        stat_data_loss = []
        stat_vd_loss = []
        stat_learning_rate = []

        sm_stat_train_acc = []
        sm_stat_vd_acc = []

        full_input_shape = {
            'global': [None, 53, 53, 53, 1],
            'local':  [None, 33, 33, 33, 1]
        }

        sess = self.session
        saver = tf.train.Saver()

        learning_rate_var = tf.placeholder(tf.float32, shape=[])
        #if self.model_state_path is not None:
        #    saver.restore(sess, self.model_state_path)

        res = tf.reshape(self.model, [batch_size, 32])
        raw_cost = tf.nn.softmax_cross_entropy_with_logits(logits=res, labels=self.Y)
        cost = tf.reduce_mean(raw_cost)
        cost += reg_power * self.reg
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_var).minimize(cost)

        correct_pred = self.correct_pred = tf.equal(tf.argmax(res, dimension=1), tf.argmax(self.Y, dimension=1))
        accuracy = self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        raw_scores = tf.argmax(res, dimension=1)
        raw_correct = tf.argmax(self.Y, 1)
        raw_accuracy = tf.cast(correct_pred, tf.float32)

        init = tf.global_variables_initializer()
        sess.run(init)



        self.model_meta_information = {
            "name": self.name,
            "type": "convnet",
            "date": time.time(),
            "mean": voi_queue.mean,
            "std": voi_queue.std,
            "steps_to_downscale": steps_to_downscale,
            "iterations_req": iterations,
            "time_start": time.time(),
            "time_end": 0,
            "iterations_done": 0,
            "regularization_strength": reg_power,
            "drop_out_probability": drop_out,
            "batch_size": batch_size,
            "initial_learning_rate": learning_rate,
            "stats": {
                "train_accurecy": None,
                "validation_accurecy": None,
                "train_loss": None,
                "validation_loss" : None,
                "learning_rate": None
            },
            "result": None
        }


        writer = tf.summary.FileWriter("./logs/tf_logs/")
        writer.add_graph(sess.graph)
        writer.flush()
        writer.close()

        stat_step = 20
        display_step = 100
        train_loss = 0
        step = 1
        current_learning_rate = learning_rate
        # logger = tf.train.SummaryWriter('BadLog', sess.graph)
        with voi_queue as roi_generator:
            next(roi_generator)
            try:
                while step * batch_size < iterations:
                    step += 1
                    batch_x, batch_y, roi, rotation = roi_generator.send(('train', batch_size, False, True))
                    train_dict = {
                        self.Y: batch_y
                    }
                    for idx in batch_x:
                        full_input_shape[idx][0] = batch_size
                        batch_x[idx].shape = full_input_shape[idx]
                        train_dict[self.X[idx]] = batch_x[idx]

                    train_dict[learning_rate_var] = current_learning_rate


                    # Run optimization op (backprop)
                    _ = sess.run([optimizer], feed_dict=train_dict)

                    if step % stat_step == 0:
                        validation_batch_x, validation_batch_y, roi, rotation = roi_generator.send(('validate', validation_batch_size, False, True))
                        validation_dict = {
                            self.Y: validation_batch_y
                        }
                        for idx in batch_x:
                            full_input_shape[idx][0] = validation_batch_size
                            validation_batch_x[idx].shape = full_input_shape[idx]
                            validation_dict[self.X[idx]] = validation_batch_x[idx]

                        train_acc, train_loss = sess.run([accuracy, cost], feed_dict=train_dict)
                        raw_score = sess.run([self.model], feed_dict=train_dict)
                        cst, m, s, c, a = sess.run([raw_cost, res, raw_scores, raw_correct, raw_accuracy], feed_dict=train_dict)


                        validation_acc, validation_loss = sess.run([accuracy, cost], feed_dict=validation_dict)
                        stat_data_loss.append([step * batch_size, train_loss])
                        stat_train_acc.append([step * batch_size, train_acc])
                        stat_vd_acc.append([step * batch_size, validation_acc])
                        stat_vd_loss.append([step * batch_size, validation_loss])
                        stat_learning_rate.append([step * batch_size, current_learning_rate])

                        self.model_meta_information["stats"]["train_accurecy"] = sm_stat_train_acc
                        self.model_meta_information["stats"]["validation_accurecy"] = sm_stat_vd_acc
                        self.model_meta_information["stats"]["train_loss"] = stat_data_loss
                        self.model_meta_information["stats"]["validation_loss"] = stat_vd_loss
                        self.model_meta_information["stats"]["learning_rate"] = stat_learning_rate
                        self.model_meta_information["result"] = None if len(sm_stat_vd_acc) == 0 else sm_stat_vd_acc[-1]
                        self.model_meta_information["time_end"] = time.time()
                        self.model_meta_information["iterations_done"] = (step * batch_size)

                    if step % display_step == 0:
                        steps_count = int(display_step / stat_step)
                        tr_acc = int(get_mean(stat_train_acc, steps=steps_count) * 100)
                        vd_acc = int(get_mean(stat_vd_acc, steps=steps_count) * 100)
                        sm_stat_train_acc.append([step * batch_size, tr_acc])
                        sm_stat_vd_acc.append([step * batch_size, vd_acc])

                        itr = step * batch_size
                        if itr >= 1e5:
                            if vd_acc > highest_validation_acc:
                                highest_validation_acc = vd_acc
                                steps_with_no_improvement = 0
                            else:
                                steps_with_no_improvement += 1


                        update_graph(stat_data_loss, sm_stat_train_acc, sm_stat_vd_acc, stat_vd_loss, stat_learning_rate)


                        print("Iter " + str(step * batch_size) +
                              ", Loss = {:.6f}".format(train_loss) +
                              ", Training Accuracy = {:.3f}".format(tr_acc) +
                              ", Validation Accuracy = {:.3f}".format(vd_acc)
                        )
                        # Break if no improvement in the last disp
                        if steps_with_no_improvement > no_improvement_limit:
                            current_learning_rate /= 10
                            print("No Improvement for {0} cycles, scaling down LR to {1}".format(no_improvement_limit, current_learning_rate))

                            if current_learning_rate < 1e-10:
                                print("No Improvement for {0} cycles, terminating".format(no_improvement_limit))
                                raise KeyboardInterrupt

                print("Optimization Finished!")
            except KeyboardInterrupt:
                print("Optimization Terminated!")
            finally:
                pass
                #test_batch_x, test_batch_y, roi, rotation = voi_queue.get_all('test')
                # test_batch_x, test_batch_y, roi, rotation = roi_generator.send(('test', 1000, False, True))
                #self.evaluate_model(test_batch_x, test_batch_y, roi)
                #self._save(saver, sess, voi_queue)
                # logger.close()



