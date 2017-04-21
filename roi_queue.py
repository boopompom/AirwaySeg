from sklearn.decomposition import PCA
from voi import ROI
from queue import *

import os
import time
import math
import random
import binascii
import numpy as np
import threading
import pickle


class ROIQueue:


    def reload(self):

        self.fetch_times = []

        self.worker_threads = {
            'train': [],
            'validate': [],
            'test': []
        }

        self.data_templates = {
            'train': [],
            'validate': [],
            'test': []
        }

        self.q = {
            'train': Queue(maxsize=self.max_size),
            'validate': Queue(maxsize=self.max_size),
            'test': Queue(maxsize=self.max_size)
        }

        self.make_train_test_per_dataset()

    def __init__(self,
                 dataset_file,
                 random_seed=None,
                 train_angles=None,
                 validate_angles=None,
                 test_angles=None,
                 batch_size=32,
                 validate_pct=0.1,
                 test_pct=0.1,
                 engine=None):

        self.input_path = dataset_file

        self.angles = {
            'train': list(range(0, 360, 10)) if train_angles is None else train_angles,
            'validate': list(range(0, 360, 90)) if validate_angles is None else validate_angles,
            'test': list(range(0, 360, 90)) if test_angles is None else test_angles,
        }
        self.worker_thread_count = 4
        self.max_size = batch_size * 4
        self.validate_pct = validate_pct
        self.test_pct = test_pct
        self.roi_list = pickle.load(open(dataset_file, "rb"))
        self.class_counter = {}
        self.std = 1
        self.mean = 0
        
        if random_seed is not None:
            self.random_seed = random_seed
        else:
            try:
                self.random_seed = int(binascii.hexlify(os.urandom(4)), 16)
            except NotImplementedError:
                import time
                self.random_seed = int(time.time() * 256)

        random.seed(self.random_seed)

        for roi in self.roi_list:
            if roi.roi_class not in self.class_counter:
                self.class_counter[roi.roi_class] = 0
            self.class_counter[roi.roi_class] += 1

        self.engine = engine

        # These variables will be set by reload()
        self.fetch_times = None
        self.worker_threads = None
        self.data_templates = None
        self.q = None

        self.reload()

    def get_roi_list_by_image(self):
        dic = {}
        for roi in self.roi_list:

            if roi.image_id not in dic:
                dic[roi.image_id] = []
            dic[roi.image_id].append(roi)
        return dic

    def __enter__(self):
        self.start()
        return self.get()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def export(self, dataset_type, filename):
        roi_list = self.data_templates[dataset_type]
        pickle.dump(roi_list, open(filename, 'wb'))



    def get_dataset_resolution_stats(self):

        output = {}
        for class_id in self.class_counter:
            output[class_id] = {}
            for dataset_type in self.data_templates:
                output[class_id][dataset_type] = {
                    '0.5': 0,
                    '0.625': 0,
                    '1.0': 0,
                    '1.25': 0
                }

        for dataset_type in self.data_templates:
            ds = self.data_templates[dataset_type]
            for roi in ds:
                z = str(roi.image_z_spacing)
                output[roi.roi_class][dataset_type][z] += 1

        return output, self.random_seed

    def make_train_test_per_dataset(self):
        """Breaks dataset into train, validate, test and doesn't care about class distribution"""
        dataset_counts = {
            'train': 0,
            'validate': int(len(self.roi_list) * self.validate_pct),
            'test': int(len(self.roi_list) * self.test_pct),
        }
        dataset_counts['train'] = len(self.roi_list) - (dataset_counts['validate'] + dataset_counts['test'])

        random.shuffle(self.roi_list)


        for roi in self.roi_list:

            cls = roi.roi_class

            dataset_type = None
            if dataset_counts['test'] > 0 and roi.has_intersection is False:
                dataset_counts['test'] -= 1
                dataset_type = 'test'
            elif dataset_counts['validate'] > 0 and roi.has_intersection is False:
                dataset_counts['validate'] -= 1
                dataset_type = 'validate'
            else:
                dataset_counts['train'] -= 1
                dataset_type = 'train'

            self.data_templates[dataset_type].append(roi)

        # Calculate dataset mean and std only from training data
        X = None
        for roi in self.data_templates['train']:
            x = np.array([roi.get_cube(rotation=None, as_nd=True)])
            if X is None:
                X = np.array(x)
            else:
                X = np.concatenate((X, x), axis=0)
        X = np.array(X)
        self.std = np.std(X)
        self.mean = np.mean(X)

    def make_train_test_per_class(self):
        """Breaks dataset into train, validate, test enforcing class distribution matching dataset"""
        class_counts = {
            'test': {},
            'train': {},
            'validate': {}
        }

        for class_id in self.class_counter:
            test_count = math.floor(self.class_counter[class_id] * self.test_pct)
            validate_count = math.floor(self.class_counter[class_id] * self.validate_pct)
            train_count = math.floor(self.class_counter[class_id] - (test_count + validate_count))
            class_counts['validate'][class_id] = validate_count
            class_counts['test'][class_id] = test_count
            class_counts['train'][class_id] = train_count

        random.shuffle(self.roi_list)

        image_usage_map = {}

        for roi in self.roi_list:

            cls = roi.roi_class
            # Mode is decided based on class balance, we start by fulfilling test requirements
            # then we proceed to fill training data

            dataset_type = None
            if class_counts['test'][cls] > 0 and roi.has_intersection is False:
                class_counts['test'][cls] -= 1
                dataset_type = 'test'
            elif class_counts['validate'][cls] > 0 and roi.has_intersection is False:
                class_counts['validate'][cls] -= 1
                dataset_type = 'validate'
            else:
                class_counts['train'][cls] -= 1
                dataset_type = 'train'

            self.data_templates[dataset_type].append(roi)

        for idx_1 in class_counts:
            if idx_1 == 'train':
                continue
            for idx_2 in class_counts[idx_1]:
                if class_counts[idx_1][idx_2] != 0:
                    raise ValueError("Not enough entries for class {0} on dataset {1}".format(idx_2, idx_1))

    def get_all(self, dataset):

        selected_dataset = self.data_templates[dataset]

        batch_X, batch_Y, batch_roi, batch_rotation = {'image':[]}, [], [], []

        for roi in selected_dataset:
            X, Y = roi.get_set(rotation=None)
            X -= self.mean
            X /= self.std
            s = list(X.shape)
            s.append(1)
            X.shape = s
            batch_X['image'].append(X)
            batch_Y.append(Y)
            batch_roi.append(roi)
            batch_rotation.append(None)

        batch_X['image'] = np.array(batch_X['image'])
        batch_Y = np.array(batch_Y)
        return batch_X, batch_Y, batch_roi, batch_rotation

    def get(self):

        dataset_type, batch_size, flattened, normalized = yield
        while not self.shutdown_signal:

            batch_X = {
                'image': []
            }
            batch_Y = []

            batch_roi = []
            batch_rotation = []
            for i in range(batch_size):
                X, Y, roi, rotation = self.q[dataset_type].get()
                batch_Y.append(Y)
                batch_roi.append(roi)
                batch_rotation.append(rotation)
                for idx in X:
                    if idx not in batch_X:
                        batch_X[idx] = []
                    batch_X[idx].append(X[idx])

            batch_Y = np.array(batch_Y)
            for idx in batch_X:
                batch_X[idx] = np.array(batch_X[idx])

            if normalized:
                batch_X['image'] -= self.mean
                batch_X['image'] /= self.std

            s = list(batch_X['image'].shape)
            s.append(1)
            batch_X['image'].shape = s

            if flattened:
                batch_X['image'].shape = (batch_X['image'].shape[0], -1)

            self.q[dataset_type].task_done()

            dataset_type, batch_size, flattened, normalized = yield batch_X, batch_Y, batch_roi, batch_rotation


    def start(self):

        self.shutdown_signal = False

        self.reload()

        for idx in self.worker_threads:
            if len(self.data_templates[idx]) == 0:
                continue
            self.worker_threads[idx] = []
            for c in range(self.worker_thread_count):
                t = threading.Thread(target=self.worker, args=(idx, ))
                self.worker_threads[idx].append(t)
                t.daemon = True
                t.start()

    def stop(self):
        self.shutdown_signal = True
        for idx in self.worker_threads:
            if self.q[idx].full():
                for i in range(self.worker_thread_count):
                    self.q[idx].get_nowait()

            for t in self.worker_threads[idx]:
                t.join()

    def worker(self, mode):

        rotations = self.angles

        while not self.shutdown_signal:

            selected_dataset = self.data_templates[mode]
            selected_range = rotations[mode]
            selected_q = self.q[mode]

            rotation_list = []
            for idx in range(0, 3):
                rotation_list.append(random.choice(selected_range))

            selected_roi = random.choice(selected_dataset)
            X, Y = selected_roi.get_set(rotation=rotation_list)
            full_x = {
                'image': X
            }

            if self.engine is not None:
                full_x = self.engine(full_x, selected_roi)

            item = (full_x, Y, selected_roi, rotation_list)
            selected_q.put(item)
