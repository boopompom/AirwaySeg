from queue import *
from glob import glob
from voi import VOI

import json
import numpy as np
import SimpleITK as sitk


import os
import time
import math
import random
import binascii
import threading
import pickle


class VOIQueue:


    def reload(self):

        self.fetch_times = []

        self.worker_threads = {
            'train': [],
            'validate': [],
            'test': []
        }

        self.files = {
            'train': [],
            'validate': [],
            'test': []
        }

        self.available_files = {
            'train': [],
            'validate': [],
            'test': []
        }

        self.epoch_count = {
            'train': 0,
            'validate': 0,
            'test': 0
        }

        self.data_templates = {
            'train': [],
            'validate': [],
            'test': []
        }

        self.active_images = {
            'train': {},
            'validate': {},
            'test': {}
        }

        self.q = {
            'train': Queue(maxsize=self.max_size),
            'validate': Queue(maxsize=self.max_size),
            'test': Queue(maxsize=self.max_size)
        }

        self.make_train_test_per_dataset()

    def __init__(self, dataset_path, epochs_per_dataset=5, random_seed=None, train_angles=None, batch_size=32, validate_count=2, test_count=2):

        self.input_path = dataset_path
        self.datasets_per_batch = 5

        self.angles = {
            'train': list(range(0, 360, 10)) if train_angles is None else train_angles,
            'validate': [0],
            'test': [0]
        }

        self.worker_thread_count = 4
        self.max_size = batch_size * 24
        self.validate_count = validate_count

        self.counts = {
            'train': float("inf"),
            'validate': validate_count,
            'test': test_count
        }

        self.epochs_per_dataset = {
            'train': epochs_per_dataset,
            'validate': -1,
            'test': -1
        }

        self.class_counter = {}
        self.std = 1
        self.mean = 0

        # Randomize random seed
        if random_seed is not None:
            self.random_seed = random_seed
        else:
            try:
                self.random_seed = int(binascii.hexlify(os.urandom(4)), 16)
            except NotImplementedError:
                import time
                self.random_seed = int(time.time() * 256)
        random.seed(self.random_seed)

        # These variables will be set by reload()
        self.fetch_times = None
        self.worker_threads = None
        self.data_templates = None
        self.q = None
        self.files = None
        self.available_files = None
        self.epoch_count = None
        self.active_files = None
        self.active_images = None

        self.fileset_lock = threading.Lock()
        self.worker_sema = {
            'train': threading.Semaphore(self.worker_thread_count),
            'validate':  threading.Semaphore(self.worker_thread_count),
            'test':  threading.Semaphore(self.worker_thread_count)
        }
        self.shutdown_signal = False

        self.reload()

    def __enter__(self):
        self.start()
        return self.get()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def export(self, dataset_type, filename):
        roi_list = self.data_templates[dataset_type]
        pickle.dump(roi_list, open(filename, 'wb'))



    def make_train_test_per_dataset(self):

        """Breaks dataset into train, validate, test and doesn't care about class distribution"""
        files = glob(os.path.relpath("{0}/*.json".format(self.input_path)))

        random.shuffle(files)

        offset = 0
        self.files['test'] = files[offset : offset + self.counts['test']]
        offset += self.counts['test']
        self.files['validate'] = files[offset : offset + self.counts['validate']]
        offset += self.counts['validate']
        self.files['train'] = files[offset:]

        self.available_files['train'] = self.files['train'][:]
        self.available_files['validate'] = self.files['validate'][:]
        self.available_files['test'] = self.files['test'][:]

        self.load_next_dataset('train')
        self.load_next_dataset('validate')
        self.load_next_dataset('test')

        self.std = 1
        self.mean = 0

    def load_next_dataset(self, mode):
        with self.fileset_lock:
            try:
                # Wait for all threads to finish working with datasets and block them
                for i in range(0, self.worker_thread_count):
                    self.worker_sema[mode].acquire()

                limit = min(self.counts[mode], self.datasets_per_batch)
                if len(self.available_files[mode]) < limit:
                    self.available_files[mode] = self.files[mode][:]

                self.data_templates[mode] = []
                self.active_images[mode] = {}
                for i in range(0, limit):
                    filename = self.available_files[mode].pop()
                    self.load_dataset(mode, filename)
                self.epoch_count[mode] = 0
            finally:
                # Allow worker to resume
                for i in range(0, self.worker_thread_count):
                    self.worker_sema[mode].release()

    def load_dataset(self, mode, filename):
        modes = {
            'global': [53, 53, 53],
            'local': [33, 33, 33]
        }
        voi_list = []
        with open(filename) as data_file:
            data = json.load(data_file)

            dataset_id = data['id']
            dicom_path = data['dicom_path']

            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
            reader.SetFileNames(dicom_names)

            self.active_images[mode][dataset_id] = reader.Execute()

            for voi_data in data['vois']:
                self.data_templates[mode].append(VOI(dataset_id, data['dataset_path'], dicom_path, voi_data, modes=modes))


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
                'local': [],
                'global': []
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
                self.q[dataset_type].task_done()

            batch_Y = np.array(batch_Y)
            for idx in batch_X:
                batch_X[idx] = np.array(batch_X[idx])

            s = list(batch_X['local'].shape)
            s.append(1)
            batch_X['local'].shape = s

            s = list(batch_X['global'].shape)
            s.append(1)
            batch_X['global'].shape = s

            if flattened:
                batch_X['global'].shape = (batch_X['global'].shape[0], -1)
                batch_X['local'].shape = (batch_X['local'].shape[0], -1)

            dataset_type, batch_size, flattened, normalized = yield batch_X, batch_Y, batch_roi, batch_rotation


    def start(self):

        self.shutdown_signal = False

        # self.reload()

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
            time.sleep(0.0001)

            if self.epochs_per_dataset[mode] != -1 and self.epoch_count[mode] > self.epochs_per_dataset[mode]:
                self.load_next_dataset(mode)

            try:
                self.worker_sema[mode].acquire()

                selected_dataset = self.data_templates[mode]
                selected_range = rotations[mode]
                selected_q = self.q[mode]

                rotation_list = []
                for idx in range(0, 3):
                    rotation_list.append(random.choice(selected_range))

                voi = random.choice(selected_dataset)
                voi_image = self.active_images[mode][voi.image_id]
                cube = voi.get_cube(voi_image, 'global', rotation=rotation_list)


                dim_local = 33
                dim_global = cube.shape[0]

                start_offset = int((dim_global - dim_local) / 2)
                end_offset = start_offset + dim_local

                X = {
                    'global': cube,
                    'local' : cube[start_offset:end_offset, start_offset:end_offset, start_offset:end_offset]
                }
                Y = voi.y

                item = (X, Y, voi, rotation_list)

                selected_q.put(item)

                self.epoch_count[mode] += 1 / len(selected_dataset)

            finally:
                self.worker_sema[mode].release()

