from queue import *

import SimpleITK as sitk
import time
import random
from math import *
import numpy as np
import threading
import pickle
from image_helper import ImageHelper

class CubeQueue:

    def __init__(self, image_filename, model_object, input_cube_size=None):

        self.shutdown_signal = False
        self.worker_threads = []
        self.worker_thread_count = 1

        self.requires_padding = False
        self.model_cube_size = np.array([15, 15, 23])
        if input_cube_size is None:
            self.input_cube_size = self.model_cube_size
        else :
            self.input_cube_size = np.array(input_cube_size)

        for i in range(3):
            if self.input_cube_size[i] != self.model_cube_size[i]:
                self.requires_padding = True
            if self.input_cube_size[i] > self.model_cube_size[i]:
                raise ValueError("Input Cube size cannot be larger than model cube size " + str(self.model_cube_size))

        self.q = None
        self.spacing_factor, self.image = ImageHelper.read_image(image_filename, correct_spacing=True)
        self.org_ndarray = sitk.GetArrayFromImage(self.image)
        ndarray = np.float32(np.swapaxes(self.org_ndarray, 0, 2))
        ndarray -= np.mean(ndarray, axis=0)
        ndarray /= np.std(ndarray, axis=0)

        self.ndarray = ndarray
        self.model = model_object

    def get_image(self):
        return self.image

    def get_ndarray(self):
        return self.org_ndarray

    def get(self):
        while not self.shutdown_signal:
            x = self.q.get()
            self.q.task_done()
            yield x

    def __enter__(self):
        self.start()
        return self.get()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def start(self):
        self.shutdown_signal = False
        self.q = Queue(maxsize=20)
        self.worker_threads = []

        for c in range(self.worker_thread_count):
            t = threading.Thread(target=self.worker)
            self.worker_threads.append(t)
            t.daemon = True
            t.start()

    def stop(self):

        self.shutdown_signal = True

        if self.q.full():
            for i in range(self.worker_thread_count):
                self.q.get_nowait()

            for t in self.worker_threads:
                t.join()

    def worker(self):
        while not self.shutdown_signal:
            q = self.q

            cube_size = self.input_cube_size
            cube_half_size = cube_size / 2

            image_size = list(self.ndarray.shape)
            x_range = [
                [floor(int(image_size[0] * .15) + cube_half_size[0]), floor(int(image_size[0] * .45) - cube_half_size[0])],
                [floor(int(image_size[0] * .55) + cube_half_size[0]), floor(int(image_size[0] * .80) - cube_half_size[0])]
            ]
            y_range = [floor(int(image_size[1] * .25) + cube_half_size[1]), floor(int(image_size[1] * .75) - cube_half_size[1])]
            z_range = [floor(int(image_size[2] * .3) + cube_half_size[2]), floor(int(image_size[2] * .6) - cube_half_size[2])]

            random_center = [
                random.randint(*x_range[random.randint(0, 1)]),
                random.randint(*y_range),
                random.randint(*z_range)
            ]
            cube = self.ndarray[
                floor(random_center[0] - cube_half_size[0]):  floor(random_center[0] + cube_half_size[0]),
                floor(random_center[1] - cube_half_size[1]):  floor(random_center[1] + cube_half_size[1]),
                floor(random_center[2] - cube_half_size[2]):  floor(random_center[2] + cube_half_size[2]),
            ]

            cube = ImageHelper.pad_image(cube, self.model_cube_size)
            cube = ImageHelper.add_dim(cube)
            result = self.model.test({
                'image': np.array([cube])
            })
            q.put([random_center, cube, result])

