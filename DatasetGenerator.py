import gc
import os
import sys
import time
import math
import random
import pickle
import datetime
import threading
import importlib
import numpy as np
import SimpleITK as sitk

from sklearn.decomposition import PCA
from glob import glob
from queue import *

from image_helper import ImageHelper
from running_stat import RunningStat
from basic_stdout import BasicStdout
from voi import ROI


class DatasetGenerator:

    worker_buffer_size = 1000
    dataset_buffer_size = 50000
    test_pct = 0.00
    dataset_tag = "rois"
    pca_components_factor = 0.75

    def __init__(self,
                 input_path,
                 output_filename,
                 voi_shape=None,
                 padding_strategy='repeat',
                 class_map=None):

        ignore_classes = []
        if class_map is not None:
            for class_from in class_map:
                class_to = class_map[class_from]
                if class_to is None:
                    ignore_classes.append(class_from)

        if len(voi_shape.shape) != 3:
            raise ValueError('VOI has to 3D')

        if voi_shape[0] % 2 == 0 or voi_shape[1] % 2 == 0 or voi_shape[2] % 2 == 0:
            raise ValueError('VOI dimensions has to be odd')

        if padding_strategy not in ['zero', 'repeat', 'neighbour']:
            raise ValueError('Padding strategy {0} is not implemented'.format(padding_strategy))

        self.class_map = class_map
        self.ignore_classes = ignore_classes

        self.voi_count = 0

        self.input_path = input_path
        self.output_filename = output_filename

        self.images = {}
        self.image_data = {}
        self.image_locks = {}
        self.class_counter = {}

        self.voi_list = []

        self.padding_strategy = padding_strategy


    def get_image(self, id):

        is_last_run = False

        self.image_locks[id].acquire()

        if id not in self.images:
            image_path = os.path.abspath(self.input_path + id + '.img.gz')
            self.images[id] = ImageHelper.read_image(image_path)

        self.image_locks[id].release()

        return self.images[id], is_last_run

    def generate_classes(self):

        lookup = {}
        counter = 0
        for cls in self.class_counter:
            lookup[cls] = counter
            counter += 1

        for cls in lookup:
            val = lookup[cls]
            lookup[cls] = np.zeros(counter)
            lookup[cls][val] = 1

        self.class_lookup = lookup

    def read_rois(self):
        
        self.image_data = {}
        self.class_counter = {}
        images = glob(self.input_path + "*.img.gz")
        for image in images:
            basename = os.path.basename(image).replace('.img.gz', '')
            self.image_data[basename] = []
            # self.image_locks[basename] = threading.Lock()

        rois = glob(self.input_path + "*.roi")
        for roi in rois:
            basename = os.path.basename(roi).split('.Merged.', 1)[0]
            if basename not in self.image_data:
                continue
            roi_list = []
            with open(roi) as f:
                lines = f.readlines()
                version = int(lines[0])
                num_roi = int(lines[1])
                # num_roi = 1
                start_offset = 2
                rois_read = 0
                roi_lines = []
                for j in range(start_offset, len(lines)):
                    current_line = lines[j].strip()
                    if current_line == 'end':
                        r = ROI(basename, roi_lines, class_map=self.class_map)
                        rois_read += 1
                        if r.roi_class is not None:
                            roi_list.append(r)
                            self.roi_count += 1
                            roi_lines = []
                            if r.roi_class not in self.class_counter:
                                self.class_counter[r.roi_class] = 0
                            self.class_counter[r.roi_class] += 1
                        if rois_read == num_roi:
                            break
                        else:
                            continue
                    roi_lines.append(current_line)

            self.image_data[basename] = roi_list
            self.roi_list.extend(roi_list)

        self.generate_classes()
        self.calc_bounds(self.roi_list)


    def calc_bounds(self, roi_list):
        self.roi_size_min = [float("inf"), float("inf"), float("inf")]
        self.roi_size_max = [float("-inf"), float("-inf"), float("-inf")]

        for roi in roi_list:
            for i in range(3):
                self.roi_size_min[i] = min(self.roi_size_min[i], roi.roi_size[i])
                self.roi_size_max[i] = max(self.roi_size_max[i], roi.roi_size[i])

    def load_cubes_crop(self):
        cube_shape = None
        counter = 0
        for idx in self.image_data:
            sys.stdout.write("\rLoading Images {0} out of {1}...".format(counter, len(self.image_data)))
            sys.stdout.flush()
            counter += 1

            image_path = os.path.abspath(self.input_path + idx + '.img.gz')
            image = ImageHelper.read_image(image_path, correct_spacing=False)
            roi_list = self.image_data[idx]
            cube_shape = self.load_image_cubes(image, roi_list, self.roi_size_min)
        return self.roi_size_min


    def load_cubes_upscale(self):

        cube_shape = None
        counter = 0

        new_rois = []
        for idx in self.image_data:
            if len(self.image_data[idx]) == 0:
                continue
            roi_list = self.image_data[idx]
            image_path = os.path.abspath(self.input_path + idx + '.img.gz')
            image, rois = ImageHelper.scale_rois(
                image_path, roi_list,
                0.5, interpolation_strategy=self.interpolation_strategy
            )
            new_rois.extend(rois)

        self.calc_bounds(new_rois)

        scaled_rois = []
        for idx in self.image_data:
            if len(self.image_data[idx]) == 0:
                continue

            sys.stdout.write("\rLoading Images {0} out of {1}...".format(counter, len(self.image_data)))
            sys.stdout.flush()
            counter += 1

            roi_list = self.image_data[idx]
            image_path = os.path.abspath(self.input_path + idx + '.img.gz')
            image, rois = ImageHelper.scale_rois(
                image_path, roi_list,
                0.5, interpolation_strategy=self.interpolation_strategy
            )
            cube_shape = self.load_image_cubes(image, rois, self.roi_size_max)
            scaled_rois.extend(rois)

        return scaled_rois, self.roi_size_max

    def load_cubes_raw(self):
        cube_shape = None
        counter = 0
        for idx in self.image_data:
            sys.stdout.write("\rLoading Images {0} out of {1}...".format(counter, len(self.image_data)))
            sys.stdout.flush()
            counter += 1

            image_path = os.path.abspath(self.input_path + idx + '.img.gz')
            image = ImageHelper.read_image(image_path, correct_spacing=False)
            roi_list = self.image_data[idx]
            cube_shape = self.load_image_cubes(image, roi_list, self.roi_size_max)
        return self.roi_size_max

    def load_image_cubes(self, image, roi_list, padded_size):
        cube = None
        image_spacing = list(image.GetSpacing())
        for roi in roi_list:
            roi.rescale_depth(1)
            if roi.roi_class not in self.spacing_dist:
                self.spacing_dist[roi.roi_class] = []
            self.spacing_dist[roi.roi_class].append(image_spacing[2])
            cube = roi.load_cube(image, self.padding_strategy, padded_size=padded_size)
        return cube.shape

    def load_cubes(self, padded_size):
        cube = None
        counter = 0
        for idx in self.image_data:
            roi_list = self.image_data[idx]
            for roi in roi_list:
                sys.stdout.write("\rLoading Cubes {0} out of {1}...".format(counter, self.roi_count))
                sys.stdout.flush()
                counter += 1
                roi.rescale_depth(1)
                if roi.roi_class not in self.spacing_dist:
                    self.spacing_dist[roi.roi_class] = []
                self.spacing_dist[roi.roi_class].append(self.images[idx].GetSpacing()[2])
                cube = roi.load_cube(self.images[idx], self.padding_strategy, padded_size=padded_size)
        return cube.shape

    def extract_cubes(self):

        print("Reading Rois...")
        self.read_rois()

        cube_size = None
        if self.scale_strategy == 'none':
            roi_list, cube_size  = self.load_cubes_raw()
        elif self.scale_strategy == 'crop':
            roi_list, cube_size  = self.load_cubes_crop()
        elif self.scale_strategy == 'upscale':
            roi_list, cube_size  = self.load_cubes_upscale()

        print("")
        print("Cube size {0}".format(cube_size))

        print("Detecting intersections...")
        for roi_1 in roi_list:
            roi_1.lookup_table = self.class_lookup
            roi_1.y = self.class_lookup[roi_1.roi_class]
            for roi_2 in roi_list:
                roi_1.intersects(roi_2)


        print("Writing data...")
        pickle.dump(roi_list, open(self.output_filename, "wb"))


    def extract_cubes_one_image(self):

        counter = 0
        rois = []
        self.read_rois()
        idx = None

        idx = '081832_20060531_1149725568.2203'
        roi_list = self.image_data[idx]

        spacing_factor = 1
        if idx not in self.images:
            image_path = os.path.abspath(self.input_path + idx + '.img.gz')

            # spacing_factor, self.images[idx] = ImageHelper.read_image(image_path, correct_spacing=True)
            spacing_factor = 1
            self.images[idx] = ImageHelper.read_image(image_path, correct_spacing=False)

        the_one_roi = None
        for roi in roi_list:
            if roi.roi_start_idx[0] != 101:
                continue
            the_one_roi = roi
            roi.rescale_depth(spacing_factor)
            cube = roi.load_cube(self.images[idx], padded_size=self.cube_size)
            assert (list(cube.shape) == self.cube_max_size)
            counter += 1
            print(counter)

        counter = 0
        roi_list = [the_one_roi]

        for roi_1 in roi_list:
            counter += 1
            roi_1.lookup_table = self.class_lookup
            roi_1.y = self.class_lookup[roi_1.roi_class]
            for roi_2 in roi_list:
                roi_1.intersects(roi_2)
            print(counter)

        rois.extend(self.image_data[idx])

        pickle.dump([the_one_roi], open("one_image.p", "wb"))

