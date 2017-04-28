import math
import string
import random
import hashlib

import numpy as np
import SimpleITK as sitk

from image_helper import ImageHelper


class VOI:

    def __init__(self, image_id, dicom_path, dataset_path, json, modes=None):

        if modes is None:
            raise ValueError("VOI modes dict is required, example {'local': [25, 25, 25]}")

        self.center = np.array(json['idx'], dtype=np.int16)
        self.image_id = image_id
        self.dicom_path = dicom_path
        self.dataset_path = dataset_path

        self.cubes = {}
        self.cubes_nd = {}
        self.vois_size = {}
        self.cubes_start = {}
        self.cubes_size = {}

        for idx in modes:
            self.cubes[idx] = None
            self.cubes_nd[idx] = None

            self.vois_size[idx] = np.array(modes[idx], dtype=np.int16)

            dg = np.sqrt(np.sum(np.power(self.vois_size[idx], 2)))
            if dg % 2 == 0:
                dg -= 1

            self.cubes_size[idx] = np.array([dg, dg, dg], dtype=np.int16)
            for sz in self.vois_size[idx]:
                if sz % 2 == 0:
                    raise ValueError("Even number sized VOIs are not supported")

            self.cubes_start[idx] = self.center - (self.cubes_size[idx] - 1) / 2
            self.cubes_start[idx] = np.int16(self.cubes_start[idx])

        self.modes = modes

        self.y = np.array(json['cls_arr'])
        self.y_name = json['cls_name']
        self.y_val = json['cls']

    def get_nd_cube_center(self, cube, mode):
        size = np.int16(self.modes[mode])

        new_center = (np.array(cube.shape) / 2)
        new_start = np.int16(new_center - (size/2))
        new_end = np.int16(new_center + (size/2))

        return cube[
           new_start[0]:new_end[0],
           new_start[1]:new_end[1],
           new_start[2]:new_end[2]
        ]

    def get_im_cube_center(self, im, mode):
        size = self.modes[mode]

        x = sitk.Image()
        new_center = np.array(im.GetSize()) / 2
        new_start = np.int16(new_center - size)
        new_end = np.int16(new_center + size)

        new_start = [int(new_start[0]), int(new_start[1]), int(new_start[2])]
        return sitk.RegionOfInterest(im, size, new_start)

    def get_cube(self, image, mode=None, rotation=None, as_nd=True):

        if mode not in self.modes:
            raise ValueError("Invalid VOI mode {0}".format(mode))

        if rotation is None or sum(rotation) == 0:
            rotation = None

        if self.cubes[mode] is None:
            sz = self.cubes_size[mode]
            st = self.cubes_start[mode]

            voi_size = [int(sz[0]), int(sz[1]), int(sz[2])]
            voi_start = [int(st[0]), int(st[1]), int(st[2])]
            try:
                self.cubes[mode] = sitk.RegionOfInterest(image, voi_size, voi_start)
                self.cubes_nd[mode] = sitk.GetArrayFromImage(self.cubes[mode])
            except:
                return None
                #print("ERR")


        cube = self.cubes[mode]
        cube_nd = self.cubes_nd[mode]
        size = self.cubes_size[mode]

        if rotation is None:
            return self.get_nd_cube_center(cube_nd, mode) if as_nd else self.get_im_cube_center(cube, mode)

        cube_shape = np.array(cube_nd.shape)
        half_cube_shape = cube_shape/2

        rotation = np.float32(rotation) * (math.pi/180)
        rotation_center = (cube_shape/2) - 0.5

        rotation= [float(rotation[0]), float(rotation[1]), float(rotation[2])]
        center_idx = [int(rotation_center[0]), int(rotation_center[1]), int(rotation_center[2])]
        center_pnt = cube.TransformIndexToPhysicalPoint(center_idx)

        rigid_euler = sitk.Euler3DTransform()
        rigid_euler.SetRotation(rotation[0], rotation[1], rotation[2])
        rigid_euler.SetCenter(center_pnt)

        rotated_cube = sitk.Resample(cube, rigid_euler, sitk.sitkLinear, 0, sitk.sitkFloat32)
        if not as_nd:
            return self.get_im_cube_center(rotated_cube, mode)

        return self.get_nd_cube_center(sitk.GetArrayFromImage(rotated_cube), mode)
