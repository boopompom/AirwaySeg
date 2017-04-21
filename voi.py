import math
import string
import random
import hashlib

import numpy as np
import SimpleITK as sitk

from image_helper import ImageHelper


class ROI:

    @staticmethod
    def gen_id(image_id, center, size):

        key = "{0},{1:.2f},{2:.2f},{3:.2f},{4:.2f},{5:.2f},{6:.2f}".format(
            image_id,
            center[0], center[1], center[2],
            size[0], size[1], size[2]
        )
        return hashlib.sha256(key.encode()).hexdigest()

    def intersects(self, roi):

        if self == roi:
            return False

        if roi.image_id != self.image_id:
            return False

        this_size = np.array(self.cube.shape)
        that_size = np.array(roi.cube.shape)

        this_start = self.roi_center - (this_size / 2)
        that_start = roi.roi_center - (that_size / 2)

        this_end = self.roi_center + (this_size / 2)
        that_end = roi.roi_center + (that_size / 2)

        this_x0 = this_start[0]
        this_x1 = this_end[0]
        this_y0 = this_start[1]
        this_y1 = this_end[1]
        this_z0 = this_start[2]
        this_z1 = this_end[2]

        that_x0 = that_start[0]
        that_x1 = that_end[0]
        that_y0 = that_start[1]
        that_y1 = that_end[1]
        that_z0 = that_start[2]
        that_z1 = that_end[2]

        intersect = not(
            this_x1 < that_x0 or
            this_x0 > that_x1 or
            this_y1 < that_y0 or
            this_y0 > that_y1 or
            this_z1 < that_z0 or
            this_z0 > that_z1
        )

        if not self.has_intersection and intersect:
            self.has_intersection = True

        return intersect

    def rescale_depth(self, spacing_factor):

        self.spacing_factor = spacing_factor

        self.roi_size[2] *= spacing_factor
        self.roi_center[2] *= spacing_factor

        self.roi_size = np.int16(np.ceil(self.roi_size))

        self.roi_start_idx = np.int16(self.roi_center - (self.roi_size / 2))
        self.roi_end_idx = np.int16(self.roi_center + (self.roi_size / 2))
        self.roi_idx_size = np.int16(self.roi_size)

    def test(self, image):

        cube = self.get_cube(padding=False)

        start_idx = self.roi_center - (self.roi_size / 2)
        end_idx = self.roi_center + (self.roi_size / 2)
        sz = end_idx - start_idx

        for idx, i in enumerate(self.roi_center):
            assert float.is_integer(start_idx[idx])
            assert float.is_integer(end_idx[idx])
            assert float.is_integer(sz[idx])

        start_idx = np.int16(start_idx)
        end_idx = np.int16(end_idx)
        sz = np.int16(sz)


        im = sitk.RegionOfInterest(image, sz.tolist(), start_idx.tolist())
        im.SetOrigin([0, 0, 0])
        correct_cube = np.swapaxes(sitk.GetArrayFromImage(im), 0, 2)

        assert np.sum(np.power(cube - correct_cube, 2)) == 0

        # sitk.WriteImage(sitk.GetImageFromArray(np.swapaxes(cube, 0, 2)), "before_rotation.nrrd")
        # cube = self.get_cube(rotation=(0, 0, 45), padding=False)
        # sitk.WriteImage(sitk.GetImageFromArray(np.swapaxes(cube, 0, 2)), "after_rotation.nrrd")



        pass



    def __init__(self, idx, section, class_map=None):

        self.cube = None
        self.y = None
        self.lookup_table = None
        self.padded_size = None
        self.has_intersection = False
        self.spacing_factor = 1
        self.image_id = None
        self.first_number = None
        self.roi_org_class = None
        self.roi_class = None
        self.param_count = None
        self.roi_params = None
        self.roi_type = None
        self.roi_center = None
        self.roi_size = None
        self.id = None
        self.roi_start_idx = None
        self.roi_end_idx = None
        self.roi_idx_size = None
        self.image_z_spacing = None

        if idx is None and section is None:
            return

        self.image_id = idx
        self.first_number = int(section[0])

        self.roi_org_class = section[1]
        self.roi_class = self.roi_org_class
        if class_map is not None:
            if self.roi_org_class not in class_map:
                raise ValueError("Class {0} not found in class map".format(self.roi_org_class))
            self.class_map = class_map
            self.roi_class = class_map[self.roi_org_class]

        self.param_count = int(section[2])

        after_param_offset = 2 + self.param_count

        self.roi_params = section[2:after_param_offset]

        self.roi_type = int(section[after_param_offset + 1])
        self.roi_center = np.array([float(x) for x in section[after_param_offset + 2].split(',')])
        self.roi_size = np.array([float(x) for x in section[after_param_offset + 3].split(',')])
        if self.roi_size[0] % 2 == 0 or self.roi_size[1] % 2 == 0 or self.roi_size[2] % 2 == 0:
            raise ValueError("Even number sized ROIs are not supported")

        self.id = ROI.gen_id(idx, self.roi_center, self.roi_size)

        self.rescale_depth(self.spacing_factor)


    def get_set(self, rotation=None):
        X = self.get_cube(rotation=rotation, as_nd=True)
        Y = self.y
        return X, Y

    def get_cube(self, rotation=None, padding=True, as_nd=True):

        cube_shape = np.array(self.cube.shape)
        half_cube_shape = cube_shape/2
        half_roi_size = (self.roi_idx_size/2)
        idx_start = np.int16(np.floor(half_cube_shape - half_roi_size))
        idx_end = np.int16(np.floor(half_cube_shape + half_roi_size))

        actual_size = self.roi_idx_size
        padded_size = self.padded_size

        # print(cube_shape)
        # print(padded_size)
        # print(actual_size)

        if rotation is not None:
            if sum(rotation) == 0:
                rotation = None

        ndarray = None
        if rotation is None:
            # No rotation, skip all these cpu intensive operations
            ndarray = self.cube[
                idx_start[0]:idx_end[0],
                idx_start[1]:idx_end[1],
                idx_start[2]:idx_end[2]
            ]
        else:

            # We have rotation
            rotation = np.float32(rotation) * (math.pi/180)

            rotation_center = half_cube_shape

            rigid_euler = sitk.Euler3DTransform()
            # rigid_euler.SetRotation(float(rotation[0]), float(rotation[1]), float(rotation[2]))
            rigid_euler.SetRotation(float(rotation[0]), float(rotation[1]), float(rotation[2]))

            rigid_euler.SetCenter(rotation_center - 0.5)
            ndarray = np.swapaxes(
                        sitk.GetArrayFromImage(
                            sitk.Resample(
                                sitk.GetImageFromArray(np.swapaxes(self.cube, 0, 2)),
                                rigid_euler,
                                sitk.sitkLinear,
                                0,
                                sitk.sitkFloat32
                            )
                        ),
                        0, 2)
            new_center = (np.array(ndarray.shape) / 2)
            new_start = np.int16(new_center - half_roi_size)
            new_end = np.int16(new_center + half_roi_size)

            ndarray = np.float32(ndarray[
                new_start[0]:new_end[0],
                new_start[1]:new_end[1],
                new_start[2]:new_end[2]
            ])

        # Padding is done after rotation, maybe try to pad before rotation
        if padding:
            # print(ndarray.shape)
            # print(padded_size)
            ndarray = ImageHelper.pad_image(ndarray, padded_size, mode='zero')
            # print(ndarray.shape)
        return ndarray if as_nd else sitk.GetImageFromArray(np.swapaxes(ndarray, 0, 2))

    def load_cube(self, image, padding_strategy, padded_size=None):

        output_array = None
        if padded_size is None:
            padded_size = self.roi_size

        self.padded_size = padded_size

        diagonal_length = np.sqrt(np.sum(np.power(padded_size, 2)))

        diagonal_size = [diagonal_length , diagonal_length , diagonal_length]
        for idx, i in enumerate(self.roi_size):
            if (diagonal_size[idx] - i) % 2 != 0:
                diagonal_size[idx] += 1
                pass

        # self.image_z_spacing = image.GetSpacing()[2]
        diagonal_size = np.array(diagonal_size)

        if padding_strategy == 'neighbour':
            max_sz = diagonal_size
            max_start = np.int16(np.floor(self.roi_center - max_sz / 2))
            max_end = np.int16(np.floor(self.roi_center + max_sz / 2))
            output_array = np.float32(ImageHelper.get_roi(image, max_start.tolist(), np.int16(max_sz).tolist()))
            """
            output_array = np.float32(ndarray[
                 max_start[0]:max_end[0],
                 max_start[1]:max_end[1],
                 max_start[2]:max_end[2]
            ])
            """
        elif padding_strategy in [None, 'zero', 'repeat']:
            output_array = np.float32(ImageHelper.get_roi(image, self.roi_start_idx.tolist(),  self.roi_idx_size.tolist()))
            output_array = ImageHelper.pad_image(output_array, diagonal_size, mode=padding_strategy)
            """
                        output_array = np.float32(ndarray[
                             self.roi_start_idx[0]:self.roi_end_idx[0],
                             self.roi_start_idx[1]:self.roi_end_idx[1],
                             self.roi_start_idx[2]:self.roi_end_idx[2]
                         ])
            """
        else:
            raise ValueError("Unrecognized padding mode {0}".format(padding_strategy))

        self.cube = output_array
        # self.test(image)

        return self.cube