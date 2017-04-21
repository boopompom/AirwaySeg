import math
import numpy as np
import SimpleITK as sitk



class ImageHelper:


    @staticmethod
    def copy_roi(original_roi):
        from voi import ROI
        new_roi = ROI(None, None, None)

        new_roi.y = original_roi.y
        new_roi.lookup_table = original_roi.lookup_table
        new_roi.has_intersection = original_roi.has_intersection
        new_roi.spacing_factor = original_roi.spacing_factor
        new_roi.image_id = original_roi.image_id
        new_roi.first_number = original_roi.first_number
        new_roi.roi_org_class = original_roi.roi_org_class
        new_roi.param_count = original_roi.param_count
        new_roi.roi_class = original_roi.roi_class
        new_roi.roi_params = original_roi.roi_params
        new_roi.roi_type = original_roi.roi_type
        new_roi.id = original_roi.id
        new_roi.image_z_spacing = original_roi.image_z_spacing

        new_roi.cube = np.copy(original_roi.cube)
        new_roi.padded_size = np.copy(original_roi.padded_size)
        new_roi.roi_center = np.copy(original_roi.roi_center)
        new_roi.roi_size = np.copy(original_roi.roi_size)
        new_roi.roi_start_idx = np.copy(original_roi.roi_start_idx)
        new_roi.roi_end_idx = np.copy(original_roi.roi_end_idx)
        new_roi.roi_idx_size = np.copy(original_roi.roi_idx_size)

        return new_roi

    @staticmethod
    def add_dim(ndarray):
        s = list(ndarray.shape)
        if len(s) > 3:
            return ndarray
        new_shape = list(s)
        new_shape.append(1)
        ndarray.shape = new_shape
        return ndarray

    @staticmethod
    def remove_dim(ndarray):
        s = list(ndarray.shape)
        if len(s) <= 3:
            return ndarray
        return np.squeeze(ndarray, 3)

    @staticmethod
    def crop_image(ndarray, target_shape):
        lookup = []
        current_shape = np.array(ndarray.shape)

        for idx, v in enumerate(target_shape):
            target = target_shape[idx]
            current = current_shape[idx]
            diff = current - target
            if diff < 0:
                raise ValueError("Cropping failed, target dimensions is larger than image dimensions")

            diff_pair = [int(diff / 2), int(diff / 2)]

            # if diff is odd, add one to the "right" side
            if diff % 2 == 1:
                diff_pair[1] += 1

            lookup.append(diff_pair)

        new_array = ndarray[
            lookup[0][0]:current_shape[0] - lookup[0][1],
            lookup[1][0]:current_shape[1] - lookup[1][1],
            lookup[2][0]:current_shape[2] - lookup[2][1],
        ]

        return new_array

    @staticmethod
    def get_roi(image, start, size):
        im = sitk.RegionOfInterest(image, size, start)
        im.SetOrigin([0, 0, 0])
        return np.swapaxes(sitk.GetArrayFromImage(im), 0, 2)

    @staticmethod
    def pad_image(ndarray, target_shape, mode='repeat'):

        if mode is None:
            return ImageHelper.crop_image(ndarray, target_shape)

        lookup = []
        current_shape = np.array(ndarray.shape)

        for idx, v in enumerate(target_shape):
            target = target_shape[idx]
            current = current_shape[idx]
            diff = target - current
            if diff < 0:
                raise ValueError("Padding failed, target dimensions is smaller than image dimensions")

            diff_pair = [int(diff / 2), int(diff / 2)]

            # if diff is odd, add one to the "right" side
            if diff % 2 == 1:
                diff_pair[1] += 1

            lookup.append(diff_pair)

        if mode == 'zero':
            return np.lib.pad(ndarray, lookup, 'constant', constant_values=-1024)
        elif mode == 'repeat':
            return np.lib.pad(ndarray, lookup, 'symmetric')
        else:
            raise ValueError("Unrecognized padding mode {0}".format(mode))

    @staticmethod
    def extract_rois(image_path, roi_list, label_list=None, target_spacing=None, interpolation_strategy=sitk.sitkNearestNeighbor):


        # im_original = ImageHelper.read_image(image_path, correct_spacing=False)
        # ROI coordinates are expected to be in Slicer Coordinate system
        im_original = sitk.ReadImage(image_path)

        if target_spacing is None:
            target_spacing = np.array([-1, -1, -1])

        actual_spacing = np.array(im_original.GetSpacing())
        for idx, t in enumerate(target_spacing):
            if t == -1:
                target_spacing[idx] = actual_spacing[idx]
        scale_factor = actual_spacing / target_spacing

        actual_size = np.array(im_original.GetSize())
        target_size = np.ceil(actual_size) * scale_factor

        print(scale_factor )
        eff_roi_list = []
        im_new = None
        if scale_factor[0] == scale_factor[1] == scale_factor[2] == 1:
            im_new = im_original
        else:
            f = sitk.ResampleImageFilter()
            f.SetReferenceImage(im_original)
            f.SetSize(target_size)
            f.SetOutputSpacing(target_spacing)
            f.SetOutputOrigin(im_original.GetOrigin())
            f.SetOutputDirection(im_original.GetDirection())
            f.SetInterpolator(interpolation_strategy)
            f.SetOutputPixelType(sitk.sitkFloat32)
            f.SetDefaultPixelValue(1024)
            f.SetTransform(sitk.Transform())
            im_new = f.Execute(im_original)

        for roi in roi_list:
            org_physical = im_original.TransformContinuousIndexToPhysicalPoint(roi['center'])
            new_center = np.array(im_new.TransformPhysicalPointToIndex(org_physical))
            new_size = np.int16(np.array(roi['size']) * scale_factor)
            start_idx = np.int16(new_center - (new_size / 2))
            new_roi = {
                "start" : start_idx.tolist(),
                "center": new_center.tolist(),
                "size": new_size.tolist(),
                "data": None
            }

            cube = sitk.RegionOfInterest(im_new, new_size.tolist(), start_idx.tolist())
            cube.SetOrigin([0, 0, 0])
            cube = np.swapaxes(sitk.GetArrayFromImage(cube), 0, 2)
            new_roi['data'] = ImageHelper.add_dim(cube)
            eff_roi_list.append(new_roi)

        return im_new, eff_roi_list


    @staticmethod
    def scale_rois(image_path, roi_list, target_spacing, interpolation_strategy=sitk.sitkNearestNeighbor):

        im_original = ImageHelper.read_image(image_path, correct_spacing=False)

        sx, sy, sz = im_original.GetSpacing()
        scale_factor = sz / target_spacing

        x, y, z = im_original.GetSize()
        z = math.ceil(z * scale_factor)

        if scale_factor == 1:
            return im_original, roi_list

        f = sitk.ResampleImageFilter()
        f.SetReferenceImage(im_original)
        f.SetSize([x, y, z])
        f.SetOutputSpacing([sx, sy, target_spacing])
        f.SetOutputOrigin(im_original.GetOrigin())
        f.SetOutputDirection(im_original.GetDirection())
        f.SetInterpolator(interpolation_strategy)
        f.SetOutputPixelType(sitk.sitkFloat32)
        f.SetDefaultPixelValue(1024)
        f.SetTransform(sitk.Transform())
        im_new = f.Execute(im_original)

        # sitk.WriteImage(im_new, "test.nrrd")
        # im_new = sitk.Resample(im_original, transform, interpolation_strategy, 1024, sitk.sitkFloat32)

        new_rois = []
        for roi in roi_list:
            org_physical = im_original.TransformContinuousIndexToPhysicalPoint(roi.roi_center)
            new_idx = im_new.TransformPhysicalPointToContinuousIndex(org_physical)
            new_roi = ImageHelper.copy_roi(roi)
            new_roi.roi_center = np.array(new_idx)
            new_roi.roi_size[2] *= scale_factor
            new_roi.roi_idx_size = np.int16(new_roi.roi_size)
            new_rois.append(new_roi)

        return im_new, new_rois


    @staticmethod
    def read_image(path, correct_spacing=False):
        im = sitk.ReadImage(path)
        ndarray = sitk.GetArrayFromImage(im)

        spacing = list(im.GetSpacing())
        direction = im.GetDirection()
        origin = im.GetOrigin()

        spacing_factor = 1
        if correct_spacing:
            if spacing[2] == 0.5 or spacing[2] == 0.625:
                ndarray = ndarray[::-2, :, :]
                spacing[2] *= 2
                spacing_factor = 1/2
        else:
            ndarray = ndarray[::-1, :, :]

        del im

        new_im = sitk.GetImageFromArray(ndarray)
        new_im.SetSpacing(spacing)
        new_im.SetDirection(direction)
        new_im.SetOrigin(origin)

        del ndarray

        if correct_spacing:
            return spacing_factor, new_im
        return new_im