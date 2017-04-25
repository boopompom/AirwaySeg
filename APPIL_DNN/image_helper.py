import SimpleITK as sitk
import numpy as np

from APPIL_DNN.path_helper import PathHelper

class ImageHelper:

    @staticmethod
    def read_image(image_path, target_dim):
        try:
            image = sitk.ReadImage(image_path, sitk.sitkFloat32)
        except RuntimeError:
            return None

        return ImageHelper.pad_image(image, target_dim)

    @staticmethod
    def pad_image(image, target_dim):
        size = image.GetSize()

        trim_start_list = [0, 0, 0]
        trim_end_list = [0, 0, 0]

        # This will trim down a image to a target size by trimming two sides in the requested dimension
        for i in range(len(size)):
            to_trim = 0
            if size[i] > target_dim[i]:
                to_trim = size[i] - target_dim[i]
            if int((to_trim * 100) / target_dim[i]) >= 50:
                # Difference between image size and target size is different by more than 50%
                # Trimming will likely destroy image data, discard the image altogether
                return None

            trim_start = int(to_trim / 2)
            trim_end = trim_start + (to_trim % 2)
            trim_start_list[i] = trim_start
            trim_end_list[i] = trim_end

        image = sitk.Crop(image, trim_start_list, trim_end_list)

        arr = sitk.GetArrayFromImage(image)

        # This will zero-pad an image to meet a specific size
        padding_map = []
        for i in range(len(target_dim)):
            padding_map.append([0, max(0, target_dim[i] - arr.shape[i])])

        x = np.pad(arr, padding_map, mode='constant', constant_values=0)

        return np.float32(x)