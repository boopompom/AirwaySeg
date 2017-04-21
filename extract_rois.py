import SimpleITK as sitk
from DatasetGenerator import DatasetGenerator

data_path = './RawData/'
data_path = './TinyData/'

# DatasetGenerator(data_path, './Output/', [15, 15, 15], ignore_classes=['_Mix_ipf']).extract_cubes()
# DatasetGenerator(data_path, './Output/', [15, 15, 15], ignore_classes=['_Mix_ipf']).extract_cubes_rescale()
# DatasetGenerator(data_path, './Output/', [15, 15, 15], ignore_classes=['_Mix_ipf']).extract_cubes_one_image()
class_map_6 = {
    '_Normal_ipf': 0,
    '_Emphysema_ipf': 1,
    '_Bronchovascular_ipf': 2,
    '_Ground Glass_ipf': 3,
    '_Ground Glass - Reticular_ipf': 4,
    '_Honeycomb_ipf': 5,
    '_Mix_ipf': None
}
class_map_2 = {
    '_Normal_ipf': 0,
    '_Emphysema_ipf': 1,
    '_Bronchovascular_ipf': None,
    '_Ground Glass_ipf': 1,
    '_Ground Glass - Reticular_ipf': 1,
    '_Honeycomb_ipf': 1,
    '_Mix_ipf': 1
}
"""
DatasetGenerator(
    data_path,
    'datasets/scaled_repeat_6.p',
    scale_strategy='upscale',
    padding_strategy='repeat',
    interpolation_strategy=sitk.sitkLinear,
    class_map=class_map_6
).extract_cubes()

DatasetGenerator(
    data_path,
    'datasets/scaled_neighbour_6.p',
    scale_strategy='upscale',
    padding_strategy='neighbour',
    interpolation_strategy=sitk.sitkLinear,
    class_map=class_map_6
).extract_cubes()

"""

DatasetGenerator(
    data_path,
    'datasets/scaled_zero_6.p',
    scale_strategy='upscale',
    padding_strategy='zero',
    interpolation_strategy=sitk.sitkLinear,
    class_map=class_map_6
).extract_cubes()

"""
DatasetGenerator(
    data_path,
    'datasets/scaled_neighbour_6.p',
    scale_strategy='upscale',
    padding_strategy='neighbour',
    interpolation_strategy=sitk.sitkLinear,
    class_map=class_map_6
).extract_cubes()

DatasetGenerator(
    data_path,
    'datasets/scaled_neighbour_2.p',
    scale_strategy='upscale',
    padding_strategy='neighbour',
    interpolation_strategy=sitk.sitkLinear,
    class_map=class_map_2
).extract_cubes()

DatasetGenerator(
    data_path,
    'datasets/scaled_repeat_2.p',
    scale_strategy='upscale',
    padding_strategy='repeat',
    interpolation_strategy=sitk.sitkLinear,
    class_map=class_map_2
).extract_cubes()
"""



"""
DatasetGenerator(
    data_path,
    'datasets/one_roi_normal.p',
    scale_strategy='none',
    padding_strategy='zero',
    interpolation_strategy=sitk.sitkLinear,
    class_map={
        '_Normal_ipf': 0,
        '_Emphysema_ipf': 1,
        '_Bronchovascular_ipf': 2,
        '_Ground Glass_ipf': 3,
        '_Ground Glass - Reticular_ipf': 4,
        '_Honeycomb_ipf': 5,
        '_Mix_ipf': 6
    }
).extract_cubes()

DatasetGenerator(
    data_path,
    'datasets/one_roi_normal_repeat.p',
    scale_strategy='none',
    padding_strategy='repeat',
    interpolation_strategy=sitk.sitkLinear,
    class_map={
        '_Normal_ipf': 0,
        '_Emphysema_ipf': 1,
        '_Bronchovascular_ipf': 2,
        '_Ground Glass_ipf': 3,
        '_Ground Glass - Reticular_ipf': 4,
        '_Honeycomb_ipf': 5,
        '_Mix_ipf': 6
    }
).extract_cubes()
"""
