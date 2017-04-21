from DatasetGenerator import DatasetGenerator

data_path = './RawData/'
DatasetGenerator(data_path, './Output/', ignore_classes=['_Mix_ipf']).process()