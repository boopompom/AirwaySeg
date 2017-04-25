import os
import sys
import glob
import random
import copy
import numpy as np

from APPIL_DNN.config import Config

#   Data
#   | - datasets
#   | - - DatasetName
#   | - - - all_images
#   | - - - train
#   | - - - - src
#   | - - - - nrrd
#   | - - - - np
#   | - - - valid
#   | - - - - src
#   | - - - - nrrd
#   | - - - - np
#   | - - - test
#   | - - - - src
#   | - - - - nrrd
#   | - - - - np
#   | - original_data
#   | - - raw
#   | - - - shrink_factor_1
#   | - - - shrink_factor_2
#   | - - segmented
#   | - - - shrink_factor_1
#   | - - - shrink_factor_2
#


class PathHelper:

    @staticmethod
    def glob(input_path):

        existent_files = glob.glob(input_path)

        for i in existent_files:
            if not os.path.isfile(i):
                existent_files.remove(i)
                continue

            filesize = os.stat(i).st_size
            if filesize == 0:
                existent_files.remove(i)
                continue
        return existent_files

    @staticmethod
    def get_dataset_root_path(dataset_name):
        root = Config.get('data_root')
        return os.path.abspath('/'.join([root, 'datasets', dataset_name]))


    @staticmethod
    def get_original_path(typ=None, shrink_factor=None):

        root = Config.get('data_root')
        parts = [root, 'original_data']

        if typ is not None:
            parts.append(typ)

        if shrink_factor is not None:
            parts.append(str(shrink_factor))

        return os.path.abspath('/'.join(parts))


    @staticmethod
    def get_dataset_path(dataset_name, typ=None, subtype=None):

        root = Config.get('data_root')
        parts = [root, 'datasets', dataset_name]

        if typ is not None:
            parts.append(typ)

        if subtype is not None:
            parts.append(subtype)

        return os.path.abspath('/'.join(parts))

    @staticmethod
    def exit_error(message):
        sys.stderr.write("\n Error: {0}".format(message))
        sys.stderr.flush()
        sys.exit(1)


    @staticmethod
    def create_sym_links(input_files, output_path, delete_existent=False):

        if delete_existent:
            try:
                files_to_delete = glob.glob(os.path.abspath(output_path + '/*'))
                [os.unlink(f) for f in files_to_delete]
            except:
                pass

        for input_file in input_files:
            try:
                basename = os.path.basename(input_file)
                full_output_path = os.path.abspath(output_path + '/' + basename)
                os.symlink(input_file, full_output_path)
            except FileExistsError:
                os.unlink(full_output_path)
                os.symlink(input_file, full_output_path)