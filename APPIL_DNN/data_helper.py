import numpy as np
import csv
import os
import random
import math

from APPIL_DNN.config import Config
from APPIL_DNN.path_helper import PathHelper
from APPIL_DNN.basic_stdout import BasicStdout
from APPIL_DNN.image_helper import ImageHelper
from APPIL_DNN.image_path_helper import ImagePathHelper


class DataHelper:


    @staticmethod
    def write_batch(X, Y, batch_number, np_path):
        filename = "data_{0}".format(batch_number)
        # print("Writing file " + filename)
        np.save(np_path + '/' + filename, np.array(X))

        filename = "labels_{0}".format(batch_number)
        # print("Writing file " + filename)
        np.save(np_path + '/' + filename, np.array(Y))

    @staticmethod
    def path_to_numpy(input_path, output_path, output_dimensions, batch_max_size, mean, var):

        num_examples, num_classes, labels = DataHelper.get_labels()
        std_printer = BasicStdout.get_instance()

        # require a uniform distribution of data
        dist = [1/num_classes] * num_classes

        images = ImagePathHelper.get_image_list(input_path, dist=dist, count=None, exclude=None)
        # images = PathHelper.glob(input_path + "/*.nrrd")
        random.shuffle(images)

        total_files = len(images)
        std_printer.set_format(
            'Converting image to numpy array, {files_done} out of {total_files} files processed' +
            '({batches_written} batches written) ({files_done_pct:.2f}%)\r'
        )
        std_printer.set_variables({
            'total_files': total_files,
            'batches_written': 0,
            'files_done': 0,
            'files_done_pct': 0
        })

        cum_batch_size = 0
        batch_number = 0
        X = []
        Y = []

        if len(images) == 0:
            PathHelper.exit_error("No images found in {0}".format(input_path))

        for image_path in images:

            cls = list(0 for p in range(num_classes))
            # print("Processing subject {0})".format(record_id))
            record_id = ((os.path.splitext(os.path.basename(image_path))[0]).split('_'))[0]
            label = labels[record_id]
            cls[label] = 1

            arr = ImageHelper.read_image(image_path, output_dimensions)
            if arr is None:
                with std_printer as vars:
                    vars['files_done'] += 1
                    if vars['total_files'] != 0:
                        vars['files_done_pct'] = (vars['files_done'] / vars['total_files']) * 100
                    else :
                        vars['files_done_pct'] = 0

                continue

            # Zero center and normalize data
            arr -= mean
            arr /= math.sqrt(abs(var))
            # arr /= abs(var)
            # arr  = np.int16(arr)

            if cum_batch_size + arr.nbytes > batch_max_size:
                # print("Batch size reached, starting a new batch")
                DataHelper.write_batch(X, Y, batch_number, output_path)
                X = []
                Y = []
                cum_batch_size = 0
                batch_number += 1
                with std_printer as vars:
                    vars['batches_written'] += 1

            cum_batch_size += arr.nbytes

            X.append(arr)
            Y.append(cls)

            with std_printer as vars:
                vars['files_done'] += 1
                if vars['total_files'] != 0:
                    vars['files_done_pct'] = (vars['files_done'] / vars['total_files']) * 100
                else :
                    vars['files_done_pct'] = 0

        DataHelper.write_batch(X, Y, batch_number, output_path)
        with std_printer as vars:
            vars['batches_written'] += 1
        std_printer.add_line("Done!")

    @staticmethod
    def get_labels():
        root = Config.get('data_root')
        labels_file = os.path.abspath('/'.join([root, 'original_data', 'labels.csv']))
        num_examples = 0
        num_classes = 0
        result = {}
        with open(labels_file, 'r') as csvfile:
            csv_reader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
            for row in csv_reader:
                record_id = row['record_id']
                num_examples += 1

                emphysema = 0 if row['emphysema'] == '' else int(row['emphysema'])
                bronchiectasis = 0 if row['bronchiectasis1'] == '' else int(row['bronchiectasis1'])

                cls = -1

                if emphysema == 0 and bronchiectasis == 0:
                    cls = 0
                if emphysema == 1 and bronchiectasis == 0:
                    cls = 1
                if emphysema == 0 and bronchiectasis == 1:
                    cls = 2
                if emphysema == 1 and bronchiectasis == 1:
                    cls = 3

                # This will set up a 4 class dataset
                # result[record_id] = cls
                # num_classes = 4

                # This will set up a 2 class dataset
                result[record_id] = emphysema
                num_classes = 2

            return num_examples, num_classes, result




