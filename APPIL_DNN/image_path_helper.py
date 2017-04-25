import os
import random
import copy
import csv
import numpy as np

from APPIL_DNN.config import Config
from APPIL_DNN.path_helper import PathHelper


class ImagePathHelper:

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

    @staticmethod
    def get_image_list(input_path, dist=None, count=None, exclude=None):
        input_files = PathHelper.glob(os.path.join(input_path, "*.nrrd"))
        random.shuffle(input_files)

        if exclude is None:
            exclude = []

        for x in exclude:
            if x in input_files:
                input_files.remove(x)

        if dist is None:
            if count is None:
                return input_files
            else:
                return input_files[0:count]

        num_examples, num_classes, labels = ImagePathHelper.get_labels()

        is_relative_count = True
        if dist is not None:
            if sum(dist) != 1.0:
                is_relative_count = False
                max_dist = sum(dist)
                for idx, val in enumerate(dist):
                    dist[idx] /= max_dist

            if count is None:
                count = len(input_files)
                # raise ValueError("You have to specify a total count of samples")

        count_per_class = np.int16((np.array(dist) * count))
        remaining_count_per_class = np.copy(count_per_class)

        selected_files = {}
        for idx, val in enumerate(input_files):
            record_id = ((os.path.splitext(os.path.basename(val))[0]).split('_'))[0]
            label = labels[record_id]
            if remaining_count_per_class[label] != 0:
                if label not in selected_files:
                    selected_files[label] = []
                selected_files[label].append(val)
                remaining_count_per_class[label] -= 1

            if sum(remaining_count_per_class) == 0:
                total_files = []
                for key in selected_files:
                    total_files.extend(selected_files[key])
                return total_files

        # The distribution is specified in terms of absolute counts and we couldn't meet that count
        # Exit with an error
        if not is_relative_count:
            raise StopIteration("Required number of file could not be fulfilled from input directory")

        # The distribution was specified in terms of relative counts, we couldn't meet that so we simply reduce
        # the size of the output dataset so that the balance is kept
        class_size = min(count_per_class - remaining_count_per_class)

        total_files = []
        for key in selected_files:
            file_list = selected_files[key]
            if len(file_list) > class_size:
                file_list = file_list[0:class_size]
            print(len(file_list))
            total_files.extend(file_list)

        return total_files



    @staticmethod
    def get_next_image_path(input_path, dist=None, count=0, exclude=None):
        input_files = PathHelper.glob(os.path.join(input_path, "*.nrrd"))
        random.shuffle(input_files)

        num_examples, num_classes, labels = ImagePathHelper.get_labels()

        if dist is not None:
            if sum(dist) != 1.0:
                max_dist = sum(dist)
                for idx, val in enumerate(dist):
                    dist[idx] /= max_dist

            if count <= 0:
                raise ValueError("You have to specify a total count of samples")

        if exclude is None:
            exclude = []

        for x in exclude:
            if x in input_files:
                input_files.remove(x)

        # No specific distribution? Just grab files randomly
        if dist is None:
            while len(input_files):
                selected_image = random.choice(input_files)
                input_files.remove(selected_image)
                record_id = ((os.path.splitext(os.path.basename(selected_image))[0]).split('_'))[0]
                label = labels[record_id]
                yield selected_image, label
            return

        # build class map
        class_map = list(range(len(input_files)))
        if dist is not None:
            class_map = [[] for i in range(num_classes)]
            for i in range(len(input_files)):
                f = input_files[i]
                record_id = ((os.path.splitext(os.path.basename(f))[0]).split('_'))[0]
                label = labels[record_id]
                class_map[label].append((f, label))

        original_map = copy.deepcopy(class_map)

        count_yielded = 0
        while count_yielded < count:
            selected_class = np.random.choice(num_classes, 1, p=dist)[0]

            if len(class_map[selected_class]) == 0:
                class_map[selected_class] = copy.deepcopy(original_map[selected_class])

            selected = random.choice(class_map[selected_class])
            class_map[selected_class].remove(selected)

            count_yielded += 1
            yield selected