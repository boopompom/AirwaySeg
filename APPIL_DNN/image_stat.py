#!/usr/bin/env python3

import numpy as np
import glob
import math
import random
import sys
import os
import threading

import SimpleITK as sitk
from queue import *

from APPIL_DNN.image_helper import ImageHelper
from APPIL_DNN.running_stat import RunningStat
from APPIL_DNN.path_helper import PathHelper
from APPIL_DNN.basic_stdout import BasicStdout
from APPIL_DNN.config import Config


class ImageStat:

    def __init__(self, input_path, output_dim):

        self.input_path = input_path
        self.output_dimensions = output_dim
        self.rs = RunningStat()
        self.q = Queue()
        self.std_printer = BasicStdout.get_instance()

    def worker(self):

        while True:
            image_path = self.q.get()
            if image_path is None:
                self.q.task_done()
                break

            arr = ImageHelper.read_image(image_path, self.output_dimensions)
            if arr is not None:
                self.rs.add_batch(arr)

            files_done = self.std_printer.get_variable('files_done') + 1
            total_files = self.std_printer.get_variable('total_files')

            self.std_printer.set_variable('files_done', files_done)
            self.std_printer.set_variable('files_done_pct', (files_done / total_files) * 100)

            self.q.task_done()

    def get_stats(self):

        images = PathHelper.glob(os.path.abspath(self.input_path) + '/*.nrrd')

        max_threads = Config.get('max_process')

        total_files = len(images)

        self.std_printer.set_format(
            'Gathering image statistics {files_done} out of {total_files} files processed ({files_done_pct:.2f}%)\r')
        self.std_printer.set_variables({
            'total_files': total_files,
            'files_done': 0,
            'files_done_pct': 0
        })

        if len(images) == 0:
            PathHelper.exit_error("No images found in {0}".format(self.input_path))

        for image_path in images:
            self.q.put(image_path)

        threads = []
        for c in range(max_threads):
            t = threading.Thread(target=self.worker)
            t.start()
            threads.append(t)
            self.q.put(None)

        self.q.join()

        for t in threads:
            t.join()

        self.std_printer.add_line(
            "Data mean {m}, Data variance {v}"
                .format(**{'m': self.rs.get_meanvalue(), 'v': self.rs.get_variance()}))

        return self.rs.get_meanvalue(), self.rs.get_variance()


