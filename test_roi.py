
import time
import sys
import numpy as np
from voi_queue import VOIQueue
from voi import VOI

import SimpleITK as sitk
import threading
import json
import pprint


roi_q = VOIQueue('./Output/', random_seed=1992)



train_counter = {}
test_counter = {}
with roi_q as roi_generator:
    next(roi_generator)
    counter = 0
    collisons = 0
    try:
        while True:
            X_train, Y_train, roi_train, rot_train = roi_generator.send(('train', 1, True, False))
            X_test, Y_test, roi_test, rot_test = roi_generator.send(('test', 1, True, False))

            roi_train = roi_train[0]
            roi_test = roi_test[0]

            if roi_train.id not in train_counter:
                train_counter[roi_train.id] = 0
            train_counter[roi_train.id] += 1

            if roi_test.id not in test_counter:
                test_counter[roi_test.id] = 0
            test_counter[roi_test.id] += 1

            X_train = (X_train['image'][0])
            X_test = (X_test['image'][0])

            # h_train = np.histogram(X_train, bins=50)
            # h_test = np.histogram(X_test, bins=50)
            # print(h_train)
            # print(h_test)
            # exit()

            # print(X_train.shape)
            # print(X_test.shape)
            # x = np.corrcoef(X_train, X_test)
            x = abs(np.sum(X_train - X_test))
            if x == 0:
                collisons += 1
            # print(x)
            # x = abs(int(x[0, 1] * 100))

            sys.stdout.write("\rExamined {0} examples found {1}".format(counter, collisons))
            sys.stdout.flush()
            time.sleep(0.001)
            counter += 1
            #if x > 50:
            #    print(str(x) + " Yalhwy")
    except KeyboardInterrupt:
        print("")
        print(train_counter)
        print(test_counter)