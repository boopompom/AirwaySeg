import os
from APPIL_DNN.process_runner import ProcessRunner

def build_set(is_binary):
    root_path = "D:/Vida/ResultsDir2.0/"
    bin_path = os.path.abspath('CPP/bld/Debug/DICOMProcessor.exe')

    dirs = os.listdir(root_path)

    # enabled_labels = [1, 25, 44, 45, 26, 27, 92, 91, 2, 5, 23, 22, 10, 20, 21, 4, 9, 8, 3, 6, 48, 39, 51, 137, 188, 189, 43, 42, 17, 28, 29]

    # Right side
    enabled_labels = [1, 7, 3, 25, 45, 44, 48, 49, 51, 92, 137, 188, 189, 27, 6, 91]

    dirs_to_process = []
    for dir in dirs:

        full_dir = os.path.abspath(root_path + dir)
        dicom_path = os.path.join(full_dir, 'dicom')
        airway_path = os.path.join(full_dir, 'ZUNU_vida-aircolor.img.gz')
        lung_path = os.path.join(full_dir, 'ZUNU_vida-lung.img.gz')
        xml_path =  os.path.join(full_dir, 'ZUNU_vida-xmlTree.xml')

        dicom_exists = os.path.exists(dicom_path)
        airway_exists = os.path.isfile(airway_path)
        lung_exists = os.path.isfile(lung_path)
        xml_path = os.path.isfile(xml_path)

        is_valid_dir = dicom_exists and airway_exists and lung_exists and xml_path

        if is_valid_dir != True:
            print("Skipping invalid folder {0}".format(dir))
            continue

        dirs_to_process.append(full_dir)

    batches_per_exe = 5
    if len(dirs_to_process) > 0:
        fmt = 'Extracting VOIs, {files_done} out of {total_files} images processed ({files_done_pct:.2f}%)\r'
        runner = ProcessRunner(fmt, max_process=1)

        for i in range(0, len(dirs_to_process), batches_per_exe):

            batch_dirs = dirs_to_process[i:i+batches_per_exe]
            segs = [bin_path]
            [segs.extend(['-i', i]) for i in batch_dirs]
            [segs.extend(['-l', str(l)]) for l in enabled_labels]
            segs.extend(['-d', '53'])
            if is_binary:
                segs.extend(['-b'])
                segs.extend(['-v', '10000'])
                segs.extend(['-o', os.path.abspath('./Output/2_class/')])
            else:
                segs.extend(['-v', '30'])
                segs.extend(['-o', os.path.abspath('./Output/32_class/')])
            segs.extend(['-r', '1992'])
            segs.extend(['-t', str(batches_per_exe)])
            runner.enqueue(batches_per_exe, segs)
            print(" ".join(segs))
            exit()
        runner.start()

build_set(False)
# build_set(True)
