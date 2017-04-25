import datetime
import logging
import os
import subprocess
import sys
import time
import threading
import queue

from APPIL_DNN.config import Config
from APPIL_DNN.basic_stdout import BasicStdout


class ProcessRunner:

    def __init__(self, fmt, max_process=None):

        #self.max_process = Config.get('max_process')
        if max_process is not None:
            self.max_process = max_process
        self.threads = []
        self.fmt = fmt
        self.q = queue.Queue()
        self.std_printer = BasicStdout.get_instance()
        self.std_printer.set_format(fmt)
        self.std_printer.set_variables({
            'total_files': 0,
            'files_done': 0,
            'files_done_pct': 0
        })

        formatter = logging.Formatter("[%(asctime)s]  %(name)-12s %(levelname)s: %(message)s", style="{")

        self.logger = logging.getLogger('process_runner')
        file_handler = logging.FileHandler("process_runner.log")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

        self.exit_flag = False

    def enqueue(self, file_count, params):
        total_files = self.std_printer.get_variable('total_files')
        self.std_printer.set_variable('total_files', total_files + file_count)
        self.q.put((file_count, params))

    def perform_task(self):

        while not self.exit_flag:
            data = None

            try:
                data = self.q.get_nowait()
            except:
                return

            file_count, params = data

            started = datetime.datetime.now()
            self.logger.debug("Process {cmd} started at {time}".format(**{
                'cmd': " ".join(params),
                'time': started.strftime('[%d-%m-%Y %H:%M:%S]')
            }))
            result = subprocess.run(
                params,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            ended = datetime.datetime.now()
            elapsed = ended - started
            self.logger.debug("Process {cmd} ended at {time} and ran in {elapsed} seconds : \n {out}".format(**{
                'cmd': " ".join(params),
                'out': result.stdout,
                'time': ended.strftime('[%d-%m-%Y %H:%M:%S]'),
                'elapsed': elapsed.seconds
            }))

            if len(result.stderr):
                self.logger.error("Process {cmd} returned an error : \n {err}".format(**{
                    'cmd': " ".join(params),
                    'err': result.stderr
                }))

            self.logger.handlers[0].flush()

            with self.std_printer as vars:
                vars['files_done'] += 1
                if vars['total_files'] != 0:
                    vars['files_done_pct'] = (vars['files_done'] / vars['total_files']) * 100
                else :
                    vars['files_done_pct'] = 0

            self.q.task_done()

    def cancel(self):
        self.exit_flag = True

    def start(self):

        for c in range(self.max_process):
            t = threading.Thread(target=self.perform_task)
            t.daemon = True
            t.start()
            self.threads.append(t)



        try:
            while True:
                time.sleep(1)
                if self.q.empty():
                    break

        except (KeyboardInterrupt, SystemExit):
            self.fmt = self.fmt + " ---- Quitting, please wait...."
            self.std_printer.set_format(self.fmt)
            self.cancel()
            for t in self.threads:
                t.join()
