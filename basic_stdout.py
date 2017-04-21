import threading
import sys


class BasicStdout:

    class __InnerClass:
        def __init__(self):
            self.lock = threading.Lock()
            self.format = ""
            self.variables = {}

        def set_format(self, fmt):
            self.lock.acquire()
            self.format = fmt
            sys.stdout.flush()
            self.lock.release()

        def set_variable(self, key, value):
            self.lock.acquire()
            self.variables[key] = value
            self.lock.release()
            self.update()


        def set_variables(self, variables):
            self.variables = variables

        def __enter__(self):
            self.lock.acquire()
            self.is_within_with = True
            return self.variables

        def __exit__(self, type, value, traceback):
            self.lock.release()
            self.update()

        def add_line(self, text):
            self.lock.acquire()
            print("\n" + text + "\n".format(**self.variables))
            self.lock.release()

        def update(self, variables=None):

            if variables is None:
                variables = {}

            full_dict = {**variables, **self.variables}
            self.lock.acquire()
            sys.stdout.write("\033[K")
            sys.stdout.write(self.format.format(**full_dict))
            sys.stdout.flush()
            self.lock.release()

    instance = None

    @staticmethod
    def get_instance():
        if BasicStdout.instance is None:
            BasicStdout.instance = BasicStdout.__InnerClass()
        return BasicStdout.instance
