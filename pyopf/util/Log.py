import logging

import colorlog

__all__ = ['Log']

class Log:

    def __init__(self, path_to_log, case_name, logger_name="PyOPF", stochastic=False, log_method_time=True):
        self.logger = colorlog.getLogger(logger_name)
        self.flag_log_method_time = log_method_time

        # if logger exists, set the level to debug; otherwise, add the stream and file handlers
        if self.logger.hasHandlers():
            self.logger.setLevel(logging.DEBUG)
            self.logger.propagate = False

        else:

            self.logger = colorlog.getLogger(logger_name)
            self.logger.propagate = False

            # stream handler
            console_handler = colorlog.StreamHandler()
            console_handler.setLevel(logging.INFO)
            c_format = colorlog.ColoredFormatter(
                '%(log_color)s%(name)s - %(levelname)s - %(white)s%(message)s',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }

            )
            console_handler.setFormatter(c_format)
            self.logger.addHandler(console_handler)

            # file handler
            if not stochastic:
                file_handler = logging.FileHandler(path_to_log + case_name + '.log', 'w+')

                file_handler.setLevel(logging.DEBUG)
                f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(f_format)
                self.logger.addHandler(file_handler)

            self.logger.setLevel(logging.DEBUG)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def error(self, msg):
        self.logger.error(msg)
