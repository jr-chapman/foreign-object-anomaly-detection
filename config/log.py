# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Python Logging - https://rob-blackbourn.medium.com/how-to-use-python-logging-queuehandler-with-dictconfig-1e8b1284e27a
# --------------------------------------------------------

import os
import logging
from logging.handlers import QueueHandler, QueueListener
import datetime
import queue

def clear_logs(directory, prefix):
    if os.path.exists(directory):
        for fileName in os.listdir(directory):
            if fileName.startswith(prefix):
                os.remove(os.path.join(directory, fileName))

def logging_setup(directory, prefix, logger_name="logger"):
    clear_logs(directory, prefix)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    log_queue = queue.Queue(-1)
    queue_handler = QueueHandler(log_queue)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    os.makedirs(directory, exist_ok=True)
    log_path = os.path.join(directory, f'{prefix}{timestamp}.log')

    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    FORMAT = '%(asctime)s - %(filename)s - %(levelname)s - %(message)s'
    handler.setFormatter(logging.Formatter(FORMAT))

    logger.addHandler(queue_handler)
    logger.propagate=False
    
    listener = QueueListener(log_queue, handler)
    listener.start()

    logger.info("Logger was created")

    return logger, listener