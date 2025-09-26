# -*- coding: utf-8 -*-
# @Time    : 2025/9/23 19:16
# @Author  : chengxiang.luo
# @Email   : chengxiang.luo@foxmail.com
# @File    : rxn_logger.py
# @Software: PyCharm
import logging
import re
from logging import handlers

logfile = 'rxn-yield-predictor.log'
fmt = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger(logfile)
logger.setLevel(logging.INFO)
logger_handler = handlers.TimedRotatingFileHandler(filename=logfile, when='D',
                                                   backupCount=30, encoding='utf-8',
                                                   interval=1)
logger_handler.setFormatter(fmt)

logger_handler.suffix = '%Y%m%d'
logger_handler.extMatch = re.compile(r"^\d{8}$")
logger.addHandler(logger_handler)
