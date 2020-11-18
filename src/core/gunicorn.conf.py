# coding=utf-8
import os
import multiprocessing

_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), ''))
_VAR = os.path.join(_ROOT, 'var')
_ETC = os.path.join(_ROOT, 'etc')

loglevel = 'info'

bind = '0.0.0.0:7778'
workers = 1
timeout = 30  # 30 seconds

