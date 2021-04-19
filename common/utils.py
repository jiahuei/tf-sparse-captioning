# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 23:38:59 2019

@author: jiahuei

Utility functions.

"""
import os
import math
import time
import logging
import requests
import tarfile
from zipfile import ZipFile
from typing import Union
from tqdm import tqdm
# from PIL import Image
from common.natural_sort import natural_keys

try:
    from natsort import natsorted, ns
except ImportError:
    natsorted = None

logger = logging.getLogger(__name__)
pjoin = os.path.join


# Image.MAX_IMAGE_PIXELS = None
# By default, PIL limit is around 89 Mpix (~ 9459 ** 2)


def configure_logging(
        logging_level: [int, str] = logging.INFO,
        logging_fmt: str = "%(levelname)s: %(name)s: %(message)s",
        logger_obj: Union[None, logging.Logger] = None,
) -> logging.Logger:
    """
    Setup logging on the root logger, because `transformers` calls `logger.info` upon import.

    Adapted from:
        https://stackoverflow.com/a/54366471/5825811
        Configures a simple console logger with the given level.
        A use-case is to change the formatting of the default handler of the root logger.

    Format variables:
        https://docs.python.org/3/library/logging.html#logrecord-attributes
    """
    logger_obj = logger_obj or logging.getLogger()  # either the given logger or the root logger
    # logger_obj.removeHandler(stanza.log_handler)
    logger_obj.handlers.clear()
    logger_obj.setLevel(logging_level)
    # If the logger has handlers, we configure the first one. Otherwise we add a handler and configure it
    if logger_obj.handlers:
        console = logger_obj.handlers[0]  # we assume the first handler is the one we want to configure
    else:
        console = logging.StreamHandler()
        logger_obj.addHandler(console)
    console.setFormatter(logging.Formatter(logging_fmt))
    console.setLevel(logging_level)
    # # Work around to update TensorFlow's absl.logging threshold which alters the
    # # default Python logging output behavior when present.
    # # see: https://github.com/abseil/abseil-py/issues/99
    # # and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
    # try:
    #     import absl.logging
    # except ImportError:
    #     pass
    # else:
    #     absl.logging.set_verbosity("info")
    #     absl.logging.set_stderrthreshold("info")
    #     absl.logging._warn_preinit_stderr = False
    return logger_obj


def maybe_download_from_url(url, dest_dir,
                            wget=True, wget_args=None,
                            file_name=None, file_size=None):
    """
    Downloads file from URL, streaming large files.
    """
    if wget_args is None:
        wget_args = []
    if file_name is None:
        file_name = url.split('/')[-1]
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    fpath = pjoin(dest_dir, file_name)
    if os.path.isfile(fpath):
        logger.info('Found file `{}`'.format(file_name))
        return fpath
    if wget:
        import subprocess
        subprocess.call(['wget', url] + wget_args, cwd=dest_dir)
    else:
        import requests
        response = requests.get(url, stream=True)
        chunk_size = 1024 ** 2  # 1 MB
        if response.ok:
            logger.info('Downloading `{}`'.format(file_name))
        else:
            print('ERROR: Download error. Server response: {}'.format(response))
            return False
        time.sleep(0.2)

        # Case-insensitive Dictionary of Response Headers.
        # The length of the request body in octets (8-bit bytes).
        try:
            file_size = int(response.headers['Content-Length'])
        except:
            pass
        if file_size is None:
            num_iters = None
        else:
            num_iters = math.ceil(file_size / chunk_size)
        tqdm_kwargs = dict(desc='Download progress',
                           total=num_iters,
                           unit='MB')
        with open(fpath, 'wb') as handle:
            for chunk in tqdm(response.iter_content(chunk_size), **tqdm_kwargs):
                if not chunk: break
                handle.write(chunk)
    logger.debug('Download complete: `{}`'.format(file_name))
    return fpath


def maybe_download_from_google_drive(id, fpath, file_size=None):
    URL = 'https://docs.google.com/uc?export=download'
    chunk_size = 1024 ** 2  # 1 MB
    fname = os.path.basename(fpath)
    out_path = os.path.split(fpath)[0]
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if os.path.isfile(fpath):
        logger.info('Found file `{}`'.format(fname))
        return fpath
    logger.info('Downloading `{}`'.format(fname))

    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    if file_size is not None:
        num_iters = math.ceil(file_size / chunk_size)
    else:
        num_iters = None
    tqdm_kwargs = dict(desc='Download progress',
                       total=num_iters,
                       unit='MB')
    with open(fpath, 'wb') as handle:
        for chunk in tqdm(response.iter_content(chunk_size), **tqdm_kwargs):
            if not chunk: break
            handle.write(chunk)
    logger.debug('Download complete: `{}`'.format(fname))
    return fpath


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def extract_tar_gz(fpath):
    """
    Extracts tar.gz file into the containing directory.
    """
    tar = tarfile.open(fpath, 'r')
    members = tar.getmembers()
    opath = os.path.split(fpath)[0]
    for m in tqdm(iterable=members,
                  total=len(members),
                  desc='Extracting `{}`'.format(os.path.split(fpath)[1])):
        tar.extract(member=m, path=opath)
    # tar.extractall(path=opath, members=progress(tar))   # members=None to extract all
    tar.close()


def extract_zip(fpath):
    """
    Extracts zip file into the containing directory.
    """
    with ZipFile(fpath, 'r') as zip_ref:
        for m in tqdm(
                iterable=zip_ref.namelist(),
                total=len(zip_ref.namelist()),
                desc='Extracting `{}`'.format(os.path.split(fpath)[1])):
            zip_ref.extract(member=m, path=os.path.split(fpath)[0])
        # zip_ref.extractall(os.path.split(fpath)[0])


def maybe_get_ckpt_file(net_params, remove_tar=True):
    """
    Download, extract, remove.
    """
    if os.path.isfile(net_params['ckpt_path']) or os.path.isfile(net_params['ckpt_path'] + '.index'):
        pass
    else:
        url = net_params['url']
        tar_gz_path = maybe_download_from_url(
            url, os.path.split(net_params['ckpt_path'])[0])
        extract_tar_gz(tar_gz_path)
        if remove_tar: os.remove(tar_gz_path)
