# Author: charlessodre
# Github: https://github.com/charlessodre/
# Create: 2019/07
# Info: Miscellaneous support functions


# import necessary packages
import os
import shutil
import glob
import time
from PIL import Image
from PIL import ImageOps


def get_files(files_path, extension_file='*'):
    """
    Return all files in directories.
    :param str files_path: path where are files.
    :param str extension_file: extension files.
    """
    list_files = []

    for root, dirs, files in os.walk(files_path):
        for file in glob.glob(os.path.join(root, extension_file)):
            list_files.append(file)
    return list_files


def file_move(source_path, dest_path):
    """
    Move file
    :param str source_path: path source files.
    :param str dest_path: path destiny files.
    """
    shutil.move(source_path, dest_path)


def file_copy(source_path, dest_path):
    """
    Copy file
    :param str source_path: path source files.
    :param str dest_path: path destiny files.
    """
    shutil.copy(source_path, dest_path)


def get_current_hour():
    """
    Return current hour.
    """
    return time.strftime("%H:%M:%S")


# format seconds to hh:mm:ss
def format_seconds_hhmmss(seconds):
    """
     Format seconds to hh:mm:ss.
    :param float seconds: seconds.

    """
    return time.strftime('%H:%M:%S', time.gmtime(seconds))


def resize_image(image, width, height=None):
    """
    Move file
    :param str image: path image to resize.
    :param int width: width size in pixels.
    :param int height: height size in pixels.
    """
    img = Image.open(image)

    if height is None:
        wpercent = (width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img_resized = img.resize((width, hsize), Image.ANTIALIAS)

    else:
        img_resized = img.resize((width, height), Image.ANTIALIAS)

    return img_resized
