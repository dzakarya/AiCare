import cv2
import numpy as np


def ImgToBin(img):
    # Convert digital data to binary format
    imgstr = cv2.imencode('.jpg',img)[1].tostring()
    return imgstr


def BinToImg(data):
    # Convert binary data to proper format and write it on Hard Disk
    nparr = np.fromstring(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img