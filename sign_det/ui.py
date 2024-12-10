from PIL import Image
import cv2
import os
from SOURCE.yolo_files import detect
from helper_fns import gan_utils
import shutil
import glob

# Define constants for directories
MEDIA_ROOT = 'media/documents/'
SIGNATURE_ROOT = 'media/UserSignaturesSquare/'
YOLO_RESULT = 'results/yolov5/'
YOLO_OP = 'crops/DLSignature/'
GAN_IPS = 'results/gan/gan_signdata_kaggle/gan_ips/testB'
GAN_OP = 'results/gan/gan_signdata_kaggle/test_latest/images/'
GAN_OP_RESIZED = 'results/gan/gan_signdata_kaggle/test_latest/images/'


def select_cleaned_image(selection):
    ''' Returns the path of cleaned image corresponding to the document the user selected '''
    return GAN_OP + selection + '_fake.png'

def copy_and_overwrite(from_path, to_path):
    ''' Copies files from YOLO ops to GAN input folder, resetting as needed '''
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)

def signature_detection():
    ''' Performs signature detection and returns the path to the detected results folder '''
    detect.detect(MEDIA_ROOT)
    latest_detection = max(glob.glob(os.path.join(YOLO_RESULT, '*/')), key=os.path.getmtime)
    gan_utils.resize_images(os.path.join(latest_detection, YOLO_OP))
    return latest_detection + YOLO_OP


if __name__ == "__main__":
    print ('Start')
    print ('Loading...')
    signature_detection()
    print ('Done')
