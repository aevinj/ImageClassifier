import tensorflow as tf
import matplotlib.pyplot as plt
import cv2, os

DATA_DIRECTORY = 'data'
IMG_EXTENSIONS = ['jpg', 'jpeg', 'bmp', 'png']

def get_file_extension(file_path):
    # Split the file path into its base name and extension
    base_name, extension = os.path.splitext(file_path)
    
    # Remove the leading dot (.) from the extension
    extension = extension.lstrip('.')
    
    return extension

for image_class in os.listdir(DATA_DIRECTORY): 
    for image in os.listdir(os.path.join(DATA_DIRECTORY, image_class)):
        image_path = os.path.join(DATA_DIRECTORY, image_class, image)
        try: 
            img = cv2.imread(image_path)
            ext = get_file_extension(image_path)
            if ext not in IMG_EXTENSIONS:
                os.remove(image_path)
        except Exception as e: 
            os.remove(image_path)