import tensorflow as tf
import matplotlib.pyplot as plt
import cv2, os

class ImageClassifier:
    def __init__(self) -> None:
        self.DATA_DIRECTORY = 'data'
        self.IMG_EXTENSIONS = ['jpg', 'jpeg', 'bmp', 'png']

    def get_file_extension(self, file_path):
        # Split the file path into its base name and extension
        base_name, extension = os.path.splitext(file_path)
        
        # Remove the leading dot (.) from the extension
        extension = extension.lstrip('.')
        
        return extension

    def cleanData(self):
        for image_class in os.listdir(self.DATA_DIRECTORY): 
            for image in os.listdir(os.path.join(self.DATA_DIRECTORY, image_class)):
                image_path = os.path.join(self.DATA_DIRECTORY, image_class, image)
                try: 
                    img = cv2.imread(image_path)
                    ext = self.get_file_extension(image_path)
                    if ext not in self.IMG_EXTENSIONS:
                        os.remove(image_path)
                except Exception as e: 
                    os.remove(image_path)