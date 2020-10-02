from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
import time
import os
from io import BytesIO
import urllib
import pathlib 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#Disable GPU and force TF to use CPU only
tf.config.set_visible_devices([], 'GPU')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class SudokuPredict():

    def __init__(self, file_name=None, version=None):
        self.version = version if version is not None else os.environ['SUDOKU_MODEL_VERSION']
        self.model_file_name = file_name if file_name is not None else os.environ['SUDOKU_MODEL_FILE_NAME']
        print(f"Model will be loaded: {self.model_file_name} with version: {self.version}")

        self.model = load_model('model/' + self.model_file_name)

    def __load_image_from_local__(self, filepath):
        # load the image from File
        img = load_img(filepath, color_mode = "grayscale", target_size=(28, 28))
        return img

    def __load_image_from_URL__(self, URL):
        # load the image from URL
        temp_filename = self.__generate_temp_filename__()

        # download image to a temp location
        filepath, _ = urllib.request.urlretrieve(URL, temp_filename)
        img = self.__load_image_from_local__(filepath)

        # remove temp image
        os.remove(temp_filename)
        return img
    
    def get_tensorflow_parameters(self):
        return tf.version.VERSION, tf.version.COMPILER_VERSION

    def __load_image_from_memory__(self, image):
        # load the image from ByteIO
        temp_filename = self.__generate_temp_filename__()

        pathlib.Path(temp_filename).write_bytes(image.getbuffer())
        img = self.__load_image_from_local__(temp_filename)

        os.remove(temp_filename)
        return img
    
    def __preprocess_image__(self, image):
        # convert to array
        img = img_to_array(image)
        # reshape into a single sample with 1 channel
        img = img.reshape(1, 28, 28, 1)
        # prepare pixel data
        img = img.astype('float32')
        # invert colors
        img = (255.0 - img)
        img = img / 255.0

        return img

    def __predict__(self, img):
        digit = self.model.predict_classes(img)
        
        if len(digit) > 0:
            return int(digit[0])
        else:
            raise Exception("Prediction Failed")

    def predict_local(self, filepath):
        img = self.__load_image_from_local__(filepath)
        img = self.__preprocess_image__(img)

        return self.__predict__(img)
    
    def predict_URL(self, URL):
        img = self.__load_image_from_URL__(URL)
        img = self.__preprocess_image__(img)

        return self.__predict__(img)
    
    def predict_memory(self, bytesio_image):
        img = self.__load_image_from_memory__(bytesio_image)
        img = self.__preprocess_image__(img)

        return self.__predict__(img)
    
    def __generate_temp_filename__(self):
        return 'tmp/' + str(int(time.time()*1000000)) + ".png"
    