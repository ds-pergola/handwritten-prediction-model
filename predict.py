from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
import time
import os
from io import BytesIO
import urllib
import pathlib 
import pymongo
from datetime import datetime
from dotenv import load_dotenv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#Disable GPU and force TF to use CPU only
tf.config.set_visible_devices([], 'GPU')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

load_dotenv(verbose=True, override=True)

class SudokuPredict():

    def __init__(self, file_name=None, version=None):
        self.version = version if version is not None else os.environ['SUDOKU_MODEL_VERSION']
        self.model_file_name = file_name if file_name is not None else os.environ['SUDOKU_MODEL_FILE_NAME']
        print(f"Model will be loaded: {self.model_file_name} with version: {self.version}")

        self.model = load_model('model/' + self.model_file_name)

        mongo_client = pymongo.MongoClient(os.getenv('MONGO_URI'))
        self.mongo_db = mongo_client.sudoku

        self.log_doc = {}

    def __load_image_from_local(self, filepath):
        # load the image from File
        img = load_img(filepath, color_mode = "grayscale", target_size=(28, 28))
        self.log_doc["loaded_img_colormode"] = "grayscale"
        self.log_doc["loaded_img_target_size"] = (28, 28)
        return img

    def __load_image_from_URL(self, URL):
        # load the image from URL
        temp_filename = self.__generate_temp_filename()

        # download image to a temp location
        filepath, _ = urllib.request.urlretrieve(URL, temp_filename)
        img = self.__load_image_from_local(filepath)

        # remove temp image
        os.remove(temp_filename)
        return img
    
    def get_tensorflow_parameters(self):
        return tf.version.VERSION, tf.version.COMPILER_VERSION

    def __load_image_from_memory(self, image):
        # load the image from ByteIO
        temp_filename = self.__generate_temp_filename()

        pathlib.Path(temp_filename).write_bytes(image.getbuffer())
        img = self.__load_image_from_local(temp_filename)
        
        os.remove(temp_filename)
        return img
    
    def __preprocess_image(self, image):
        # convert to array
        img = img_to_array(image)
        self.log_doc["img_to_array"] = img.tolist()

        # reshape into a single sample with 1 channel
        img = img.reshape(1, 28, 28, 1)
        # prepare pixel data
        img = img.astype('float32')
        # invert colors
        img = (255.0 - img)
        img = img / 255.0
        self.log_doc["post_process"] = img.tolist()

        return img

    def __predict(self, img):
        result = self.model.predict(img)[0].tolist()
        self.log_doc["predict_result"] = result

        prob = max(result)
        self.log_doc["predict_prob"] = float(prob)

        digit = result.index(prob)
        self.log_doc["predict_digit"] = int(digit)

        #digit = self.model.predict_classes(img)
        
        return (int(digit), float(prob))

    def predict_local(self, filepath):
        img = self.__load_image_from_local(filepath)
        img = self.__preprocess_image(img)

        return self.__predict(img)
    
    def predict_URL(self, URL):
        img = self.__load_image_from_URL(URL)
        img = self.__preprocess_image(img)

        return self.__predict(img)
    
    def predict_memory(self, bytesio_image, transaction_id=None):
        self.__log_init(transaction_id)

        img = self.__load_image_from_memory(bytesio_image)
        img = self.__preprocess_image(img)
        result = self.__predict(img)

        self.__save_log()
        return result
    
    def __generate_temp_filename(self):
        return 'tmp/' + str(int(time.time()*1000000)) + ".png"
    
    def __save_log(self):
        self.mongo_db.predictions_log.insert_one(self.log_doc)
        pass
    
    def __log_init(self, transaction_id):
        self.log_doc = {}
        self.log_doc["transaction_id"] = transaction_id
        self.log_doc["model_file_name"] = self.model_file_name
        self.log_doc["version"] = self.version
        self.log_doc["datetime"] = datetime.utcnow()