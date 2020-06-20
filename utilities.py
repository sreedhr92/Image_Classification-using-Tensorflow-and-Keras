import tensorflow as tf
import tensorflow_hub as hub
import json

def class_names_val(file):
	with open(file,'r') as f:
		class_names = json.load(f)
	class_name_rectified = dict()
	for key in class_names:
		class_name_rectified[str(int(key)-1)] = class_names[key]
	return class_name_rectified

def load_model(path):
	model = tf.keras.models.load_model(path,custom_objects={'KerasLayer':hub.KerasLayer})
	return model

def process_image(image):
	image = tf.convert_to_tensor(image)
	image = tf.image.resize(image,(224,224)).numpy()
	image/=255
	return image
	