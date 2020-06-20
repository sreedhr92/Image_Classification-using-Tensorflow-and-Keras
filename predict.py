import tensorflow as tf
import tensorflow_hub as hub
import json
import argparse
import numpy as np
from PIL import Image
from utilities import process_image,class_names_val,load_model

def predict(image_path,model_path,top_k,class_name):
    model = load_model(model_path)
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image,axis=0)
    prob = model.predict(image)
    prob = prob[0].tolist()
    keys,values = tf.math.top_k(prob,int(top_k))
    keys = keys.numpy().tolist()
    values = values.numpy().tolist()
    print("The class with highest probablity is",class_name[str(values[keys.index(max(keys))])])
    print('The top {:,} classes are'.format(int(top_k)))
    print('prob =',keys)
    print('class =',values)
    class_names = [class_name[str(i)] for i in values]
    display_value = dict(zip(class_names,values))
    print(display_value)
    
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = " parser")
    parser.add_argument("image_path",help="Path of the image", default="")
    parser.add_argument("saved_model",help="Path of the model", default="")
    parser.add_argument("--top_k", help="Fetch top k predictions", required = False, default = 1)
    parser.add_argument("--category_names", help="Class map json file", required = False, default = "label_map.json")
    args = parser.parse_args()

    class_name = class_names_val(args.category_names)

    predict(args.image_path, args.saved_model, args.top_k, class_name)