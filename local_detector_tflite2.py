import argparse
import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf

from utils.tools import *
from utils.visualize import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", default="examples/3.JPEG", help="the model input")
    parser.add_argument(
        "--dest_folder", default="out/", help="folder to store the results")
    parser.add_argument(
        "--model_path", default="weights/modelphotoshop1.tflite", help="path to the tflite model")
    parser.add_argument(
        "--gpu_id", default='0', help="the id of the gpu to run model on")
    parser.add_argument(
        "--no_crop",
        action="store_true",
        help="do not use a face detector, instead run on the full input image")
    args = parser.parse_args()

    img_path = args.input_path
    dest_folder = args.dest_folder
    model_path = args.model_path
    gpu_id = args.gpu_id

    # Loading the model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Getting input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Loading the image
    im_w, im_h = Image.open(img_path).size
    if args.no_crop:
        face = Image.open(img_path).convert('RGB')
    else:
        faces = face_detection(img_path, verbose=False)
        if len(faces) == 0:
            print("no face detected by dlib, exiting")
            sys.exit()
        face, box = faces[0]
    face = resize_shorter_side(face, 400)[0]
    face_np = np.array(face).astype(np.float32)

    # Data preprocessing
    input_shape = input_details[0]['shape']
    input_data = np.expand_dims(face_np, axis=0)
    input_data = tf.image.resize(input_data, input_shape[2:4])
    # input_data = tf.image.resize(input_data, (400,448))
    input_data = (input_data - 127.5) / 127.5
    input_data = np.transpose(input_data, (0, 3, 1, 2))

    # Warping field prediction
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    flow = np.transpose(output_data.squeeze(), (1, 2, 0))
    h, w, _ = flow.shape

    # Undoing the warps
    modified = face.resize((w, h), Image.BICUBIC)
    modified_np = np.asarray(modified)
    reverse_np = warp(modified_np, flow)
    reverse = Image.fromarray(reverse_np)

    # Saving the results
    modified.save(
        os.path.join(dest_folder, 'cropped_input15.jpg'),
        quality=90)
    reverse.save(
        os.path.join(dest_folder, 'warped15.jpg'),
        quality=90)
    flow_magn = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    save_heatmap_cv(
        modified_np, flow_magn,
        os.path.join(dest_folder, 'heatmap15.jpg'))
