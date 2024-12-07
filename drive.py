#!/usr/bin/env python

import argparse
import base64
import time
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from io import BytesIO
from flask import Flask
from PIL import Image

from tensorflow.keras.models import load_model
from utils import resize_crop  # 导入 utils.py 中的 resize_crop 函数

# Fix error with Keras and TensorFlow
import tensorflow as tf

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


def process_image(image):
    """
    Process the image from the car's camera feed.

    :param image: Original image as a NumPy array.
    :return: Preprocessed image resized to (66, 200).
    """
    # Crop the region of interest (remove sky and car hood)
    image = image[35:135, :]  # Crop the image
    # Resize the cropped image using the function from utils
    image = resize_crop(image)
    return image


@sio.on('telemetry')
def telemetry(sid, data):
    print("Telemetry function called")
    # Extract telemetry data
    steering_angle = data["steering_angle"]  # Current steering angle
    throttle = data["throttle"]             # Current throttle
    speed = data["speed"]                   # Current speed
    image_path = data["image"]              # Base64 image data

    # Decode the image
    image = Image.open(BytesIO(base64.b64decode(image_path)))
    image_array = np.asarray(image)

    # Preprocess the image
    image_array = process_image(image_array)
    transformed_image_array = image_array[None, :, :, :]

    # Predict the steering angle using the trained model
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))

    # Counteract for model's bias towards 0 values
    steering_angle = steering_angle * 1

    # Control the throttle based on speed
    throttle = 0.25
    if float(speed) < 10:
        throttle = 1
        
    print("Received data from simulator: ", data)
    print('Steering Angle: %f, \t Throttle: %f' % (steering_angle, throttle))
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    """
    Callback for when the simulator connects to the server.
    """
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    """
    Send control commands to the simulator.
    """
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
                        help='Path to .keras model file.')
    args = parser.parse_args()

    # Load the model directly from .keras file
    # 加载模型
    model = load_model(args.model, compile=False, safe_mode=False)

    # 手动重新编译模型
    model.compile(optimizer="adam", loss="mse")

    # Wrap Flask application with socketio's middleware
    app = socketio.Middleware(sio, app)

    # Deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    
 

