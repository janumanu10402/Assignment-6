from fastapi import FastAPI, UploadFile, File
from typing_extensions import Annotated
from keras.models import Sequential
from pydantic import  BaseModel
import keras
import numpy as np
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt

path = "./model.keras"

def load_model(path:str) -> Sequential: # helper function to load the .keras model
    model = keras.models.load_model(path)
    return model

def load_image_into_numpy_array(data): # helper function to convert image to np array
    data1 = BytesIO(data)
    return np.array(Image.open(data1))

def predict_digit(model:Sequential, data_point:list) -> str: # helper function that runs the model
    probs = model.predict(data_point, verbose=True)
    print("Predicted Digit:", np.argmax(probs))
    return str(np.argmax(probs))

def format_image(data) -> list: # helper function to format the received image (Task 2)
    data1 = BytesIO(data)
    image = Image.open(data1)
    image = image.resize((28,28)).convert("L") # resizes the image to 28*28 pixel and converts to grayscale
    return np.array(image)

app = FastAPI() # initialization of FastAPI module

@app.get("/") # test route to ensure server is running
async def root():
    return {"message": "Hello World"}

@app.post("/predict") # API endpoint for digit prediction, supports POST request
async def predict(image:UploadFile = File(...)): # handler function for the endpoint
    model = load_model(path) 
    formatted_image = format_image(await image.read())
    # plt.imshow(formatted_image, interpolation='nearest', cmap='gray', vmin=0, vmax=255)
    # plt.show()
    formatted_image = formatted_image.reshape(1,784) # serializing into a 1-D array 
    digit = predict_digit(model, formatted_image)
    return {"digit": digit}





