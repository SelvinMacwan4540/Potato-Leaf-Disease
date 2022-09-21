from email.mime import image
from unittest import result
from fastapi import FastAPI, UploadFile, File  
from fastapi.responses import FileResponse 
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("/home/selvin/selvin/potato-disease-classification/models/model.h5")
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

# @app.get('/ping')
# async def ping():
#     return "Hello, i am alive!"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))    # read image into numpy array
    img_batch = np.expand_dims(image, 0)             # in model we have inputshape with batches of images [[],[],[]]
    predictions = MODEL.predict(img_batch)       
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)
    return {
        'class' : predicted_class,
        'confidence' : float(confidence)    
    }


@app.get('/')
async def result():
    return FileResponse('./home.html')

@app.post('/predict')
async def predict(file:UploadFile):   # variable name : data dtype ,   var : int
    result = read_file_as_image(await file.read())
    return result


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port= 8080)