import matplotlib.pyplot as plt
from keras.models import model_from_json
import numpy as np
from keras.preprocessing.image import load_img
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
label = ['angry','disgust','fear','happy','neutral','sad','surprise']

def ef(image):
    img = load_img(image,color_mode = "grayscale")
    feature = np.array(img)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

image = 'images/train/sad/130.jpg'
print("original image is of sad")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is ",pred_label)
plt.imshow(img.reshape(48,48),cmap='gray')