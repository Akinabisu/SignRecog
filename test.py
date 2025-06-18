import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data_dir = 'data'
test_img_dir = os.path.join(data_dir, 'test')
model = tf.keras.models.load_model('model/gtsrb_model.keras')
class_names = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry",
    "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons"
]

test_image_path = os.path.join(test_img_dir, os.listdir(test_img_dir)[4])
img = tf.io.read_file(test_image_path)
img = tf.image.decode_png(img, channels=3)
img = tf.image.resize(img, [64, 64])
img = tf.cast(img, tf.float32) / 255.0
img = tf.expand_dims(img, axis=0)

prediction = model.predict(img)
predicted_class = class_names[np.argmax(prediction)]

print(f"Deducted class: {predicted_class}")

plt.imshow(tf.squeeze(img))
plt.title(f"Deducted class: {predicted_class}")
plt.axis("off")
plt.show()
