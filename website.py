import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = tf.keras.models.load_model('model/gtsrb_model.keras')

class_names = [
    "Обмеження швидкості (20км/год)", "Обмеження швидкості (30км/год)", "Обмеження швидкості (50км/год)",
    "Обмеження швидкості (60км/год)", "Обмеження швидкості (70км/год)", "Обмеження швидкості (80км/год)",
    "Кінець обмеження швидкості (20км/год)", "Кінець обмеження швидкості (100км/год)", "Обмеження швидкості (120км/год)",
    "Обгін заборонено", "Обгін заборонено для вантажівок",
    "Перехрестя з другорядною дорогою", "Пріоритетна дорога", "Дати дорогу", "Стоп",
    "Проїзд заборонено", "Проїзд вантажівок заборонено", "В'їзд заборонено",
    "Загальне застереження", "Небезпечний поворот ліворуч", "Небезпечний поворот праворуч",
    "Подвійний поворот", "Нерівна дорога", "Слизька дорога", "Звуження справа",
    "Дорожні роботи", "Світлофор", "Обережно пішоходи", "Обережно діти",
    "Велосипеди", "Обережно сніг", "Обережно дикі тварини",
    "Кінець обмежень швидкості та обгону", "Поворот направо", "Поворот наліво",
    "Тільки прямо", "Прямо або направо", "Прямо або наліво", "Триматися праворуч",
    "Триматися ліворуч", "Рух по колу", "Кінець обмеження обгону",
    "Кінець обмеження обгону вантажівками"
]

def predict_label(img_path):
    image = Image.open(img_path).convert("RGB").resize((64, 64))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    class_idx = np.argmax(prediction)
    return class_names[class_idx]

@app.route('/', methods=['GET', 'POST'])


def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            label = predict_label(filepath)
            return render_template('index.html', filename=filename, label=label)
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
