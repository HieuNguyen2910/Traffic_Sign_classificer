from django.shortcuts import render
import os
import shutil
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
from django.conf import settings
from django.core.files.storage import FileSystemStorage


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
md = "C:/Users/ADMIN/Downloads/deep learning/Traffic_Sign_classificer.keras"
model_path = os.path.join(BASE_DIR, md)
lb = 'D:/python/Machine learning/Traffic_Sign_classificer/tsc/labels.csv'
labels_path = os.path.join(BASE_DIR, lb)

model = load_model(model_path)
df_labels = pd.read_csv(labels_path)
classes = df_labels['Name'].tolist()

IMG_SIZE = (224, 224)


def predict_image(request):
    if request.method == 'POST':
        file = request.FILES.get('image')
        
        if file:
            fs = FileSystemStorage()
            filename = fs.save(file.name, file)  
            uploaded_file_path = os.path.join(settings.MEDIA_ROOT, filename)

            img = Image.open(uploaded_file_path).convert('RGB')
            img = img.resize(IMG_SIZE)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            predicted_label = classes[predicted_class]

            uploaded_file_url = fs.url(filename)

            return render(request, 'result.html', {
                'predicted_label': predicted_label,
                'uploaded_file_url': uploaded_file_url
            })
    
    return render(request, 'upload.html')