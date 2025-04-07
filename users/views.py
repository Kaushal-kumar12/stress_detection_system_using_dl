import base64
import json
import os
import time
from urllib import request
from celery import shared_task
import numpy as np
from io import BytesIO
from PIL import Image
import uuid
import subprocess
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.shortcuts import get_object_or_404, render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.models import User
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
import cv2
#from .models import VideoUpload
from keras.models import model_from_json
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from StressDetectionSystem import settings
from .forms import UserRegistrationForm, VideoUploadForm
from .models import UserProfile, EmotionModel, VideoUpload
from .tasks import process_video_task



# Views
def homepage(request):
    if request.user.is_authenticated:
        return redirect('user_dashboard')
    return render(request, 'homepage.html')

def register(request):
    if request.user.is_authenticated:
        return redirect('user_dashboard')

    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']

            # Check if username exists
            existing_user = User.objects.filter(username=username).first()
            if existing_user:
                if existing_user.is_active:
                    messages.error(request, 'Username already exists. Please choose a different one.')
                    return render(request, 'register.html', {'form': form})
                else:
                    # Reactivate the user
                    existing_user.is_active = True
                    existing_user.set_password(password)  # Hash new password
                    existing_user.save()
                    messages.success(request, 'Account reactivated. You can now log in.')
                    return redirect('login')

            user = form.save()
            login(request, user)  # Auto-login after registration
            messages.success(request, 'Registration successful! You are now logged in.')
            return redirect('user_dashboard')

    else:
        form = UserRegistrationForm()

    return render(request, 'register.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('user_dashboard')

    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)

            if user:
                if not user.is_active:
                    messages.error(request, 'This account is inactive. Please register again or contact support.')
                    return redirect('register')
                login(request, user)
                messages.success(request, 'Login successful!')
                return redirect('user_dashboard')
            else:
                messages.error(request, 'Invalid username or password.')
        else:
            messages.error(request, 'Please correct the errors below.')

    else:
        form = AuthenticationForm()

    return render(request, 'login.html', {'form': form})

@login_required
def user_dashboard(request):
    if not request.user.is_authenticated:
        return redirect('login')
    return render(request, 'user_dashboard.html')


# Load pre-trained facial emotion recognition model
model = model_from_json(open("Facial Expression Recognition.json", "r").read())
model.load_weights("fer.h5")

# Define emotion labels
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Load Haarcascade Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion detection function
def detect_emotion_from_image(img: Image.Image):
    try:
        img = img.convert('RGB')
        img = np.array(img)
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Ensure correct grayscale conversion

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            x, y, w, h = faces[0]  # Process first detected face
            roi_gray = gray_img[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))

            # Prepare the image for the model
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255.0  # Normalize pixel values

            # Predict emotion
            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            predicted_emotion = emotions[max_index]
            return predicted_emotion
        else:
            return 'No face detected'
    except Exception as e:
        return str(e)

# LiveCam View
@csrf_exempt
def livecam(request):
    if request.method == "POST":
        try:
            image_data = json.loads(request.body).get('image', '')
            img = Image.open(BytesIO(base64.b64decode(image_data.split(',')[1])))
            emotion = detect_emotion_from_image(img)
            return JsonResponse({'emotion': emotion})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return render(request, 'livecam.html')

class UploadImageView(LoginRequiredMixin, View):
    def get(self, request):
        return render(request, 'upload_image.html')

    def post(self, request):
        uploaded_file = request.FILES.get('image')
        if uploaded_file:
            pass  # Add logic to save or process the file
        return redirect('user_dashboard')
    
    # View for image upload and emotion detection
def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']

        # Save uploaded image
        original_path = os.path.join(settings.MEDIA_ROOT, 'uploads', uploaded_image.name)
        with open(original_path, 'wb+') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)

        # Process the image
        emotion, processed_path = process_uploaded_image(original_path)

        # Generate URLs for templates
        original_image_url = f"{settings.MEDIA_URL}uploads/{uploaded_image.name}"
        processed_image_url = f"{settings.MEDIA_URL}processed/{uploaded_image.name}"

        context = {
            'emotion': emotion,
            'original_image_url': original_image_url,
            'processed_image_url': processed_image_url,
        }
        return render(request, 'upload_result.html', context)

    return render(request, 'upload_image.html')


def process_uploaded_image(image_path):
    import cv2
    import numpy as np
    from tensorflow.keras.models import model_from_json

    # Load the uploaded image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the face detector
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Load the pre-trained emotion recognition model
    model = model_from_json(open("Facial Expression Recognition.json", "r").read())
    model.load_weights("fer.h5")

    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    detected_emotion = "No face detected"
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Draw rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Preprocess face for prediction
            face_gray = cv2.resize(cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY), (48, 48))
            face_normalized = face_gray / 255.0
            face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))

            # Predict emotion
            predictions = model.predict(face_reshaped)
            emotion_index = np.argmax(predictions)
            detected_emotion = emotions[emotion_index]

            # Add text label
            cv2.putText(img, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Save the processed image
    processed_image_path = os.path.join(settings.MEDIA_ROOT, 'processed', os.path.basename(image_path))
    os.makedirs(os.path.dirname(processed_image_path), exist_ok=True)
    cv2.imwrite(processed_image_path, img)

    return detected_emotion, processed_image_path

"""# Load the emotion recognition model globally
def load_emotion_model():
    model = model_from_json(open("Facial Expression Recognition.json", "r").read())
    model.load_weights("fer.h5")
    return model

emotion_model = load_emotion_model()
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
"""

import os
import cv2
import numpy as np
import subprocess
from django.shortcuts import render
from django.conf import settings
from keras.models import model_from_json

def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        uploaded_video = request.FILES['video']

        # Define directories
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        processed_dir = os.path.join(settings.MEDIA_ROOT, 'processed')

        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        # Save uploaded video
        uploaded_video_path = os.path.join(upload_dir, uploaded_video.name)
        with open(uploaded_video_path, 'wb+') as destination:
            for chunk in uploaded_video.chunks():
                destination.write(chunk)

        # Process video & get emotion data
        processed_video_path, emotions_data = process_uploaded_video(uploaded_video_path, processed_dir)

        # Generate URLs for template
        original_video_url = f"{settings.MEDIA_URL}uploads/{uploaded_video.name}"
        processed_video_url = f"{settings.MEDIA_URL}processed/{os.path.basename(processed_video_path)}"

        context = {
            'original_video_url': original_video_url,
            'processed_video_url': processed_video_url,
            'emotions_data': emotions_data,  # Pass emotions data
        }
        return render(request, 'video_result.html', context)

    return render(request, 'upload_video.html')


def process_uploaded_video(video_path, processed_dir):
    # Load model
    model = model_from_json(open("Facial Expression Recognition.json", "r").read())
    model.load_weights("fer.h5")

    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define output path
    processed_video_path = os.path.join(processed_dir, "processed_" + os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (frame_width, frame_height))

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    emotions_data = []  # Store timestamped emotions

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        detected_emotion = "No face detected"
        for (x, y, w, h) in faces:
            face_gray = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
            face_normalized = face_gray / 255.0
            face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))

            predictions = model.predict(face_reshaped)
            emotion_index = np.argmax(predictions)
            detected_emotion = emotions[emotion_index]

            # Draw rectangle and label on frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        out.write(frame)

        # Store timestamped emotion
        timestamp = frame_count / fps  # Convert frame count to seconds
        emotions_data.append({"time": round(timestamp, 2), "emotion": detected_emotion})

        frame_count += 1

    cap.release()
    out.release()
    
    return processed_video_path, emotions_data  # Return processed video and emotion data


def convert_to_h264(video_path):
    h264_video_path = video_path.replace('.mp4', '_h264.mp4')
    ffmpeg_cmd = [
        'ffmpeg', '-i', video_path, '-c:v', 'libx264', '-preset', 'slow', '-crf', '23', '-c:a', 'aac', '-b:a', '128k', h264_video_path
    ]
    subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return h264_video_path



def success_page(request):
    return render(request, 'success.html')

class AccountView(LoginRequiredMixin, View):
    def get(self, request):
        return render(request, 'user_account.html')

class LogoutView(LoginRequiredMixin, View):
    def get(self, request):
        logout(request)
        return redirect('login')
    
@login_required
def user_account(request):
    return render(request, 'user_account.html', {'user': request.user})    