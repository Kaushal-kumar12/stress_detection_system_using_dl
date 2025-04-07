import json
from django.contrib.auth.models import User
from django.db import models
from django.utils.timezone import now

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)  # Add this field
    mobile = models.CharField(max_length=15)
    dob = models.DateField()

    def __str__(self):
        return self.user.username


class EmotionModel(models.Model):
    # Add your fields here if needed, or just load the model
    # This can be adjusted if your emotion recognition model is stored differently
    name = models.CharField(max_length=255)
    model_file = models.FileField(upload_to='models/')  # Example field for model file

    @staticmethod
    def load_model(model_path):
        # Add logic here to load the pre-trained model (e.g., fer.h5)
        from tensorflow.keras.models import load_model
        return load_model(model_path)


def debug_upload_time(value):
    if value is None:
        return now()
    if not isinstance(value, str):
        print(f"Invalid upload_time value: {value} (type: {type(value)})")
    return value

class VideoUpload(models.Model):
    uploaded_video = models.FileField(upload_to='videos/uploads/', default='default_video_path')  # Provide a default value
    processed_video = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return f"Video uploaded on {self.upload_time}"