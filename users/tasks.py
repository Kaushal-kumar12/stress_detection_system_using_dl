# File: tasks.py
from celery import shared_task
from celery.utils.log import get_task_logger
import os
from .models import VideoUpload
from .utils import process_video  # Ensure process_video is moved to utils.py

logger = get_task_logger(__name__)

@shared_task
def process_video_task(uploaded_video_path, processed_video_path):
    try:
        detected_emotions = process_video(uploaded_video_path, processed_video_path, extract_emotions=True)

        # Save emotions back to the database
        video_instance = VideoUpload.objects.get(uploaded_video=uploaded_video_path)
        video_instance.processed_video = processed_video_path
        video_instance.emotions = {emotion: detected_emotions.count(emotion) for emotion in set(detected_emotions)}
        video_instance.save()

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise
