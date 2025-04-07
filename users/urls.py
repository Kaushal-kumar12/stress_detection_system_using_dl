from django.urls import path

from StressDetectionSystem import settings
from . import views
from .views import  UploadImageView, AccountView, LogoutView, upload_image
from django.conf.urls.static import static
from .views import user_account

urlpatterns = [
    path('', views.homepage, name='homepage'),  # Home page
    path('register/', views.register, name='register'),
    path('login/', views.login_view, name='login'),
    #path('register/', views.register, name='register'),
  #  path('login/', views.login, name='login'),
    path('user_dashboard/', views.user_dashboard, name='user_dashboard'),
   # path('livecam/', LiveCamView.as_view(), name='livecam'),
    path('upload_image/', UploadImageView.as_view(), name='upload_image'),
    #path('upload_video/', UploadVideoView.as_view(), name='upload_video'),
    path('account/', AccountView.as_view(), name='account'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('livecam/', views.livecam, name='livecam'),
    #path('upload/', views.detect_emotion, name='detect_emotion'),
     path('upload/', views.upload_image, name='upload_image'),
     path('upload-image/', upload_image, name='upload_image'),
    path('upload-video/', views.upload_video, name='upload_video'),
   #path("process-video/<str:video_name>/", views.process_uploaded_video, name="process_video"),
   # path('success/', views.success_page, name='success'),
   path('account/', user_account, name='user_account'),
] 
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)