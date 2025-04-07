from django.urls import path
from . import views  # Import views correctly

urlpatterns = [
    path('', views.dashboard, name='admin_dashboard'),  # Ensure 'dashboard' exists in views.py
]
