from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home page"),
    path('camera', views.camVid, name="webcam page"),
    path('ebook', views.renderEbook, name="ebook page")
]
