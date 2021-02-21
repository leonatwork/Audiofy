from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home page"),
    path('cam', views.cam, name="cam page"),
    path('camera', views.camVid, name="webcam page"),
    path('ebook', views.renderEbook, name="ebook page"),
    path('save_screenshot', views.screenShot, name="ss")
]
