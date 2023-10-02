from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.home, name='Homepage'),
    path('home/', views.home, name='Homepage')
]
