from django.urls import path
from . import views
from django.conf.urls import url


urlpatterns = [
    path('', views.home, name='home'),
    path('text', views.text, name='text'),
    path('image', views.image, name='image'),
    path('speech', views.speech, name='speech'),
    path('text1', views.text1, name='text1'),
    path('image1', views.image1, name='image1'),
    path('speech1', views.speech1, name='speech1'),
]

