from django.contrib import admin
from django.urls import path, include, re_path
from . import views
from .views import login_view, register_user
from django.contrib.auth.views import LogoutView
from .views import create


urlpatterns = [

    # The home page
    path('', views.home, name='home'),
    # Matches any html file
    path('start_rc', views.start_rc, name='start_rc'),
    path('stream', views.streamer, name='stream'),
    path('detectme', views.detectme, name='detectme'),
    path('Song_list', views.Song_list, name='Song_list'),
    
    path('create', views.create, name='create'),
    path('login/', views.login_view, name="login"),
    path('register/', views.register_user, name="register"),
    path("logout/", LogoutView.as_view(), name="logout"),
    re_path(r'^.*\.*', views.pages, name='pages'),

]


