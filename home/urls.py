from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('search', views.search, name='search'),
    path('files/<str:folder_id>/<str:file_name>', views.get_file, name='search'),
]