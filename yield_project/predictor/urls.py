from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/predict/", views.api_predict, name="api_predict"),
    path("api/lookup/", views.api_lookup, name="api_lookup"),
    path("api/gee-status/", views.api_gee_status, name="api_gee_status"),
    path("api/history/", views.api_history, name="api_history"),
]
