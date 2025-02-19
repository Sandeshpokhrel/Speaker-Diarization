from django.urls import path
from core.views import SpeakerDiarizationView

urlpatterns = [
    path('speaker_diarization/', SpeakerDiarizationView.as_view(), name='speaker_diarization'),
]