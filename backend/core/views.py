from rest_framework.generics import CreateAPIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
import os
import shutil
from django.core.files.storage import default_storage

from core.logic.diarization.inference import speaker_diarization
from core.serializers import AudioFileSerializer


class SpeakerDiarizationView(CreateAPIView):
    serializer_class = AudioFileSerializer
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        audio_dir = 'core/logic/diarization/user_input/user_0'
        os.makedirs(audio_dir, exist_ok=True)

        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        file_obj = serializer.validated_data['audio']
        file_path = os.path.join(audio_dir, file_obj.name)

        try:
            with default_storage.open(file_path, 'wb+') as destination:
                for chunk in file_obj.chunks():
                    destination.write(chunk)

            output = speaker_diarization(audio_dir)
            result = [i + ['text'] for i in output]  

            return Response({'diarization_result': result}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        finally:
            shutil.rmtree(audio_dir)
            os.makedirs(audio_dir, exist_ok=True)
