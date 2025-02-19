from djoser.serializers import UserCreateSerializer, UserSerializer
from rest_framework import serializers


class UserCreateSer(UserCreateSerializer):
    class Meta(UserCreateSerializer.Meta):
        fields = ['id', 'username', 'password', 'email', 'first_name', 'last_name', 'phone_number']
        
        
        
class UserSer(UserSerializer):
    class Meta(UserSerializer.Meta):
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'phone_number']
        


class AudioFileSerializer(serializers.Serializer):
    audio = serializers.FileField()
