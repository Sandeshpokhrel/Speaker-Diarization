#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'diarization.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    # from core.logic.diarization.inference import speaker_diarization
    # from core.logic.diarization.visualize import diarization_result, voice_activity

    # output = speaker_diarization('core/logic/diarization/user_input/user_0')
    # for i in output:
    #     i.append('text')
    #     print(i)

    # audio_file_path = 'core/logic/diarization/user_input/user_0/output_voice3.flac'
    # rttm_file_path = 'core/logic/diarization/user_input/user_0/output_voice3.rttm'

    # voice_activity(audio_file_path)
    # diarization_result(audio_file_path, rttm_file_path)
    main()