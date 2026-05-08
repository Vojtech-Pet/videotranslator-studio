"""
video_translator — core package for VideoTranslator pipeline.

Modules:
  stt             - Speech-to-text (Whisper, VAD, segment processing)
  translation     - Translation (Llama, MADLAD, API, phonetic respelling, SRT)
  tts             - Text-to-speech (Chatterbox, Piper, API, ONNX, QC)
  audio_pipeline  - Audio assembly, post-processing, music separation
  config          - ArgParser, constants, path configuration
  pipeline        - Main orchestration (entry point logic)
"""
