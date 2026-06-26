"""Import compatibility stubs for the RoboMeter eval server.

The RoboMeter server imports sentence_transformers through evaluation sampler side
effects even though progress serving does not use TorchCodec. Some shared
training environments have sentence_transformers installed with torchcodec but
without compatible FFmpeg libraries. This stub lets that optional import finish;
using AudioDecoder or VideoDecoder will still fail loudly.
"""
from __future__ import annotations

import importlib.machinery
import sys
import types


if "torchcodec" not in sys.modules:
    torchcodec = types.ModuleType("torchcodec")
    torchcodec.__spec__ = importlib.machinery.ModuleSpec("torchcodec", loader=None)
    decoders = types.ModuleType("torchcodec.decoders")
    decoders.__spec__ = importlib.machinery.ModuleSpec("torchcodec.decoders", loader=None)

    class AudioDecoder:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("TorchCodec is unavailable in this RoboMeter server environment")

    class VideoDecoder:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("TorchCodec is unavailable in this RoboMeter server environment")

    decoders.AudioDecoder = AudioDecoder
    decoders.VideoDecoder = VideoDecoder
    torchcodec.decoders = decoders
    sys.modules["torchcodec"] = torchcodec
    sys.modules["torchcodec.decoders"] = decoders
