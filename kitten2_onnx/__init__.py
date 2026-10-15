"""ONNX export experiments for KittenTTS 2."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("kitten2-onnx")
except PackageNotFoundError:
    __version__ = "0.1.dev0"

__all__ = ["__version__"]
