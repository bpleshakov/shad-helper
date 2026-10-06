"""Local Apple Vision masks used only to compare visible lecture content."""

import importlib
import platform
import sys


def _native_modules():
    if sys.platform != "darwin":
        raise RuntimeError("Person-aware visual selection requires macOS 12 or later.")
    version = platform.mac_ver()[0]
    if not version or int(version.split(".")[0]) < 12:
        raise RuntimeError("Person-aware visual selection requires macOS 12 or later.")
    try:
        objc = importlib.import_module("objc")
        vision = importlib.import_module("Vision")
        quartz = importlib.import_module("Quartz")
        foundation = importlib.import_module("Foundation")
    except ImportError as error:
        raise RuntimeError(
            "Person-aware visual selection requires Apple Vision bindings. "
            "Install them with: uv sync --extra local"
        ) from error
    if not hasattr(vision, "VNGeneratePersonSegmentationRequest"):
        raise RuntimeError("Person-aware visual selection requires macOS 12 or later.")
    return objc, vision, quartz, foundation


def check_person_segmentation() -> None:
    """Check platform and bindings without running inference or loading a model."""
    _native_modules()


def _copy_mask(quartz, buffer, *, allow_empty=False):
    import numpy as np

    if (buffer is None or quartz.CVPixelBufferIsPlanar(buffer)
            or quartz.CVPixelBufferGetPixelFormatType(buffer) != quartz.kCVPixelFormatType_OneComponent8):
        return None
    flags = quartz.kCVPixelBufferLock_ReadOnly
    if quartz.CVPixelBufferLockBaseAddress(buffer, flags) != 0:
        return None
    try:
        width = quartz.CVPixelBufferGetWidth(buffer)
        height = quartz.CVPixelBufferGetHeight(buffer)
        stride = quartz.CVPixelBufferGetBytesPerRow(buffer)
        address = quartz.CVPixelBufferGetBaseAddress(buffer)
        if width <= 0 or height <= 0 or stride < width or address is None:
            return None
        # Copy before unlocking: a CoreVideo address is only valid while locked.
        data = np.frombuffer(address.as_buffer(height * stride), dtype=np.uint8)
        pixels = data.reshape(height, stride)[:, :width].copy()
    finally:
        quartz.CVPixelBufferUnlockBaseAddress(buffer, flags)
    # A successful empty mask can mean no person; full coverage remains uncertain.
    if (not allow_empty and not pixels.any()) or (pixels == 255).all():
        return None
    return pixels.astype(np.float32) / 255.0


class PersonSegmenter:
    """Reuse an accurate request within one video; return None on native failures."""

    def __init__(self, *, allow_empty_masks=False):
        self.allow_empty_masks = allow_empty_masks
        self.objc, self.vision, self.quartz, self.foundation = _native_modules()
        self.request = (
            self.vision.VNGeneratePersonSegmentationRequest.alloc()
            .initWithCompletionHandler_(None)
        )
        if self.request is None:
            raise RuntimeError("Apple Vision could not initialize person segmentation.")
        self.request.setRevision_(self.vision.VNGeneratePersonSegmentationRequestRevision1)
        self.request.setQualityLevel_(self.vision.VNGeneratePersonSegmentationRequestQualityLevelAccurate)
        self.request.setOutputPixelFormat_(self.quartz.kCVPixelFormatType_OneComponent8)

    def __call__(self, rgb):
        import cv2
        import numpy as np

        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
                or rgb.ndim != 3 or rgb.shape[2] != 3 or not all(rgb.shape[:2])):
            raise ValueError("Person segmentation expects a nonempty RGB uint8 image.")
        height, width = rgb.shape[:2]
        try:
            with self.objc.autorelease_pool():
                # NSData owns a copy, so Vision never reads a mutable NumPy buffer.
                rgba = np.empty((height, width, 4), dtype=np.uint8)
                rgba[:, :, :3], rgba[:, :, 3] = rgb, 255
                pixels = rgba.tobytes()
                data = self.foundation.NSData.dataWithBytes_length_(pixels, len(pixels))
                provider = self.quartz.CGDataProviderCreateWithCFData(data)
                image = self.quartz.CGImageCreate(
                    width, height, 8, 32, width * 4,
                    self.quartz.CGColorSpaceCreateDeviceRGB(),
                    self.quartz.kCGImageAlphaNoneSkipLast | self.quartz.kCGBitmapByteOrderDefault,
                    provider, None, False, self.quartz.kCGRenderingIntentDefault,
                )
                if image is None:
                    return None
                handler = self.vision.VNImageRequestHandler.alloc().initWithCGImage_orientation_options_(
                    image, self.quartz.kCGImagePropertyOrientationUp, {},
                )
                success, error = handler.performRequests_error_([self.request], None)
                results = self.request.results() if success and error is None else None
                if not results:
                    return None
                mask = _copy_mask(self.quartz, results[0].pixelBuffer(),
                                  allow_empty=self.allow_empty_masks)
                if mask is None:
                    return None
                # Image buffers remain top-down; no Core Image coordinate transform.
                return cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        except (self.objc.error, OSError, ValueError, TypeError, RuntimeError,
                AttributeError, BufferError, cv2.error):
            # Native request/conversion errors are uncertainty, not an empty mask.
            return None
