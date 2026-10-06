import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from shad_helper.person_segmentation import PersonSegmenter, check_person_segmentation


class NativeHarness:
    def __init__(self):
        self.entered = self.exited = 0
        self.buffer = bytearray([0, 255, 99, 99, 127, 64, 99, 99])
        self.address = SimpleNamespace(as_buffer=lambda count: memoryview(self.buffer)[:count])
        self.request = SimpleNamespace(
            setRevision_=Mock(), setQualityLevel_=Mock(), setOutputPixelFormat_=Mock(),
            results=Mock(return_value=[SimpleNamespace(pixelBuffer=lambda: self.buffer)]),
        )
        self.request_init = Mock(return_value=self.request)
        self.request_alloc = Mock(return_value=SimpleNamespace(
            initWithCompletionHandler_=self.request_init))
        self.handler = SimpleNamespace(performRequests_error_=Mock(return_value=(True, None)))
        self.handler_init = Mock(return_value=self.handler)
        self.vision = SimpleNamespace(
            VNGeneratePersonSegmentationRequest=SimpleNamespace(alloc=self.request_alloc),
            VNGeneratePersonSegmentationRequestRevision1=1,
            VNGeneratePersonSegmentationRequestQualityLevelAccurate=0,
            VNImageRequestHandler=SimpleNamespace(alloc=lambda: SimpleNamespace(
                initWithCGImage_orientation_options_=self.handler_init)),
        )

        @contextlib.contextmanager
        def pool():
            self.entered += 1
            try:
                yield
            finally:
                self.exited += 1

        self.objc = SimpleNamespace(autorelease_pool=pool, error=RuntimeError)
        self.foundation = SimpleNamespace(NSData=SimpleNamespace(
            dataWithBytes_length_=lambda data, length: bytes(data[:length])))
        self.quartz = SimpleNamespace(
            kCVPixelFormatType_OneComponent8=1, kCVPixelBufferLock_ReadOnly=2,
            kCGImageAlphaNoneSkipLast=3, kCGBitmapByteOrderDefault=0,
            kCGRenderingIntentDefault=0, kCGImagePropertyOrientationUp=1,
            CGDataProviderCreateWithCFData=lambda data: data,
            CGColorSpaceCreateDeviceRGB=lambda: "RGB",
            CGImageCreate=Mock(return_value="CGImage"),
            CVPixelBufferIsPlanar=lambda buffer: False,
            CVPixelBufferGetPixelFormatType=lambda buffer: 1,
            CVPixelBufferLockBaseAddress=Mock(return_value=0),
            CVPixelBufferUnlockBaseAddress=Mock(return_value=0),
            CVPixelBufferGetWidth=lambda buffer: 2,
            CVPixelBufferGetHeight=lambda buffer: 2,
            CVPixelBufferGetBytesPerRow=lambda buffer: 4,
            CVPixelBufferGetBaseAddress=lambda buffer: self.address,
        )

    def modules(self):
        return self.objc, self.vision, self.quartz, self.foundation


class PersonSegmentationTests(unittest.TestCase):
    def test_preflight_checks_bindings_without_initializing_a_request(self):
        native = NativeHarness()
        modules = dict(zip(("objc", "Vision", "Quartz", "Foundation"), native.modules()))
        with patch("shad_helper.person_segmentation.sys.platform", "darwin"), \
                patch("shad_helper.person_segmentation.platform.mac_ver", return_value=("26.4", (), "")), \
                patch("shad_helper.person_segmentation.importlib.import_module", side_effect=modules.__getitem__):
            check_person_segmentation()
        native.request_alloc.assert_not_called()

    def test_missing_bindings_have_actionable_setup_error(self):
        with patch("shad_helper.person_segmentation.sys.platform", "darwin"), \
                patch("shad_helper.person_segmentation.platform.mac_ver", return_value=("26.4", (), "")), \
                patch("shad_helper.person_segmentation.importlib.import_module", side_effect=ImportError), \
                self.assertRaisesRegex(RuntimeError, "uv sync --extra local"):
            check_person_segmentation()

    def test_unsupported_platform_or_os_fails_before_importing_bindings(self):
        for system, version in (("linux", ""), ("darwin", "11.7")):
            with self.subTest(system=system, version=version), \
                    patch("shad_helper.person_segmentation.sys.platform", system), \
                    patch("shad_helper.person_segmentation.platform.mac_ver", return_value=(version, (), "")), \
                    patch("shad_helper.person_segmentation.importlib.import_module") as imports:
                with self.assertRaisesRegex(RuntimeError, "macOS 12"):
                    check_person_segmentation()
                imports.assert_not_called()

    def test_rgb_channels_stride_orientation_and_copied_mask(self):
        native = NativeHarness()
        rgb = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [10, 20, 30]]], dtype=np.uint8)

        def unlock(*args):
            native.buffer[:] = bytes(len(native.buffer))
            return 0

        native.quartz.CVPixelBufferUnlockBaseAddress.side_effect = unlock
        with patch("shad_helper.person_segmentation._native_modules", return_value=native.modules()):
            segmenter = PersonSegmenter()
            mask = segmenter(rgb)
        native.request.setRevision_.assert_called_once_with(1)
        native.request_init.assert_called_once_with(None)
        native.request.setQualityLevel_.assert_called_once_with(0)
        native.request.setOutputPixelFormat_.assert_called_once_with(1)
        args = native.quartz.CGImageCreate.call_args.args
        self.assertEqual(args[:5], (2, 2, 8, 32, 8))
        self.assertEqual(args[7], bytes([255, 0, 0, 255, 0, 255, 0, 255,
                                       0, 0, 255, 255, 10, 20, 30, 255]))
        native.handler_init.assert_called_once_with("CGImage", 1, {})
        np.testing.assert_allclose(mask, np.array([[0, 255], [127, 64]], dtype=np.float32) / 255)
        self.assertEqual(mask.dtype, np.float32)
        native.quartz.CVPixelBufferLockBaseAddress.assert_called_once()
        native.quartz.CVPixelBufferUnlockBaseAddress.assert_called_once()
        self.assertEqual((native.entered, native.exited), (1, 1))

    def test_noncontiguous_input_and_mask_resize_reuse_one_request(self):
        native = NativeHarness()
        rgb = np.zeros((3, 8, 3), dtype=np.uint8)[:, ::2, :]
        self.assertFalse(rgb.flags.c_contiguous)
        with patch("shad_helper.person_segmentation._native_modules", return_value=native.modules()):
            segmenter = PersonSegmenter()
            for _ in range(2):
                mask = segmenter(rgb)
                self.assertEqual(mask.shape, rgb.shape[:2])
                self.assertTrue(np.isfinite(mask).all())
                self.assertGreaterEqual(float(mask.min()), 0)
                self.assertLessEqual(float(mask.max()), 1)
        native.request_alloc.assert_called_once()
        self.assertEqual((native.entered, native.exited), (2, 2))

    def test_native_failures_and_uniform_masks_are_uncertain(self):
        changes = [
            lambda native: setattr(native.handler.performRequests_error_, "return_value", (False, "error")),
            lambda native: setattr(native.request.results, "return_value", []),
            lambda native: setattr(native.quartz.CVPixelBufferLockBaseAddress, "return_value", -6662),
            lambda native: setattr(native.quartz, "CVPixelBufferGetBaseAddress", lambda buffer: None),
            lambda native: setattr(native, "buffer", bytearray(8)),
            lambda native: setattr(native, "buffer", bytearray([255] * 8)),
            lambda native: setattr(native.handler.performRequests_error_, "side_effect", RuntimeError("native failure")),
        ]
        for change in changes:
            native = NativeHarness()
            change(native)
            with self.subTest(change=change), \
                    patch("shad_helper.person_segmentation._native_modules", return_value=native.modules()):
                self.assertIsNone(PersonSegmenter()(np.zeros((2, 2, 3), dtype=np.uint8)))
            self.assertEqual((native.entered, native.exited), (1, 1))

    def test_buffer_is_unlocked_when_copy_fails(self):
        native = NativeHarness()
        native.address.as_buffer = lambda count: memoryview(b"too short")[:2]
        with patch("shad_helper.person_segmentation._native_modules", return_value=native.modules()):
            self.assertIsNone(PersonSegmenter()(np.zeros((2, 2, 3), dtype=np.uint8)))
        native.quartz.CVPixelBufferUnlockBaseAddress.assert_called_once()

    def test_invalid_frame_is_a_caller_error(self):
        native = NativeHarness()
        with patch("shad_helper.person_segmentation._native_modules", return_value=native.modules()):
            segmenter = PersonSegmenter()
            with self.assertRaisesRegex(ValueError, "RGB uint8"):
                segmenter(np.zeros((2, 2), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "RGB uint8"):
                segmenter(np.zeros((2, 2, 3), dtype=np.float32))
        native.handler.performRequests_error_.assert_not_called()


if __name__ == "__main__":
    unittest.main()
