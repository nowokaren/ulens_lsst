"""Offline tests for the minimal image-processor boundary."""

import unittest

from ulens_lsst.image_processors import (
    IMAGE_PROCESSORS,
    ImageProcessResult,
    ImageProcessor,
    InjectionForcedPhotometryProcessor,
    get_image_processor,
)


class TestImageProcessors(unittest.TestCase):
    def test_registry_contains_forced_photometry_processor(self):
        self.assertIs(
            get_image_processor("injection_forced_photometry").__class__,
            InjectionForcedPhotometryProcessor,
        )
        self.assertIn("injection_forced_photometry", IMAGE_PROCESSORS)

    def test_unknown_processor_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Unknown image processor"):
            get_image_processor("not_registered")

    def test_result_allows_processor_specific_products(self):
        result = ImageProcessResult(products={"photometry": [1], "diagnostics": {}})
        self.assertEqual(result.status, "success")
        self.assertIn("diagnostics", result.products)

    def test_base_interface_is_explicit(self):
        with self.assertRaises(NotImplementedError):
            ImageProcessor().process(None, None, None)


if __name__ == "__main__":
    unittest.main()
