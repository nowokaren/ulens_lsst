"""Offline unit tests for image discovery, DataLink, and Butler access."""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from astropy.io.votable import from_table, writeto
from astropy.table import Table

from ulens_lsst.image_access import (
    ButlerImageAccess,
    ImageProduct,
    TapImageAccess,
    download_tap_image,
    find_butler_images,
    find_tap_images,
    load_butler_exposure,
    load_fits_exposure,
    make_image_access,
    make_tap_service,
    resolve_datalink_product,
    image_product_from_record,
    _footprint_overlaps_region,
    validate_image_access_config,
)


class FakeResponse:
    def __init__(self, content=b"FITS", *, url="https://example.org/data/image.fits", headers=None):
        self.content = content
        self.url = url
        self.headers = headers or {"Content-Type": "application/fits"}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.content


def make_datalink_votable(rows):
    table = Table(rows=rows, names=("semantics", "access_url", "content_type", "error_message"))
    stream = BytesIO()
    writeto(from_table(table), stream)
    return stream.getvalue()


class TestImageProduct(unittest.TestCase):
    def test_descriptor_round_trips_through_json(self):
        descriptor = ImageProduct(
            backend="butler",
            stable_id="uuid-123",
            collection="LSSTComCam/DP1",
            product_type="visit_image",
            visit=123,
            detector=4,
            band="r",
            dataset_type="visit_image",
            data_id={"visit": 123, "detector": 4},
        )

        restored = ImageProduct.from_dict(descriptor.to_json())

        self.assertEqual(restored, descriptor)
        self.assertEqual(restored.to_dict()["data_id"], {"visit": 123, "detector": 4})

    def test_old_calexp_catalog_row_becomes_butler_descriptor(self):
        descriptor = image_product_from_record(
            {"visit": 123.0, "detector": 4.0, "band": "r", "expMidptMJD": 61000.5},
            data_preview="dp1",
        )
        self.assertEqual(descriptor.backend, "butler")
        self.assertEqual(descriptor.dataset_type, "visit_image")
        self.assertEqual(descriptor.data_id, {"visit": 123, "detector": 4})
        self.assertEqual(descriptor.expMidptMJD, 61000.5)


class TestTapOperations(unittest.TestCase):
    def test_make_tap_service_reports_missing_token_without_leaking_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            token_path = Path(tmp) / "missing.token"
            with self.assertRaisesRegex(FileNotFoundError, "TAP token file is unavailable"):
                make_tap_service(token_path, auth="token")

    def test_injected_session_is_used_for_service_and_datalink_security(self):
        session = Mock()
        service = Mock()
        with patch("pyvo.dal.TAPService", return_value=service) as tap_service:
            built_service, built_session = make_tap_service(session=session)

        self.assertIs(built_service, service)
        self.assertIs(built_session, session)
        tap_service.assert_called_once()
        session.add_security_method_for_url.assert_called_once()

    def test_find_tap_images_filters_obs_core_and_returns_plain_rows(self):
        table = Table(
            rows=[(
                "LSST.DP1", "visit-123", "ivo://rubin/123", "image",
                "lsst.visit_image", 2, "https://example.org/datalink/123",
                "application/x-votable+xml", 12.0, -34.0, "CIRCLE ICRS 12 -34 0.1",
                61000.0, 61000.01, "r",
            )],
            names=(
                "obs_collection", "obs_id", "obs_publisher_did", "dataproduct_type",
                "dataproduct_subtype", "calib_level", "access_url", "access_format",
                "s_ra", "s_dec", "s_region", "t_min", "t_max", "lsst_band",
            ),
        )
        service = Mock()
        service.search.return_value.to_table.return_value = table

        rows = find_tap_images(
            service,
            obs_collection="LSST.DP1",
            product_subtype="lsst.visit_image",
            obs_id="visit-123",
            region=(12, -34, 0.1),
            bands=["r"],
            limit=5,
        )

        query = service.search.call_args.args[0]
        self.assertIn("FROM ivoa.ObsCore", query)
        self.assertIn("dataproduct_subtype = 'lsst.visit_image'", query)
        self.assertIn("obs_id = 'visit-123'", query)
        self.assertIn("CONTAINS(POINT('ICRS', s_ra, s_dec)", query)
        self.assertIn("lsst_band IN ('r')", query)
        self.assertIn("TOP 5", query)
        self.assertIsInstance(rows[0], dict)
        self.assertEqual(rows[0]["obs_id"], "visit-123")

    def test_find_tap_images_validates_limit_and_region(self):
        service = Mock()
        for invalid in (0, -1, True, 1.5, "2"):
            with self.subTest(limit=invalid), self.assertRaisesRegex(ValueError, "limit"):
                find_tap_images(service, limit=invalid)
        with self.assertRaisesRegex(ValueError, "region"):
            find_tap_images(service, region=(400, 0, 1))
        service.search.assert_not_called()

    def test_find_tap_images_can_filter_explicit_visit_and_detector(self):
        service = Mock()
        service.search.return_value.to_table.return_value = Table(
            names=("obs_collection",), dtype=("U12",)
        )
        find_tap_images(service, visit=123, detector=4)
        query = service.search.call_args.args[0]
        self.assertIn("lsst_visit = 123", query)
        self.assertIn("lsst_detector = 4", query)

    def test_obs_core_footprint_filter_keeps_overlap_and_rejects_disjoint_image(self):
        footprint = "POLYGON ICRS 9.9 -19.9 10.1 -19.9 10.1 -20.1 9.9 -20.1"
        self.assertTrue(_footprint_overlaps_region(footprint, (10.2, -20, 0.11)))
        self.assertFalse(_footprint_overlaps_region(footprint, (10.4, -20, 0.05)))

    def test_resolve_datalink_selects_this_and_joins_relative_url(self):
        response = FakeResponse(
            make_datalink_votable([
                ("#preview", "preview.fits", "application/fits", ""),
                ("#this", "../signed/image.fits?token=temporary", "application/fits", ""),
            ]),
            url="https://example.org/datalink/table.xml",
            headers={"Content-Type": "application/x-votable+xml"},
        )
        session = Mock()
        session.get.return_value = response

        product = resolve_datalink_product(session, response.url)

        self.assertEqual(
            product["access_url"],
            "https://example.org/signed/image.fits?token=temporary",
        )
        self.assertEqual(product["semantics"], "#this")

    def test_resolve_datalink_rejects_missing_ambiguous_and_error_products(self):
        cases = [
            ([("#preview", "preview.fits", "application/fits", "")], LookupError),
            (
                [("#this", "one.fits", "application/fits", ""),
                 ("#this", "two.fits", "application/fits", "")],
                RuntimeError,
            ),
            ([("", "", "", "product unavailable")], RuntimeError),
        ]
        for rows, error in cases:
            with self.subTest(error=error):
                session = Mock()
                session.get.return_value = FakeResponse(
                    make_datalink_votable(rows),
                    headers={"Content-Type": "application/x-votable+xml"},
                )
                with self.assertRaises(error):
                    resolve_datalink_product(session, "https://example.org/datalink")

    def test_resolve_datalink_rejects_non_votable_and_http_errors(self):
        session = Mock()
        session.get.return_value = FakeResponse(
            b"<html>sign in</html>", headers={"Content-Type": "text/html"}
        )
        with self.assertRaisesRegex(ValueError, "not a VOTable"):
            resolve_datalink_product(session, "https://example.org/datalink")

        response = FakeResponse()
        response.raise_for_status = Mock(side_effect=RuntimeError("HTTP 403"))
        session.get.return_value = response
        with self.assertRaisesRegex(RuntimeError, "403"):
            resolve_datalink_product(session, "https://example.org/datalink")

    def test_download_is_atomic_reuses_cache_and_serializes_same_product(self):
        class SlowSession:
            def __init__(self):
                self.calls = 0
                self.guard = threading.Lock()

            def get(self, url, stream):
                with self.guard:
                    self.calls += 1
                time.sleep(0.05)
                return FakeResponse(b"complete FITS", headers={
                    "Content-Type": "application/fits", "Content-Length": "13"
                })

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "image.fits"
            session = SlowSession()
            with ThreadPoolExecutor(max_workers=2) as pool:
                paths = list(pool.map(
                    lambda _: download_tap_image(
                        session, {"access_url": "https://example.org/image"}, output
                    ),
                    range(2),
                ))

            self.assertEqual(paths, [output, output])
            self.assertEqual(output.read_bytes(), b"complete FITS")
            self.assertEqual(session.calls, 1)
            self.assertEqual(list(Path(tmp).glob("*.part")), [])
            self.assertTrue((Path(tmp) / ".locks").is_dir())

    def test_download_removes_partial_file_when_content_length_is_wrong(self):
        session = Mock()
        session.get.return_value = FakeResponse(
            b"short", headers={"Content-Type": "application/fits", "Content-Length": "100"}
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "image.fits"
            with self.assertRaisesRegex(IOError, "Content-Length"):
                download_tap_image(session, "https://example.org/image", output)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(tmp).glob("*.part")), [])

    def test_download_error_does_not_expose_signed_url(self):
        response = FakeResponse()
        response.raise_for_status = Mock(
            side_effect=RuntimeError(
                "403 https://signed.example/image.fits?token=temporary-secret"
            )
        )
        session = Mock(get=Mock(return_value=response))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "HTTP 403") as raised:
                download_tap_image(
                    session,
                    "https://signed.example/image.fits?token=temporary-secret",
                    Path(tmp) / "image.fits",
                )
        self.assertNotIn("temporary-secret", str(raised.exception))

    def test_load_fits_exposure_checks_file_and_uses_exposuref(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.fits"
            with self.assertRaises(FileNotFoundError):
                load_fits_exposure(path)
            path.write_bytes(b"synthetic")
            exposure = Mock()
            with patch("lsst.afw.image.ExposureF", return_value=exposure) as loader:
                self.assertIs(load_fits_exposure(path), exposure)
            loader.assert_called_once_with(str(path))

    def test_tap_backend_creates_descriptor_and_loads_from_stable_datalink(self):
        table = Table(
            rows=[("LSST.DP1", "visit-1", "publisher:1", "image", "lsst.visit_image",
                   "https://example.org/datalink/1", 10.0, -20.0, 61000.0, 61000.02,
                   "r", 123, 4,
                   "POLYGON ICRS 9.9 -19.9 10.1 -19.9 10.1 -20.1 9.9 -20.1")],
            names=("obs_collection", "obs_id", "obs_publisher_did", "dataproduct_type",
                   "dataproduct_subtype", "access_url", "s_ra", "s_dec", "t_min", "t_max",
                   "lsst_band", "lsst_visit", "lsst_detector", "s_region"),
        )
        ccd_visit = Table(
            rows=[(123, 4, 61001.0, 2.0, 0.8, 24.0, "r")],
            names=("visitId", "detector", "expMidptMJD", "psfSigma", "seeing", "magLim", "band"),
        )
        service = Mock()
        service.search.side_effect = [Mock(to_table=Mock(return_value=table)), Mock(to_table=Mock(return_value=ccd_visit))]
        session = Mock()
        access = TapImageAccess(service=service, session=session, cache_dir="/tmp/image-cache")

        products = access.find_exposures(region=(10, -20, 0.1), bands=["r"])

        self.assertIsInstance(products[0], ImageProduct)
        restored = ImageProduct.from_dict(products[0].to_json())
        self.assertEqual(restored.stable_id, "publisher:1")
        self.assertEqual(restored.datalink_url, "https://example.org/datalink/1")
        self.assertEqual(restored.obs_id, "visit-1")
        self.assertEqual(restored.expMidptMJD, 61001.0)
        self.assertAlmostEqual(restored.obscore_midpoint_mjd, 61000.01)
        self.assertEqual(restored.psfSigma, 2.0)
        self.assertEqual(restored.seeing, 0.8)

    def test_cached_exposure_load_does_not_need_a_live_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cached.fits"
            path.write_bytes(b"FITS")
            product = ImageProduct(
                backend="tap", stable_id="product-1", collection="LSST.DP1",
                product_type="image", product_subtype="lsst.visit_image",
                datalink_url="https://example.org/datalink/1", local_path=str(path),
            )
            access = TapImageAccess(service=Mock(), cache_dir=tmp)
            exposure = object()
            with patch("ulens_lsst.image_access.load_fits_exposure", return_value=exposure) as loader:
                self.assertIs(access.load_exposure(product), exposure)

        loader.assert_called_once_with(path)

    def test_expired_signed_url_is_refreshed_without_persisting_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            product = ImageProduct(
                backend="tap", stable_id="product-1", collection="LSST.DP1",
                product_type="image", product_subtype="lsst.visit_image",
                datalink_url="https://example.org/datalink/1",
            )
            access = TapImageAccess(service=Mock(), session=Mock(), cache_dir=tmp)
            expired = RuntimeError("expired signed URL")
            expired.response = Mock(status_code=403)
            exposure = object()
            with (
                patch("ulens_lsst.image_access.resolve_datalink_product", side_effect=[
                    {"access_url": "https://signed.example/expired"},
                    {"access_url": "https://signed.example/refreshed"},
                ]) as resolve,
                patch("ulens_lsst.image_access.download_tap_image", side_effect=[expired, Path(tmp) / "image.fits"]) as download,
                patch("ulens_lsst.image_access.load_fits_exposure", return_value=exposure),
            ):
                self.assertIs(access.load_exposure(product), exposure)

        self.assertEqual(resolve.call_count, 2)
        self.assertEqual(download.call_count, 2)
        self.assertEqual(product.datalink_url, "https://example.org/datalink/1")
        self.assertNotIn("signed.example", product.to_json())


class TestButlerAccess(unittest.TestCase):
    def test_butler_factory_is_lazy_and_defaults_to_legacy_backend(self):
        access = make_image_access()
        self.assertIsInstance(access, ButlerImageAccess)
        self.assertEqual(access.collection, "2.2i/runs/DP0.2")
        self.assertEqual(access.dataset_type, "calexp")
        self.assertIsNone(access._butler)

    def test_unknown_backend_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Unknown image_access.backend"):
            make_image_access({"backend": "unknown"})

    def test_tap_backend_is_scoped_to_validated_dp1(self):
        with self.assertRaisesRegex(ValueError, "only for DP1"):
            make_image_access({"backend": "tap"}, data_preview="dp0")
        access = make_image_access(
            {"backend": "tap", "collection": "LSST.DP1", "product_subtype": "lsst.visit_image"},
            data_preview="dp1",
            service=Mock(),
            session=Mock(),
        )
        self.assertIsInstance(access, TapImageAccess)
        self.assertEqual(access.collection, "LSST.DP1")
        self.assertEqual(access.product_subtype, "lsst.visit_image")

    def test_tap_configuration_requires_collection_subtype_and_known_auth(self):
        for config, message in (
            ({"backend": "tap"}, "collection is required"),
            ({"backend": "tap", "collection": "LSST.DP1"}, "product_subtype is required"),
            ({"backend": "tap", "collection": "LSST.DP1", "product_subtype": "lsst.visit_image", "auth": "magic"}, "auth must be"),
        ):
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, message):
                validate_image_access_config(config, data_preview="dp1")

    def test_rsp_auth_uses_integrated_service_and_external_auth_uses_token(self):
        service = Mock()
        service._session = Mock()
        with patch("lsst.rsp.get_tap_service", return_value=service) as get_service:
            built_service, session = make_tap_service(auth="rsp")
        get_service.assert_called_once_with("tap")
        self.assertIs(built_service, service)
        self.assertIs(session, service._session)

    def test_find_butler_images_and_load_mapping(self):
        exposure = __import__("lsst.afw.image", fromlist=["ExposureF"]).ExposureF(3, 4)
        butler = Mock()
        butler.query_datasets.return_value = ["dataset-ref"]
        self.assertEqual(
            find_butler_images(butler, dataset_type="visit_image", where="band = 'r'", limit=2),
            ["dataset-ref"],
        )
        butler.query_datasets.assert_called_once_with(
            "visit_image", limit=2, where="band = 'r'"
        )

        butler.get.return_value = exposure
        result = load_butler_exposure(
            butler,
            {"dataset_type": "visit_image", "data_id": {"visit": 10, "detector": 2}},
        )
        self.assertIs(result, exposure)
        butler.get.assert_called_once_with("visit_image", visit=10, detector=2)

    def test_butler_data_coordinate_mapping_becomes_serializable_descriptor(self):
        ref = SimpleNamespace(
            id="uuid-123",
            dataId=SimpleNamespace(mapping={
                "instrument": "LSSTComCam", "visit": 10, "detector": 2,
                "band": "r", "physical_filter": "r_03",
            }),
            run="LSSTComCam/DP1",
        )
        butler = Mock()
        butler.query_datasets.return_value = [ref]
        access = ButlerImageAccess(data_preview="dp1", butler=butler)

        product = access.find_exposures(region=(10, -20, 0.1), bands=["r"])[0]

        self.assertEqual(product.stable_id, "uuid-123")
        self.assertEqual(product.visit, 10)
        self.assertEqual(product.data_id["detector"], 2)
        self.assertEqual(product.data_id["physical_filter"], "r_03")
        self.assertEqual(ImageProduct.from_dict(product.to_json()), product)

    def test_butler_loader_rejects_non_exposure(self):
        butler = Mock()
        butler.get.return_value = object()
        with self.assertRaisesRegex(TypeError, "not ExposureF"):
            load_butler_exposure(
                butler,
                {"dataset_type": "calexp", "data_id": {"visit": 10, "detector": 2}},
            )


if __name__ == "__main__":
    unittest.main()
