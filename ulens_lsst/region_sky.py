import os
import logging
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from regions import CircleSkyRegion, PolygonSkyRegion
from lsst.sphgeom import HtmPixelization, UnitVector3d, LonLat
from spherical_geometry.polygon import SphericalPolygon
from lsst.daf.butler import Butler
import pandas as pd
from tqdm.auto import tqdm
from shapely.geometry import Polygon, Point
from lsst.geom import Angle, SpherePoint, degrees

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def circ_sample(ra0, dec0, radius_deg, n_points, seed=None):
    """
    Generate random points within a circular sky region.

    Args:
        ra0 (float): Central RA in degrees.
        dec0 (float): Central Dec in degrees.
        radius_deg (float): Radius of the circle in degrees.
        n_points (int): Number of points to generate.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        Tuple[List[float], List[float]]: Lists of RA and Dec in degrees.
    """
    np.random.seed(seed)
    center = SkyCoord(ra0, dec0, unit='deg')
    thetas = np.random.uniform(0, 2 * np.pi, n_points) * u.rad
    radii = np.sqrt(np.random.uniform(0, 1, n_points)) * radius_deg * u.deg
    offsets = center.directional_offset_by(thetas, radii)
    ra = [coord.ra.deg for coord in offsets]
    dec = [coord.dec.deg for coord in offsets]
    return ra, dec


def spherical_circle_edge(ra0, dec0, radius, n_points=100):
    """
    Generate points along the edge of a spherical circle.

    Args:
        ra0 (float): Central RA in degrees.
        dec0 (float): Central Dec in degrees.
        radius (Quantity): Radius with units (e.g., u.arcsec or u.deg).
        n_points (int, optional): Number of points on the edge.

    Returns:
        tuple: Arrays of RA and Dec values in degrees.
    """
    center = SkyCoord(ra0 * u.deg, dec0 * u.deg, frame='icrs')
    theta = np.linspace(0, 2 * np.pi, n_points)
    radius_deg = radius.to(u.deg).value
    border = center.directional_offset_by(theta * u.rad, radius*deg)
    return border.ra.deg, border.dec.deg

def spherical_polygon_area(vertices):
    """
    Calculate the spherical area of a polygon defined by vertices.

    Args:
        vertices (list): List of (ra, dec) tuples in degrees.

    Returns:
        float: Area in square degrees.
    """
    ra, dec = zip(*vertices)  # Extract ra and dec as plain numbers (assumed in degrees)
    poly = SphericalPolygon.from_radec(ra, dec)  # No unit parameter
    return poly.area() * (180 / np.pi) ** 2  # Call area() method and convert steradians to deg²

def spherical_circle_area(radius):
    """
    Calculate the spherical area of a circular region (spherical cap approximation).

    Args:
        radius (Quantity): Radius with units (e.g., u.arcsec or u.deg).

    Returns:
        float: Area in square degrees.
    """
    radius_deg = radius.to(u.deg).value
    theta = np.radians(radius_deg)
    area_steradians = 2 * np.pi * (1 - np.cos(theta))
    return area_steradians * (180 / np.pi) ** 2  # Convert to deg²

class SkyRegion:
    """
    Manage sky region operations including area calculation, edge generation, containment checks,
    and random point generation.

    Attributes:
        center (SkyCoord): Central coordinates of the region.
        region_type (str): Type of region ('circle', 'polygon', or 'htm').
        region (object): Region object (CircleSkyRegion, PolygonSkyRegion, or SphericalPolygon).
        radius (Quantity): Radius with units (e.g., u.arcsec or u.deg) for circle.
        vertices (list): List of (ra, dec) tuples with units for polygon or HTM triangle.
        htm_level (int): HTM level if region_type is 'htm'.
        area (Quantity): Cached area in square degrees.
    """

    def __init__(self, ra, dec, region_type='circle', radius=None, vertices=None, htm_level=None):
        """
        Initialize the SkyRegion.

        Args:
            ra (float): Right Ascension in degrees.
            dec (float): Declination in degrees.
            region_type (str, optional): Type of region ('circle', 'polygon', or 'htm'). Defaults to 'circle'.
            radius (Quantity, optional): Radius with units (e.g., u.arcsec or u.deg) for circle.
            vertices (list, optional): List of (ra, dec) tuples with units for polygon or HTM triangle.
            htm_level (int, optional): HTM level if region_type is 'htm'.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        self.center = SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')
        self.region_type = region_type.lower()
        self.radius = radius  # Store as Quantity with original unit
        self.vertices = [(ra * u.deg, dec * u.deg) for ra, dec in vertices] if vertices else None  # Store with units
        self.htm_level = htm_level
        self.area = None  # Cached area as Quantity

        if self.region_type == 'circle':
            if radius is None:
                raise ValueError("Radius must be provided for circle region.")
            self.region = CircleSkyRegion(self.center, self.radius)
        elif self.region_type == 'polygon':
            if vertices is None or len(vertices) < 3:
                raise ValueError("Vertices must be provided as a list of at least 3 (ra, dec) tuples for polygon.")
            # Convert list of (ra, dec) tuples to a 1D SkyCoord object with units
            ra = [v[0] for v in self.vertices]
            dec = [v[1] for v in self.vertices]
            self.region = PolygonSkyRegion(SkyCoord(ra, dec, frame='icrs'))
        elif self.region_type == 'htm':
            if htm_level is None:
                raise ValueError("HTM level must be provided for HTM region.")
            pixelization = HtmPixelization(htm_level)
            htm_id = pixelization.index(UnitVector3d(LonLat.fromDegrees(ra * u.deg, dec * u.deg)))
            triangle = pixelization.triangle(htm_id)
            vertices = [(LonLat.longitudeOf(v).asDegrees() * u.deg, LonLat.latitudeOf(v).asDegrees() * u.deg) for v in triangle.getVertices()]
            self.vertices = vertices
            self.region = SphericalPolygon.from_radec(*zip(*[(v[0].value, v[1].value) for v in vertices]))
        else:
            raise ValueError("Region type must be 'circle', 'polygon', or 'htm'.")

    def calculate_area(self, spherical=True):
        """
        Calculate the area of the region.

        Args:
            spherical (bool, optional): If True, use spherical geometry; otherwise, use flat approximation.
                                       Defaults to True.

        Returns:
            Quantity: Area in square degrees.
        """
        if self.area is not None and not spherical:
            return self.area

        if spherical:
            if self.region_type == 'circle':
                area_value = spherical_circle_area(self.radius)
                area = area_value * u.deg**2
            elif self.region_type == 'polygon':
                area_value = spherical_polygon_area([(v[0].value, v[1].value) for v in self.vertices])
                area = area_value * u.deg**2
            elif self.region_type == 'htm':
                area_value = spherical_polygon_area([(v[0].value, v[1].value) for v in self.vertices])
                area = area_value * u.deg**2
            self.area = area
        else:
            if self.region_type == 'circle':
                area_value = np.pi * (self.radius.to(u.deg).value ** 2)
                area = area_value * u.deg**2
            elif self.region_type == 'polygon':
                from shapely.geometry import Polygon
                flat_vertices = [(ra.value, dec.value) for ra, dec in self.vertices]
                area_value = Polygon(flat_vertices).area
                area = area_value * u.deg**2  # Approximate, assuming deg^2 for small areas
            elif self.region_type == 'htm':
                area_value = np.pi / (2 * 4 ** (self.htm_level - 1)) * (180 / np.pi) ** 2
                area = area_value * u.deg**2
            self.area = area
        logger.info(f"Calculated {self.region_type} area: {self.area.value:.2f} {self.area.unit} (spherical={spherical})")
        return self.area

    def edge(self, n_points=100):
        """
        Generate points along the edge of the region.

        Args:
            n_points (int, optional): Number of points to generate. Defaults to 100.

        Returns:
            tuple: Arrays of RA and Dec values in degrees.
        """
        if self.region_type == 'circle':
            ra_edge, dec_edge = spherical_circle_edge(self.center.ra.deg, self.center.dec.deg, self.radius, n_points)
        elif self.region_type == 'polygon':
            # Interpolate points along polygon edges
            ra_vertices = np.array([v[0].value for v in self.vertices])
            dec_vertices = np.array([v[1].value for v in self.vertices])
            n_segments = len(self.vertices)
            points_per_segment = n_points // n_segments
            remainder = n_points % n_segments

            ra_edge = []
            dec_edge = []
            for i in range(n_segments):
                start_idx = i
                end_idx = (i + 1) % n_segments
                start_ra, start_dec = ra_vertices[start_idx], dec_vertices[start_idx]
                end_ra, end_dec = ra_vertices[end_idx], dec_vertices[end_idx]
                t = np.linspace(0, 1, points_per_segment + (1 if i < remainder else 0))
                interp_ra = start_ra + t * (end_ra - start_ra)
                interp_dec = start_dec + t * (end_dec - start_dec)
                ra_edge.extend(interp_ra[:-1])  # Exclude the last point to avoid duplication
                dec_edge.extend(interp_dec[:-1])

            # Ensure the last point connects to the first
            ra_edge.append(ra_vertices[0])
            dec_edge.append(dec_vertices[0])

            ra_edge = np.array(ra_edge)
            dec_edge = np.array(dec_edge)
        elif self.region_type == 'htm':
            raise NotImplementedError("Edge generation for HTM triangles not yet implemented.")
        logger.info(f"Generated {n_points} edge points for {self.region_type} region")
        return ra_edge, dec_edge

    def contains(self, ra, dec):
        """
        Check if a point is within the spherical region.

        Args:
            ra (float): Right Ascension in degrees.
            dec (float): Declination in degrees.

        Returns:
            bool: True if the point is inside, False otherwise.
        """

        if self.region_type == 'circle':
            center_point = SkyCoord(self.center.ra, self.center.dec, frame='icrs')
            test_point = SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')
            separation = center_point.separation(test_point).to(u.deg).value
            return separation <= self.radius.to(u.deg).value
        elif self.region_type == 'polygon':
            # Convert vertices to 2D Cartesian coordinates for shapely
            flat_vertices = [(v[0].value, v[1].value) for v in self.vertices]
            poly = Polygon(flat_vertices)
            test_point = (ra, dec)
            return poly.contains(Point(test_point))
        elif self.region_type == 'htm':
            raise NotImplementedError(f"Containment check for {self.region_type} not yet implemented.")
        logger.debug(f"Checked containment for point ({ra}, {dec}): {self.contains(ra, dec)}")
        return False  # Default case (should not reach here)

    def sample(self, n_points, seed=None):
        """
        Generate random points within the spherical region.

        Args:
            n_points (int): Number of points to generate.
            seed (int, optional): Random seed for reproducibility.

        Returns:
            pd.DataFrame: DataFrame with 'ra' and 'dec' columns in degrees.
        """
        points = []
        np.random.seed(seed)
        if self.region_type == 'circle':
            ra, dec = circ_sample(self.center.ra.deg, self.center.dec.deg, self.radius.value, n_points, seed)
        elif self.region_type in ['polygon', 'htm']:
            raise NotImplementedError(f"Random point generation for {self.region_type} not yet implemented.")
        # for _ in range(n_points):
        #     while True:
        #         if self.region_type == 'circle':
        #             ra, dec = circ_sample(self.center.ra.deg, self.center.dec.deg, self.radius.value, 1, seed)
        #         elif self.region_type in ['polygon', 'htm']:
        #             raise NotImplementedError(f"Random point generation for {self.region_type} not yet implemented.")
                # if self.contains(ra, dec):
                #     points.append((ra * u.deg, dec * u.deg))  # Store with units
                #     break
        return ra, dec

class LSSTDataLoader:
    """
    Load and manage LSST/DP0 catalogs and images, integrating with SkyRegion.

    Attributes:
        butler (Butler): LSST Butler instance.
        sky_region (SkyRegion): Associated sky region.
        data_calexps (pd.DataFrame): DataFrame of calexp metadata with units where applicable.
        data_catalogs (pd.DataFrame): DataFrame of source catalogs.
    """

    def __init__(self, sky_region, butler_config='dp02-direct', collections='2.2i/runs/DP0.2'):
        """
        Initialize the LSSTDataLoader.

        Args:
            sky_region (SkyRegion): Instance of SkyRegion.
            butler_config (str, optional): Butler configuration.
            collections (str, optional): Butler collections.
        """
        self.sky_region = sky_region
        self.butler = Butler(butler_config, collections=collections)
        self.data_calexps = pd.DataFrame(columns=['detector', 'visit', 'mjd', 'band', 'seeing'])
        self.data_catalogs = pd.DataFrame()

    def load_calexps(self, n_max='all', bands=None):
        """
        Load calexps overlapping the sky region.

        Args:
            n_max (int or str, optional): Maximum number of calexps to load. Defaults to 'all'.
            bands (list, optional): List of bands to filter. Defaults to all DP0 bands.

        Returns:
            list: List of dataset references.
        """
        bands = bands or ['u', 'g', 'r', 'i', 'z', 'y']
        bands_str = f"({', '.join(map(repr, bands))})"
        logger.info(f"Loading calexps for region at {self.sky_region.center.ra.deg}, {self.sky_region.center.dec.deg}")

        if self.region_type == 'htm':
            pixelization = HtmPixelization(self.sky_region.htm_level)
            htm_id = pixelization.index(UnitVector3d(LonLat.fromDegrees(self.sky_region.center.ra.deg, self.sky_region.center.dec.deg)))
            self.datasetRefs = list(self.butler.registry.queryDatasets("calexp", htm20=htm_id, where=f"band IN {bands_str}"))
        elif self.region_type in ['circle', 'polygon']:
            target_point = SkyCoord(self.sky_region.center.ra.deg, self.sky_region.center.dec.deg, unit=u.deg)
            radius = self.sky_region.radius if self.region_type == 'circle' else 1 * u.deg  # Approximate for polygon
            circle = CircleSkyRegion(target_point, radius)
            self.datasetRefs = self.butler.query_datasets("calexp", where=f"visit_detector_region.region OVERLAPS my_region AND band IN {bands_str}",
                                                         bind={"my_region": circle}, limit=100000000)

        if isinstance(n_max, int):
            self.datasetRefs = self.datasetRefs[:n_max]
            logger.info(f"Selected first {n_max} calexps")

        self._populate_calexp_metadata()
        return self.datasetRefs

    def _populate_calexp_metadata(self):
        """Populate data_calexps DataFrame with metadata from datasetRefs."""
        if self.datasetRefs:
            ccd_visit = self.butler.get('ccdVisitTable')
            for dataRef in tqdm(self.datasetRefs, desc="Populating calexp metadata"):
                data_id = dataRef.dataId
                mjd = ccd_visit[(ccd_visit['visitId'] == data_id['visit']) & (ccd_visit['detector'] == data_id['detector'])]['expMidptMJD'].values
                seeing = ccd_visit[(ccd_visit['visitId'] == data_id['visit']) & (ccd_visit['detector'] == data_id['detector'])]['seeing'].values
                new_row = pd.DataFrame({
                    'detector': [data_id['detector']],
                    'visit': [data_id['visit']],
                    'mjd': [mjd[0] if mjd.size else np.nan] * u.day if mjd.size else [np.nan * u.day],
                    'band': [data_id['band']],
                    'seeing': [seeing[0] * u.arcsec / 3600 if seeing.size else np.nan * u.deg]  # Convert arcsec to deg
                })
                self.data_calexps = pd.concat([self.data_calexps, new_row], ignore_index=True)
            logger.info(f"Populated metadata for {len(self.datasetRefs)} calexps")

    def load_sources(self, ra=None, dec=None, radius=None):
        """
        Load source catalogs within the sky region.

        Args:
            ra (float, optional): Central RA in degrees. Defaults to sky_region center.
            dec (float, optional): Central Dec in degrees. Defaults to sky_region center.
            radius (Quantity, optional): Search radius with units (e.g., u.arcsec or u.deg). Defaults to sky_region radius.

        Returns:
            pd.DataFrame: DataFrame of source catalog data.
        """
        ra = ra or self.sky_region.center.ra.deg
        dec = dec or self.sky_region.center.dec.deg
        radius = radius or self.sky_region.radius if self.region_type == 'circle' else 1 * u.deg
        radius_deg = radius.to(u.deg).value
        query = f"SELECT coord_ra, coord_dec, objectId FROM dp02_dc2_catalogs.Object " \
                f"WHERE CONTAINS(POINT('ICRS', coord_ra, coord_dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1 " \
                "AND detect_isPrimary = 1"
        from lsst.rsp import get_tap_service
        service = get_tap_service("tap")
        job = service.submit_job(query)
        job.run()
        job.wait(phases=['COMPLETED', 'ERROR'])
        results = job.fetch_result().to_table().to_pandas()
        self.data_catalogs = results
        logger.info(f"Loaded {len(results)} sources from catalog")
        return results