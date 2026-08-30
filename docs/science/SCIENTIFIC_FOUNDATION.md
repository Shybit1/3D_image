# Scientific Foundation

Brief reference definitions for terms used throughout this project.

**DSM (Digital Surface Model)** — an elevation raster representing the
*top* of everything on the ground: buildings, tree canopy, and terrain
together. Contrast with a DTM (Digital Terrain Model), which represents
bare-earth elevation only. This project produces DSMs, consistent with
SIH26175's framing (height of visible surface features, e.g. buildings).

**DEM (Digital Elevation Model)** — a general term for any gridded
elevation dataset; often used loosely to mean DTM. SRTM data (below) is
technically closer to a DSM in vegetated/urban areas due to radar
penetration limits, which matters when using it as a calibration
reference.

**Monocular depth estimation** — inferring a per-pixel depth/distance
value from a single 2D image, without stereo or motion parallax. Modern
approaches (MiDaS, Depth Anything, DPT) use deep networks trained on large
mixed datasets to predict *relative* depth (correct ordering and rough
scale) rather than absolute metric depth, since a single image is
fundamentally scale-ambiguous without additional information (camera
intrinsics, known object sizes, or a reference surface).

**Relative depth vs. absolute elevation** — relative depth answers "which
points are closer/farther, roughly how much" with no fixed unit. Absolute
elevation requires tying that relative signal to a real coordinate system
and unit (meters above a vertical datum) via calibration.

**Georeferencing / CRS** — a georeferenced raster carries a Coordinate
Reference System and an affine pixel-to-world transform, so each pixel
maps to a real-world (x, y) location. Without this, an image is just
pixels with no inherent location.

**Vertical datum** — the zero-reference surface for elevation (e.g. mean
sea level, or an ellipsoidal model like WGS84). Different DEM sources use
different vertical datums; mixing them without conversion introduces
systematic offset error. This project does not currently perform vertical
datum reconciliation (PLANNED) — calibration fits are only as correct as
the reference data's stated datum.

**SRTM (Shuttle Radar Topography Mission)** — a near-global elevation
dataset (~30m or ~90m resolution depending on region) from a 2000 radar
mission. Commonly used as a free, if coarse, DEM reference. Radar
penetrates vegetation canopy partially, so SRTM values in forested areas
sit somewhere between bare terrain and canopy top — a known source of
calibration noise when using SRTM as a DSM reference in forests.

**GCP (Ground Control Point)** — a location with known, independently
verified real-world coordinates and elevation, used to anchor or validate
a model's spatial/vertical accuracy. A handful of well-distributed GCPs
can calibrate a relative depth map to absolute elevation even without a
full DEM.

**Uncertainty (in this project's sense)** — a measure of how much
independent forward passes of the same model (at different scales, with
flip augmentation) disagree with each other for a given pixel. High
disagreement is empirically associated with lower prediction reliability
(ambiguous texture, domain-gap regions, edges) but is not a calibrated
statistical confidence interval.

**Photogrammetry** — the general science/technique of extracting
measurements (including 3D structure) from photographs. Classical
photogrammetry uses multiple overlapping images (stereo or multi-view) to
triangulate depth geometrically; this project instead uses learned
monocular depth as a substitute when only a single image is available, per
the SIH26175 problem statement's premise.
