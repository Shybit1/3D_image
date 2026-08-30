"""
Backend orchestration for geometric measurement (AeroTwin AI Module 18).
Thin wrapper over `depthwizard.digital_twin.measurement` -- operates on
objects already extracted by `digital_twin.build_digital_twin`.
"""
from __future__ import annotations

from depthwizard.digital_twin.measurement import (
    MeasurementResult,
    measure_area,
    measure_bounding_box,
    measure_distance,
    measure_height,
    measure_volume_estimate,
)
from depthwizard.digital_twin.objects import DigitalTwinObject


def measure_object(obj: DigitalTwinObject) -> dict:
    """All standard measurements for one object, bundled -- matches the
    `POST /api/reconstruction/{job_id}/measure` endpoint shape from
    Module 40."""
    return {
        "height": measure_height(obj).__dict__,
        "area": measure_area(obj).__dict__,
        "volume_estimate": measure_volume_estimate(obj).__dict__,
        "bounding_box": measure_bounding_box(obj).__dict__,
    }


def measure_object_pair_distance(a: DigitalTwinObject, b: DigitalTwinObject) -> dict:
    return measure_distance(a, b).__dict__
