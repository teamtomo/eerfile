"""Data models for EER files."""

import os
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class EERHeader(BaseModel):
    """Data model for the EER header."""

    acquisition_id: str = ""
    camera_name: str = ""
    commercial_name: str = ""
    exposure_time_seconds: float | None = None
    dose_rate_electrons_per_pixel_per_second: float | None = None
    total_dose_electrons_per_pixel: float | None = None
    n_frames: int
    image_height_pixels: int
    image_width_pixels: int
    pixel_spacing_height_meters: float
    pixel_spacing_width_meters: float
    serial_number: str = ""
    timestamp: datetime | None = None

    @property
    def pixel_spacing_height_angstroms(self) -> float:
        """Pixel spacing height in angstroms."""
        return self.pixel_spacing_height_meters * 1e10

    @property
    def pixel_spacing_width_angstroms(self) -> float:
        """Pixel spacing width in angstroms."""
        return self.pixel_spacing_width_meters * 1e10

    @property
    def pixel_area_square_angstroms(self) -> float:
        """Pixel area in square angstroms."""
        h, w = (self.pixel_spacing_height_angstroms, self.pixel_spacing_width_angstroms)
        return h * w

    @property
    def total_dose_electrons_per_square_angstrom(self) -> float:
        """Total dose in electrons per square angstrom."""
        if self.total_dose_electrons_per_pixel is None:
            raise ValueError("total_dose_electrons_per_pixel is not available")
        return self.total_dose_electrons_per_pixel / self.pixel_area_square_angstroms

    @property
    def dose_rate_electrons_per_square_angstrom(self) -> float:
        """Dose rate in electrons per square angstrom."""
        if self.dose_rate_electrons_per_pixel_per_second is None:
            raise ValueError(
                "dose_rate_electrons_per_pixel_per_second is not available"
            )
        return (
            self.dose_rate_electrons_per_pixel_per_second
            / self.pixel_area_square_angstroms
        )

    @property
    def dose_per_frame_electrons_per_square_angstrom(self) -> float:
        """Dose per frame in electrons per square angstrom."""
        return self.total_dose_electrons_per_square_angstrom / self.n_frames

    @classmethod
    def from_file(cls, file: os.PathLike) -> "EERHeader":
        """
        Parse the EER header from a file.

        Parameters
        ----------
        file: os.PathLike
            Path to the EER file

        Returns
        -------
        header: EERHeader
            Parsed EER header
        """
        from tifffile import TiffFile

        with TiffFile(file) as tiff:
            metadata = tiff.eer_metadata

            # Helper functions to extract values from dict metadata
            def _get_dict_value(
                key_path: str, default: Any | None = None
            ) -> Any | None:
                """Extract value from dict using key name or dot notation."""
                # First try direct key access (flat dict)
                if key_path in metadata:
                    return metadata[key_path]

                # Try nested dict access (e.g., "sensorPixelSize.height")
                keys = key_path.split(".")
                value: Any | None = metadata
                for key in keys:
                    if isinstance(value, dict):
                        value = value.get(key)
                        if value is None:
                            break
                    else:
                        value = None
                        break
                if value is not None:
                    return value

                # Try list of items structure (like XML items)
                # For nested keys, check both the full path
                # and the base key (sensorPixelSize)
                base_key = keys[0] if keys else key_path
                if "items" in metadata or isinstance(metadata.get("item"), list):
                    items = metadata.get("items", metadata.get("item", []))
                    for item in items:
                        if isinstance(item, dict):
                            item_name = item.get("name")
                            # Match exact key or base key for nested paths
                            if item_name == key_path or item_name == base_key:
                                # For nested keys, try to access the nested value
                                if len(keys) > 1 and isinstance(
                                    item.get("value"), dict
                                ):
                                    nested_value = item.get("value")
                                    for sub_key in keys[1:]:
                                        if isinstance(nested_value, dict):
                                            nested_value = nested_value.get(sub_key)
                                        else:
                                            nested_value = None
                                            break
                                    if nested_value is not None:
                                        return nested_value
                                return item.get("value") or item.get("text")

                return default

            def _get_dict_text(key_path: str, default: str = "") -> str:
                value = _get_dict_value(key_path, default)
                return str(value) if value is not None else default

            def _get_dict_float(key_path: str) -> float | None:
                value = _get_dict_value(key_path)
                if value is not None:
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return None
                return None

            def _get_dict_int(key_path: str) -> int:
                value = _get_dict_value(key_path)
                if value is not None:
                    try:
                        return int(value)
                    except (ValueError, TypeError):
                        return 0
                return 0

            # Extract values from dict
            return cls(
                acquisition_id=_get_dict_text("acquisitionID"),
                camera_name=_get_dict_text("cameraName"),
                commercial_name=_get_dict_text("commercialName"),
                exposure_time_seconds=_get_dict_float("exposureTime"),
                dose_rate_electrons_per_pixel_per_second=_get_dict_float(
                    "meanDoseRate"
                ),
                total_dose_electrons_per_pixel=_get_dict_float("totalDose"),
                n_frames=_get_dict_int("numberOfFrames"),
                image_height_pixels=_get_dict_int("sensorImageHeight"),
                image_width_pixels=_get_dict_int("sensorImageWidth"),
                pixel_spacing_height_meters=_get_dict_float("sensorPixelSize.height")
                or 0.0,
                pixel_spacing_width_meters=_get_dict_float("sensorPixelSize.width")
                or 0.0,
                serial_number=_get_dict_text("serialNumber"),
            )
