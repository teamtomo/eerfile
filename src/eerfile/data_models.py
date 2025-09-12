"""Data models for EER files."""

import os
from datetime import datetime
from typing import Optional

from lxml import etree
from pydantic import BaseModel


class EERHeader(BaseModel):
    """Data model for the EER header."""

    acquisition_id: str = ""
    camera_name: str = ""
    commercial_name: str = ""
    exposure_time_seconds: Optional[float] = None
    dose_rate_electrons_per_pixel_per_second: Optional[float] = None
    total_dose_electrons_per_pixel: Optional[float] = None
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

        # define helper functions to safely extract XML data
        def _find_xml_item(name: str) -> Optional[etree._Element]:
            return root.find(f"./item[@name='{name}']")

        def _get_xml_text(name: str, default: str = "") -> str:
            element = _find_xml_item(name)
            return (
                element.text
                if element is not None and element.text is not None
                else default
            )

        def _get_xml_float(name: str) -> Optional[float]:
            element = _find_xml_item(name)
            if element is not None and element.text is not None:
                try:
                    return float(element.text)
                except (ValueError, TypeError):
                    return None
            return None

        def _get_xml_int(name: str) -> int:
            element = _find_xml_item(name)
            if element is not None and element.text is not None:
                try:
                    return int(element.text)
                except (ValueError, TypeError):
                    return 0
            return 0

        with TiffFile(file) as tiff:
            root = etree.fromstring(tiff.eer_metadata)

        return cls(
            acquisition_id=_get_xml_text("acquisitionID"),
            camera_name=_get_xml_text("cameraName"),
            commercial_name=_get_xml_text("commercialName"),
            exposure_time_seconds=_get_xml_float("exposureTime"),
            dose_rate_electrons_per_pixel_per_second=_get_xml_float("meanDoseRate"),
            total_dose_electrons_per_pixel=_get_xml_float("totalDose"),
            n_frames=_get_xml_int("numberOfFrames"),
            image_height_pixels=_get_xml_int("sensorImageHeight"),
            image_width_pixels=_get_xml_int("sensorImageWidth"),
            pixel_spacing_height_meters=_get_xml_float("sensorPixelSize.height") or 0.0,
            pixel_spacing_width_meters=_get_xml_float("sensorPixelSize.width") or 0.0,
            serial_number=_get_xml_text("serialNumber"),
        )
