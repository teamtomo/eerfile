import os
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class EERHeader(BaseModel):
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
        return self.pixel_spacing_height_meters * 1e10

    @property
    def pixel_spacing_width_angstroms(self) -> float:
        return self.pixel_spacing_width_meters * 1e10

    @property
    def pixel_area_square_angstroms(self) -> float:
        h, w = (
            self.pixel_spacing_height_angstroms,
            self.pixel_spacing_width_angstroms
        )
        return h * w

    @property
    def total_dose_electrons_per_square_angstrom(self) -> float:
        return self.total_dose_electrons_per_pixel / self.pixel_area_square_angstroms

    @property
    def dose_rate_electrons_per_square_angstrom(self) -> float:
        return self.dose_rate_electrons_per_pixel_per_second / self.pixel_area_square_angstroms

    @property
    def dose_per_frame_electrons_per_square_angstrom(self) -> float:
        return self.total_dose_electrons_per_square_angstrom / self.n_frames

    @classmethod
    def from_file(cls, file: os.PathLike) -> "EERHeader":
        from tifffile import TiffFile
        from lxml import etree

        # define a closure to grab xml items by name
        def _find_xml_item(name: str):
            return root.find(f"./item[@name='{name}']")

        with TiffFile(file) as tiff:
            root = etree.fromstring(tiff.eer_metadata)

        return cls(
            acquisition_id=_find_xml_item("acquisitionID").text,
            camera_name=_find_xml_item("cameraName").text,
            commercial_name=_find_xml_item("commercialName").text,
            exposure_time_seconds=_find_xml_item("exposureTime").text,
            dose_rate_electrons_per_pixel_per_second=_find_xml_item("meanDoseRate").text,
            total_dose_electrons_per_pixel=_find_xml_item("totalDose").text,
            n_frames=_find_xml_item("numberOfFrames").text,
            image_height_pixels=_find_xml_item("sensorImageHeight").text,
            image_width_pixels=_find_xml_item("sensorImageWidth").text,
            pixel_spacing_height_meters=_find_xml_item("sensorPixelSize.height").text,
            pixel_spacing_width_meters=_find_xml_item("sensorPixelSize.width").text,
            serial_number=_find_xml_item("serialNumber").text,
        )
