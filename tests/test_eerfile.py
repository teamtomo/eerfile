from unittest.mock import Mock, patch

import numpy as np
import pytest

import eerfile
from eerfile.data_models import EERHeader
from eerfile.render_eer import _render_frame


class TestRenderFrame:
    """Test the _render_frame helper function."""

    def test_render_frame_basic(self):
        """Test basic frame rendering functionality."""
        # Create mock EER chunk data (3 frames, 4x4 pixels)
        eer_chunk = np.array(
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17]],
                [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            ],
            dtype=np.uint16,
        )

        # Test combining first 2 frames (index 0 in chunk starting at 0)
        result = _render_frame(
            i=0,
            chunk_start=0,
            eer_frames_per_output_frame=2,
            eer_chunk=eer_chunk,
            output_dtype=np.uint16,
        )

        expected = np.array(
            [[3, 5, 7, 9], [11, 13, 15, 17], [19, 21, 23, 25], [27, 29, 31, 33]],
            dtype=np.uint16,
        )

        np.testing.assert_array_equal(result, expected)

    def test_render_frame_chunk_offset(self):
        """Test frame rendering with chunk offset."""
        eer_chunk = np.array(
            [
                [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]],
            ],
            dtype=np.uint16,
        )

        # Test frame 10 in a chunk that starts at frame 10
        result = _render_frame(
            i=10,
            chunk_start=10,
            eer_frames_per_output_frame=2,
            eer_chunk=eer_chunk,
            output_dtype=np.uint16,
        )

        expected = np.array(
            [[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]], dtype=np.uint16
        )

        np.testing.assert_array_equal(result, expected)

    def test_render_frame_partial_chunk(self):
        """Test frame rendering when chunk doesn't have enough frames."""
        # Only 1 frame available but trying to combine 2
        eer_chunk = np.array(
            [[[5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5]]], dtype=np.uint16
        )

        result = _render_frame(
            i=0,
            chunk_start=0,
            eer_frames_per_output_frame=2,
            eer_chunk=eer_chunk,
            output_dtype=np.uint16,
        )

        # Should just sum the available frame(s)
        expected = np.array(
            [[5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5]], dtype=np.uint16
        )

        np.testing.assert_array_equal(result, expected)


class TestReadFrames:
    """Test the read_frames function."""

    @patch("eerfile.read_eer.TiffFile")
    def test_read_frames_basic(self, mock_tiff_file):
        """Test basic frame range reading."""
        # Mock TiffFile behavior
        mock_data = np.random.randint(0, 100, size=(10, 4, 4), dtype=np.uint16)
        mock_tiff = Mock()
        mock_tiff.asarray.return_value = mock_data[2:5]  # frames 2-4
        mock_tiff_file.return_value.__enter__.return_value = mock_tiff

        from eerfile.read_eer import read_frames

        result = read_frames("test.eer", 2, 5)

        mock_tiff.asarray.assert_called_once_with(key=slice(2, 5))
        np.testing.assert_array_equal(result, mock_data[2:5])

    @patch("eerfile.read_eer.TiffFile")
    def test_read_frames_to_end(self, mock_tiff_file):
        """Test reading from start frame to end."""
        mock_data = np.random.randint(0, 100, size=(5, 4, 4), dtype=np.uint16)
        mock_tiff = Mock()
        mock_tiff.asarray.return_value = mock_data[3:]  # frames 3 to end
        mock_tiff_file.return_value.__enter__.return_value = mock_tiff

        from eerfile.read_eer import read_frames

        result = read_frames("test.eer", 3, None)

        mock_tiff.asarray.assert_called_once_with(key=slice(3, None))
        np.testing.assert_array_equal(result, mock_data[3:])


class TestRenderMemoryEfficient:
    """Test the memory-efficient render functions."""

    def create_mock_eer_header(self, n_frames=100, height=512, width=512):
        """Create a mock EER header for testing."""
        header = Mock(spec=EERHeader)
        header.n_frames = n_frames
        header.image_height_pixels = height
        header.image_width_pixels = width
        header.dose_per_frame_electrons_per_square_angstrom = 0.1
        return header

    @patch("eerfile.render_eer.EERHeader.from_file")
    @patch("eerfile.render_eer.read_frames")
    def test_render_chunked_processing(self, mock_read_frames, mock_header_from_file):
        """Test that render processes frames in chunks."""
        # Setup mocks - create enough frames to force multiple chunks
        mock_header = self.create_mock_eer_header(n_frames=50, height=4, width=4)
        mock_header_from_file.return_value = mock_header

        # Mock read_frames to return predictable data
        def mock_read_frames_func(file, start, end):
            n_frames = end - start
            return np.ones((n_frames, 4, 4), dtype=np.uint16) * (start + 1)

        mock_read_frames.side_effect = mock_read_frames_func

        # Test with small chunk size to force multiple chunks
        # 50 frames / 10 frames per output = 5 output frames
        # With chunk_size=2, we'll have 3 chunks (2, 2, 1 frames each)
        result = eerfile.render("test.eer", dose_per_output_frame=1.0, chunk_size=2)

        # Should have called read_frames multiple times (chunked processing)
        assert mock_read_frames.call_count > 1
        assert result.shape == (
            5,
            4,
            4,
        )  # 50 frames / 10 frames per output = 5 output frames

    @patch("eerfile.render_eer.EERHeader.from_file")
    @patch("eerfile.render_eer.read_frames")
    def test_render_default_no_chunking(self, mock_read_frames, mock_header_from_file):
        """Test that default behavior processes all frames at once (no chunking)."""
        mock_header = self.create_mock_eer_header(n_frames=100, height=512, width=512)
        mock_header_from_file.return_value = mock_header

        # Mock read_frames to return all frames at once
        mock_read_frames.return_value = np.ones((100, 512, 512), dtype=np.uint16)

        # Test without specifying chunk_size (should process all frames at once)
        result = eerfile.render("test.eer", dose_per_output_frame=1.0)

        # Should call read_frames only once (no chunking)
        assert mock_read_frames.call_count == 1
        assert result.shape == (
            10,
            512,
            512,
        )  # 100 frames / 10 frames per output = 10 output frames

    @patch("eerfile.render_eer.EERHeader.from_file")
    @patch("eerfile.render_eer.read_frames")
    def test_render_memory_error_handling(
        self, mock_read_frames, mock_header_from_file
    ):
        """Test memory error handling and helpful error messages."""
        mock_header = self.create_mock_eer_header(n_frames=100, height=512, width=512)
        mock_header_from_file.return_value = mock_header

        # Mock read_frames to raise MemoryError
        mock_read_frames.side_effect = MemoryError("Not enough memory")

        with pytest.raises(MemoryError) as exc_info:
            eerfile.render("test.eer", dose_per_output_frame=1.0, chunk_size=50)

        # Check that the error message provides helpful guidance
        assert "chunk_size" in str(exc_info.value)
        assert "25" in str(exc_info.value)  # Should suggest chunk_size // 2

    @patch("eerfile.render_eer.EERHeader.from_file")
    @patch("eerfile.render_eer.read_frames")
    def test_render_with_custom_fluence(self, mock_read_frames, mock_header_from_file):
        """Test rendering with custom total fluence."""
        mock_header = self.create_mock_eer_header(n_frames=40, height=4, width=4)
        mock_header_from_file.return_value = mock_header

        mock_read_frames.return_value = np.ones((20, 4, 4), dtype=np.uint16)

        result = eerfile.render(
            "test.eer",
            dose_per_output_frame=1.0,
            total_fluence=2.0,  # Custom fluence
        )

        # With custom fluence of 2.0 and 40 frames: dose_per_eer_frame = 2.0/40 = 0.05
        # eer_frames_per_output_frame = round(1.0/0.05) = 20
        # n_output_frames = floor(40/20) = 2
        assert result.shape == (2, 4, 4)

    @patch("eerfile.render_eer.EERHeader.from_file")
    @patch("eerfile.render_eer.read_frames")
    def test_render_thread_pool_usage(self, mock_read_frames, mock_header_from_file):
        """Test that render uses thread pool correctly."""
        mock_header = self.create_mock_eer_header(n_frames=20, height=4, width=4)
        mock_header_from_file.return_value = mock_header

        mock_read_frames.return_value = np.ones((10, 4, 4), dtype=np.uint16)

        # Test with limited worker threads
        result = eerfile.render("test.eer", dose_per_output_frame=1.0, max_workers=2)

        assert result.shape == (2, 4, 4)
        mock_read_frames.assert_called()


class TestEERHeader:
    """Test the EERHeader data model."""

    @patch("tifffile.TiffFile")
    def test_eer_header_from_file(self, mock_tiff_file):
        """Test parsing EER header from file with dict metadata."""
        # Mock TiffFile with dict metadata (newer tifffile format)
        mock_tiff = Mock()
        mock_tiff.eer_metadata = {
            "acquisitionID": "test_acquisition_123",
            "cameraName": "TestCamera",
            "commercialName": "TestCommercial",
            "exposureTime": "1.5",
            "meanDoseRate": "2.5",
            "totalDose": "10.0",
            "numberOfFrames": "100",
            "sensorImageHeight": "512",
            "sensorImageWidth": "512",
            "sensorPixelSize": {"height": "1.0e-10", "width": "1.0e-10"},
            "serialNumber": "SN12345",
        }
        mock_tiff_file.return_value.__enter__.return_value = mock_tiff

        header = EERHeader.from_file("test.eer")

        assert header.acquisition_id == "test_acquisition_123"
        assert header.camera_name == "TestCamera"
        assert header.commercial_name == "TestCommercial"
        assert header.exposure_time_seconds == 1.5
        assert header.dose_rate_electrons_per_pixel_per_second == 2.5
        assert header.total_dose_electrons_per_pixel == 10.0
        assert header.n_frames == 100
        assert header.image_height_pixels == 512
        assert header.image_width_pixels == 512
        assert header.pixel_spacing_height_meters == 1.0e-10
        assert header.pixel_spacing_width_meters == 1.0e-10
        assert header.serial_number == "SN12345"

    @patch("tifffile.TiffFile")
    def test_eer_header_from_file_nested_dict(self, mock_tiff_file):
        """Test parsing EER header with nested dict structure."""
        # Mock TiffFile with nested dict metadata
        mock_tiff = Mock()
        mock_tiff.eer_metadata = {
            "acquisitionID": "test_123",
            "numberOfFrames": "50",
            "sensorImageHeight": "256",
            "sensorImageWidth": "256",
            "sensorPixelSize": {
                "height": "2.0e-10",
                "width": "2.0e-10",
            },
        }
        mock_tiff_file.return_value.__enter__.return_value = mock_tiff

        header = EERHeader.from_file("test.eer")

        assert header.n_frames == 50
        assert header.image_height_pixels == 256
        assert header.image_width_pixels == 256
        assert header.pixel_spacing_height_meters == 2.0e-10
        assert header.pixel_spacing_width_meters == 2.0e-10

    @patch("tifffile.TiffFile")
    def test_eer_header_from_file_missing_fields(self, mock_tiff_file):
        """Test parsing EER header with missing optional fields."""
        # Mock TiffFile with minimal metadata
        mock_tiff = Mock()
        mock_tiff.eer_metadata = {
            "numberOfFrames": "10",
            "sensorImageHeight": "128",
            "sensorImageWidth": "128",
            "sensorPixelSize": {"height": "1.0e-10", "width": "1.0e-10"},
        }
        mock_tiff_file.return_value.__enter__.return_value = mock_tiff

        header = EERHeader.from_file("test.eer")

        # Required fields should have defaults
        assert header.acquisition_id == ""
        assert header.camera_name == ""
        assert header.commercial_name == ""
        assert header.serial_number == ""
        # Optional fields should be None
        assert header.exposure_time_seconds is None
        assert header.dose_rate_electrons_per_pixel_per_second is None
        assert header.total_dose_electrons_per_pixel is None
        # Required fields should be set
        assert header.n_frames == 10
        assert header.image_height_pixels == 128
        assert header.image_width_pixels == 128


class TestBasicFunctionality:
    """Test basic read and render functionality."""

    def create_mock_eer_header(self, n_frames=100, height=512, width=512):
        """Create a mock EER header for testing."""
        header = Mock(spec=EERHeader)
        header.n_frames = n_frames
        header.image_height_pixels = height
        header.image_width_pixels = width
        header.dose_per_frame_electrons_per_square_angstrom = 0.1
        return header

    @patch("eerfile.read_eer.TiffFile")
    def test_original_read_function(self, mock_tiff_file):
        """Test that the original read function still works."""
        mock_data = np.random.randint(0, 100, size=(10, 4, 4), dtype=np.uint16)
        mock_tiff = Mock()
        mock_tiff.asarray.return_value = mock_data
        mock_tiff_file.return_value.__enter__.return_value = mock_tiff

        result = eerfile.read("test.eer")

        np.testing.assert_array_equal(result, mock_data)
        mock_tiff.asarray.assert_called_once_with()

    @patch("eerfile.render_eer.EERHeader.from_file")
    @patch("eerfile.render_eer.read_frames")
    def test_original_render_signature(self, mock_read_frames, mock_header_from_file):
        """Test that render works with original signature (backward compatibility)."""
        mock_header = self.create_mock_eer_header(n_frames=20, height=4, width=4)
        mock_header_from_file.return_value = mock_header

        mock_read_frames.return_value = np.ones((10, 4, 4), dtype=np.uint16)

        # Test original signature (no new parameters)
        result = eerfile.render("test.eer", dose_per_output_frame=1.0)

        assert result.shape == (2, 4, 4)
        mock_read_frames.assert_called()
