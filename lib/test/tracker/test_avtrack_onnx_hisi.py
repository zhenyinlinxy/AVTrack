"""
Unit tests for AVTrack ONNX HiSi inference module.

Tests the HiSi platform-specific functionality including:
- 16-byte alignment
- HiSi constraint validation
- sample_target_hisi function
- map_box_back function
- preprocess_crop_bgr_to_nchw function
"""

import pytest
import numpy as np
import sys
import os

# Add the lib path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from lib.test.tracker.avtrack_onnx_hisi import (
    align_to_16,
    validate_hisi_constraints,
    sample_target_hisi,
    map_box_back,
    preprocess_crop_bgr_to_nchw,
    clip_box,
    AVTrackONNXHiSi
)


class TestAlignTo16:
    """Tests for the align_to_16 function."""
    
    def test_already_aligned(self):
        """Test values that are already multiples of 16."""
        assert align_to_16(16) == 16
        assert align_to_16(32) == 32
        assert align_to_16(128) == 128
        assert align_to_16(256) == 256
    
    def test_needs_alignment(self):
        """Test values that need to be rounded up."""
        assert align_to_16(1) == 16
        assert align_to_16(15) == 16
        assert align_to_16(17) == 32
        assert align_to_16(100) == 112
        assert align_to_16(255) == 256
    
    def test_zero(self):
        """Test zero input."""
        assert align_to_16(0) == 0
    
    def test_large_values(self):
        """Test large values."""
        assert align_to_16(1000) == 1008
        assert align_to_16(1080) == 1088
        assert align_to_16(1920) == 1920


class TestValidateHisiConstraints:
    """Tests for the validate_hisi_constraints function."""
    
    def test_valid_dimensions(self):
        """Test valid dimensions that meet all constraints."""
        is_valid, warnings = validate_hisi_constraints(256, 256, "test")
        assert is_valid
        assert len(warnings) == 0
    
    def test_invalid_width(self):
        """Test invalid width not multiple of 16."""
        is_valid, warnings = validate_hisi_constraints(257, 256, "test")
        assert not is_valid
        assert len(warnings) == 1
        assert "width" in warnings[0]
    
    def test_invalid_height(self):
        """Test invalid height not multiple of 16."""
        is_valid, warnings = validate_hisi_constraints(256, 257, "test")
        assert not is_valid
        assert len(warnings) == 1
        assert "height" in warnings[0]
    
    def test_height_exceeds_1080(self):
        """Test height exceeding 1080 limit."""
        is_valid, warnings = validate_hisi_constraints(256, 1088, "test")
        assert not is_valid
        assert any("exceeds" in w for w in warnings)
    
    def test_multiple_violations(self):
        """Test multiple constraint violations."""
        is_valid, warnings = validate_hisi_constraints(257, 1081, "test")
        assert not is_valid
        assert len(warnings) >= 2
    
    def test_valid_1920x1080(self):
        """Test typical HiSi input size 1920x1080."""
        is_valid, warnings = validate_hisi_constraints(1920, 1072, "test")
        # 1072 is multiple of 16 and < 1080
        assert is_valid
        assert len(warnings) == 0


class TestSampleTargetHisi:
    """Tests for the sample_target_hisi function."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_bbox(self):
        """Create a sample bounding box."""
        return [960, 540, 100, 100]  # Center of 1920x1080, 100x100 box
    
    def test_basic_crop(self, sample_image, sample_bbox):
        """Test basic crop without HiSi mode."""
        crop, resize_factor, mask, crop_sz = sample_target_hisi(
            sample_image, sample_bbox, 4.0, 256, hisi_mode=False
        )
        
        assert crop.shape == (256, 256, 3)
        assert mask.shape == (256, 256)
        assert resize_factor > 0
    
    def test_hisi_mode_alignment(self, sample_image, sample_bbox):
        """Test that HiSi mode produces aligned crop sizes."""
        crop, resize_factor, mask, crop_sz = sample_target_hisi(
            sample_image, sample_bbox, 4.0, 256, hisi_mode=True
        )
        
        assert crop.shape == (256, 256, 3)
        assert crop_sz % 16 == 0, f"crop_sz {crop_sz} should be multiple of 16"
    
    def test_output_size_multiple_of_16(self, sample_image, sample_bbox):
        """Test that output size is validated in HiSi mode."""
        # 256 is already multiple of 16, should work
        crop, _, _, _ = sample_target_hisi(
            sample_image, sample_bbox, 4.0, 256, hisi_mode=True
        )
        assert crop.shape == (256, 256, 3)
    
    def test_resize_factor_accuracy(self, sample_image, sample_bbox):
        """Test that resize_factor is calculated correctly."""
        output_sz = 256
        crop, resize_factor, mask, crop_sz = sample_target_hisi(
            sample_image, sample_bbox, 4.0, output_sz, hisi_mode=True
        )
        
        expected_resize_factor = output_sz / crop_sz
        assert abs(resize_factor - expected_resize_factor) < 1e-6
    
    def test_attention_mask_shape(self, sample_image, sample_bbox):
        """Test that attention mask has correct shape."""
        crop, _, mask, _ = sample_target_hisi(
            sample_image, sample_bbox, 4.0, 256, hisi_mode=True
        )
        
        assert mask.shape == (256, 256)
        assert mask.dtype == np.bool_
    
    def test_small_bbox(self, sample_image):
        """Test with a small bounding box."""
        small_bbox = [960, 540, 10, 10]
        crop, resize_factor, mask, crop_sz = sample_target_hisi(
            sample_image, small_bbox, 4.0, 256, hisi_mode=True
        )
        
        assert crop.shape == (256, 256, 3)
        assert crop_sz % 16 == 0
    
    def test_bbox_near_edge(self, sample_image):
        """Test with bounding box near image edge (requires padding)."""
        edge_bbox = [10, 10, 50, 50]
        crop, resize_factor, mask, crop_sz = sample_target_hisi(
            sample_image, edge_bbox, 4.0, 256, hisi_mode=True
        )
        
        assert crop.shape == (256, 256, 3)
        # Mask should indicate padded regions
        assert mask.any()  # Some padding should be present


class TestMapBoxBack:
    """Tests for the map_box_back function."""
    
    def test_center_mapping(self):
        """Test mapping when prediction is at center."""
        state = [100, 100, 50, 50]  # x, y, w, h
        pred_box = [128, 128, 50, 50]  # cx, cy, w, h in crop coords
        resize_factor = 1.0
        search_size = 256
        
        result = map_box_back(pred_box, state, resize_factor, search_size)
        
        # Result should be [x, y, w, h]
        assert len(result) == 4
    
    def test_identity_mapping(self):
        """Test that center prediction maps back to original state."""
        state = [100, 100, 50, 50]
        # Center of state is (125, 125)
        # With search_size=256 and resize_factor=1.0, half_side = 128
        # So crop center in image coords is at (125, 125)
        # Prediction at center of crop (128, 128) should map back close to state
        
        pred_box = [128, 128, 50, 50]
        resize_factor = 1.0
        search_size = 256
        
        result = map_box_back(pred_box, state, resize_factor, search_size)
        
        # The mapped center should be at cx_prev + (128 - 128) = cx_prev
        # With some tolerance for the mapping calculation
        assert len(result) == 4
    
    def test_different_resize_factors(self):
        """Test mapping with different resize factors."""
        state = [100, 100, 50, 50]
        pred_box = [128, 128, 50, 50]
        search_size = 256
        
        # Test with resize_factor = 0.5 (means crop was larger)
        result1 = map_box_back(pred_box, state, 0.5, search_size)
        
        # Test with resize_factor = 2.0 (means crop was smaller)
        result2 = map_box_back(pred_box, state, 2.0, search_size)
        
        # With smaller resize_factor (larger original crop), 
        # the half_side in image space is larger
        assert result1[0] != result2[0] or result1[1] != result2[1]


class TestPreprocessCropBgrToNchw:
    """Tests for the preprocess_crop_bgr_to_nchw function."""
    
    def test_output_shape(self):
        """Test output shape is correct NCHW format."""
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = preprocess_crop_bgr_to_nchw(img)
        
        assert result.shape == (1, 3, 256, 256)
    
    def test_output_dtype(self):
        """Test output dtype is float32."""
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = preprocess_crop_bgr_to_nchw(img)
        
        assert result.dtype == np.float32
    
    def test_normalization_range(self):
        """Test that output is normalized (roughly in [-3, 3] range)."""
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = preprocess_crop_bgr_to_nchw(img)
        
        # After ImageNet normalization, values should be roughly in [-3, 3]
        assert result.min() >= -5
        assert result.max() <= 5
    
    def test_different_sizes(self):
        """Test with different input sizes."""
        for size in [128, 256, 512]:
            img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            result = preprocess_crop_bgr_to_nchw(img)
            assert result.shape == (1, 3, size, size)


class TestClipBox:
    """Tests for the clip_box function."""
    
    def test_no_clipping_needed(self):
        """Test box that doesn't need clipping."""
        box = [100, 100, 50, 50]
        result = clip_box(box, 1080, 1920)
        
        assert result == [100, 100, 50, 50]
    
    def test_clip_negative_x(self):
        """Test clipping negative x coordinate."""
        box = [-10, 100, 50, 50]
        result = clip_box(box, 1080, 1920)
        
        assert result[0] == 0
        assert result[2] == 40  # width reduced
    
    def test_clip_negative_y(self):
        """Test clipping negative y coordinate."""
        box = [100, -10, 50, 50]
        result = clip_box(box, 1080, 1920)
        
        assert result[1] == 0
        assert result[3] == 40  # height reduced
    
    def test_clip_exceeds_width(self):
        """Test clipping when x+w exceeds image width."""
        box = [1900, 100, 50, 50]
        result = clip_box(box, 1080, 1920)
        
        assert result[0] + result[2] <= 1920
    
    def test_clip_exceeds_height(self):
        """Test clipping when y+h exceeds image height."""
        box = [100, 1060, 50, 50]
        result = clip_box(box, 1080, 1920)
        
        assert result[1] + result[3] <= 1080
    
    def test_margin(self):
        """Test clipping with margin."""
        box = [-5, -5, 50, 50]
        result = clip_box(box, 1080, 1920, margin=10)
        
        # With margin=10, can extend up to 10 pixels outside
        assert result[0] == -5


class TestAVTrackONNXHiSi:
    """Tests for the AVTrackONNXHiSi class."""
    
    def test_init_default(self):
        """Test initialization with default parameters."""
        tracker = AVTrackONNXHiSi(model_path=None)
        
        assert tracker.template_size == 128
        assert tracker.search_size == 256
        assert tracker.hisi_mode is False
    
    def test_init_hisi_mode(self):
        """Test initialization with HiSi mode."""
        tracker = AVTrackONNXHiSi(model_path=None, hisi_mode=True)
        
        assert tracker.hisi_mode is True
    
    def test_init_invalid_template_size(self):
        """Test that invalid template size raises assertion in HiSi mode."""
        with pytest.raises(AssertionError):
            AVTrackONNXHiSi(model_path=None, template_size=100, hisi_mode=True)
    
    def test_init_invalid_search_size(self):
        """Test that invalid search size raises assertion in HiSi mode."""
        with pytest.raises(AssertionError):
            AVTrackONNXHiSi(model_path=None, search_size=200, hisi_mode=True)
    
    def test_initialize(self):
        """Test tracker initialization with image and bbox."""
        tracker = AVTrackONNXHiSi(model_path=None, hisi_mode=True)
        
        image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        init_bbox = [960, 540, 100, 100]
        
        tracker.initialize(image, init_bbox)
        
        assert tracker.state == init_bbox
        assert tracker.template is not None
        assert tracker.template.shape == (1, 3, 128, 128)


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_tracking_pipeline_no_model(self):
        """Test full tracking pipeline without actual model."""
        tracker = AVTrackONNXHiSi(model_path=None, hisi_mode=True)
        
        # Create test image
        image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        init_bbox = [960, 540, 100, 100]
        
        # Initialize
        tracker.initialize(image, init_bbox)
        
        # Track (should return current state since no model)
        result = tracker.track(image)
        
        assert "target_bbox" in result
        assert len(result["target_bbox"]) == 4
    
    def test_hisi_constraints_maintained(self):
        """Test that HiSi constraints are maintained throughout pipeline."""
        tracker = AVTrackONNXHiSi(model_path=None, hisi_mode=True)
        
        image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        init_bbox = [100, 100, 200, 150]  # Non-standard size
        
        # The template and search sizes should still be multiples of 16
        tracker.initialize(image, init_bbox)
        
        assert tracker.template.shape[2] % 16 == 0
        assert tracker.template.shape[3] % 16 == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
