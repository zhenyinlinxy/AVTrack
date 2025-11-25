"""
AVTrack ONNX Inference with HiSi (Hisilicon) Platform Support

This module provides ONNX inference capabilities optimized for HiSi platform hardware
acceleration. The HiSi platform has specific hardware limitations:
1. resize and crop function input/output dimensions must be multiples of 16
2. resize function input height cannot exceed 1080
3. Input image size is 1920x1080

Usage:
    python -m lib.test.tracker.avtrack_onnx_hisi --model path/to/model.onnx --video path/to/video.mp4
    python -m lib.test.tracker.avtrack_onnx_hisi --model path/to/model.onnx --image path/to/image.jpg --hisi_mode
"""

import math
import cv2
import numpy as np
import argparse
import os
from typing import Tuple, Optional, List


def align_to_16(size: int) -> int:
    """Align size to multiple of 16 (round up).
    
    Args:
        size: The size value to align
        
    Returns:
        The size rounded up to the nearest multiple of 16
    """
    return ((size + 15) // 16) * 16


def validate_hisi_constraints(width: int, height: int, context: str = "") -> Tuple[bool, List[str]]:
    """Validate that dimensions meet HiSi platform constraints.
    
    Args:
        width: Width dimension to validate
        height: Height dimension to validate
        context: Description of the operation being validated
        
    Returns:
        Tuple of (is_valid, list of warning messages)
    """
    warnings = []
    is_valid = True
    
    if width % 16 != 0:
        warnings.append(f"{context}: width {width} is not a multiple of 16")
        is_valid = False
    
    if height % 16 != 0:
        warnings.append(f"{context}: height {height} is not a multiple of 16")
        is_valid = False
    
    if height > 1080:
        warnings.append(f"{context}: height {height} exceeds HiSi limit of 1080")
        is_valid = False
        
    return is_valid, warnings


def sample_target_hisi(im: np.ndarray, target_bb: list, search_area_factor: float, 
                       output_sz: int, hisi_mode: bool = False) -> Tuple[np.ndarray, float, np.ndarray, int]:
    """Extract a square crop centered at target bounding box with HiSi alignment support.
    
    This function extracts a square crop centered at the target bounding box,
    with area equal to search_area_factor^2 times the target area. When hisi_mode
    is enabled, all dimensions are aligned to 16-pixel boundaries.
    
    Args:
        im: Input image (BGR format, HxWx3)
        target_bb: Target bounding box [x, y, w, h]
        search_area_factor: Ratio of crop size to target size
        output_sz: Size to resize the extracted crop (always square)
        hisi_mode: If True, align crop size to 16-pixel boundary
        
    Returns:
        Tuple of:
            - im_crop_padded: Extracted and resized crop
            - resize_factor: Factor by which the crop was resized
            - att_mask: Attention mask (False where valid, True where padded)
            - actual_crop_sz: Actual crop size used (may differ from calculated if hisi_mode)
    """
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb
    
    # Calculate crop size
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
    
    if crop_sz < 1:
        raise ValueError('Too small bounding box.')
    
    # In HiSi mode, align crop_sz to 16
    actual_crop_sz = crop_sz
    if hisi_mode:
        actual_crop_sz = align_to_16(crop_sz)
        
        # Handle height limit - if crop exceeds 1080, cap at 1072 (multiple of 16 < 1080)
        # This ensures the resize input stays under 1080 height limit
        if actual_crop_sz > 1080:
            print(f"[HiSi Info] Crop size {actual_crop_sz} exceeded 1080, capping at 1072")
            actual_crop_sz = 1072
        
        # Validate HiSi constraints for crop
        is_valid, warnings = validate_hisi_constraints(actual_crop_sz, actual_crop_sz, "crop")
        for warning in warnings:
            print(f"[HiSi Warning] {warning}")
    
    # Calculate crop boundaries centered on target
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    
    x1 = round(cx - actual_crop_sz * 0.5)
    x2 = x1 + actual_crop_sz
    y1 = round(cy - actual_crop_sz * 0.5)
    y2 = y1 + actual_crop_sz
    
    # Calculate padding needed
    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)
    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)
    
    # Crop target region from image
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    
    # Pad to make square crop
    im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, 
                                         cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    # Create attention mask (True = padded/invalid, False = valid)
    H, W = im_crop_padded.shape[:2]
    att_mask = np.ones((H, W), dtype=np.bool_)
    end_x = -x2_pad if x2_pad > 0 else None
    end_y = -y2_pad if y2_pad > 0 else None
    att_mask[y1_pad:end_y, x1_pad:end_x] = False
    
    # Calculate resize factor based on actual crop size
    resize_factor = output_sz / actual_crop_sz
    
    # Validate output size for HiSi
    if hisi_mode:
        is_valid, warnings = validate_hisi_constraints(output_sz, output_sz, "output resize")
        for warning in warnings:
            print(f"[HiSi Warning] {warning}")
    
    # Resize to output size
    im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
    att_mask = cv2.resize(att_mask.astype(np.uint8), (output_sz, output_sz)).astype(np.bool_)
    
    return im_crop_padded, resize_factor, att_mask, actual_crop_sz


def map_box_back(pred_box: list, state: list, resize_factor: float, 
                 search_size: int) -> list:
    """Map predicted box coordinates from crop space back to image space.
    
    Args:
        pred_box: Predicted box in crop coordinates [cx, cy, w, h]
        state: Previous state/bounding box [x, y, w, h]
        resize_factor: Factor by which the crop was resized
        search_size: Size of the search region (crop output size)
        
    Returns:
        Mapped bounding box in image coordinates [x, y, w, h]
    """
    # Center of previous state
    cx_prev = state[0] + 0.5 * state[2]
    cy_prev = state[1] + 0.5 * state[3]
    
    cx, cy, w, h = pred_box
    
    # Half side of the original crop in image space
    half_side = 0.5 * search_size / resize_factor
    
    # Map center back to image coordinates
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    
    return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


def preprocess_crop_bgr_to_nchw(img_arr: np.ndarray) -> np.ndarray:
    """Preprocess BGR image crop to normalized NCHW format for ONNX inference.
    
    Args:
        img_arr: BGR image array (H, W, 3)
        
    Returns:
        Preprocessed array in NCHW format (1, 3, H, W), float32, normalized
    """
    # ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
    
    # Convert BGR to RGB and rearrange to NCHW
    img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    img_4d = img_rgb[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
    
    # Normalize
    img_normalized = (img_4d / 255.0 - mean) / std
    
    return img_normalized.astype(np.float32)


def clip_box(box: list, H: int, W: int, margin: int = 0) -> list:
    """Clip bounding box to image boundaries with optional margin.
    
    Args:
        box: Bounding box [x, y, w, h]
        H: Image height
        W: Image width
        margin: Margin to allow outside boundaries
        
    Returns:
        Clipped bounding box [x, y, w, h]
    """
    x, y, w, h = box
    
    x1 = max(0 - margin, x)
    y1 = max(0 - margin, y)
    x2 = min(W + margin, x + w)
    y2 = min(H + margin, y + h)
    
    return [x1, y1, x2 - x1, y2 - y1]


class AVTrackONNXHiSi:
    """AVTrack ONNX Tracker with HiSi platform support."""
    
    def __init__(self, model_path: str, template_size: int = 128, 
                 search_size: int = 256, template_factor: float = 2.0,
                 search_factor: float = 4.0, hisi_mode: bool = False):
        """Initialize the tracker.
        
        Args:
            model_path: Path to ONNX model file
            template_size: Template image size (default 128, must be multiple of 16)
            search_size: Search region size (default 256, must be multiple of 16)
            template_factor: Factor for template extraction
            search_factor: Factor for search region extraction
            hisi_mode: Enable HiSi platform optimizations
        """
        self.model_path = model_path
        self.template_size = template_size
        self.search_size = search_size
        self.template_factor = template_factor
        self.search_factor = search_factor
        self.hisi_mode = hisi_mode
        
        # Validate sizes for HiSi mode
        if hisi_mode:
            assert template_size % 16 == 0, f"template_size must be multiple of 16, got {template_size}"
            assert search_size % 16 == 0, f"search_size must be multiple of 16, got {search_size}"
            print(f"[HiSi Info] HiSi mode enabled")
            print(f"[HiSi Info] Template size: {template_size}, Search size: {search_size}")
        
        # State variables
        self.state = None  # Current target state [x, y, w, h]
        self.template = None  # Template features
        self.template_crop_sz = None  # Actual template crop size
        
        # Load ONNX model
        self.session = None
        if model_path and os.path.exists(model_path):
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(model_path)
                print(f"[Info] ONNX model loaded from {model_path}")
            except ImportError:
                print("[Warning] onnxruntime not installed, running in demo mode")
            except Exception as e:
                print(f"[Warning] Failed to load ONNX model: {e}")
    
    def initialize(self, image: np.ndarray, init_bbox: list):
        """Initialize tracker with first frame and bounding box.
        
        Args:
            image: First frame (BGR format)
            init_bbox: Initial bounding box [x, y, w, h]
        """
        # Extract template
        z_patch, resize_factor, z_mask, crop_sz = sample_target_hisi(
            image, init_bbox, self.template_factor, 
            self.template_size, self.hisi_mode
        )
        
        self.template_crop_sz = crop_sz
        
        # Preprocess template
        self.template = preprocess_crop_bgr_to_nchw(z_patch)
        
        # Save initial state
        self.state = list(init_bbox)
        
        if self.hisi_mode:
            print(f"[HiSi Info] Template extracted with crop_sz={crop_sz}, "
                  f"resize_factor={resize_factor:.4f}")
    
    def track(self, image: np.ndarray) -> dict:
        """Track object in a new frame.
        
        Args:
            image: Current frame (BGR format)
            
        Returns:
            Dictionary with 'target_bbox' key containing [x, y, w, h]
        """
        H, W = image.shape[:2]
        
        # Extract search region
        x_patch, resize_factor, x_mask, crop_sz = sample_target_hisi(
            image, self.state, self.search_factor,
            self.search_size, self.hisi_mode
        )
        
        # Preprocess search region
        search = preprocess_crop_bgr_to_nchw(x_patch)
        
        if self.hisi_mode:
            # Log dimensions for debugging
            print(f"[HiSi Debug] Search crop_sz={crop_sz}, "
                  f"resize_factor={resize_factor:.4f}, "
                  f"search shape={search.shape}")
        
        # Run inference
        pred_box = self._run_inference(search)
        
        if pred_box is None:
            # If no inference available, return current state
            return {"target_bbox": self.state}
        
        # Scale prediction by search_size and resize_factor
        pred_box = [p * self.search_size / resize_factor for p in pred_box]
        
        # Map box back to image coordinates
        new_state = map_box_back(pred_box, self.state, resize_factor, self.search_size)
        
        # Clip to image boundaries
        self.state = clip_box(new_state, H, W, margin=10)
        
        return {"target_bbox": self.state}
    
    def _run_inference(self, search: np.ndarray) -> Optional[list]:
        """Run ONNX inference.
        
        Args:
            search: Preprocessed search region (1, 3, H, W)
            
        Returns:
            Predicted box [cx, cy, w, h] or None if no model loaded
        """
        if self.session is None:
            return None
        
        try:
            # Get input names
            input_names = [inp.name for inp in self.session.get_inputs()]
            output_names = [out.name for out in self.session.get_outputs()]
            
            # Prepare inputs (template + search)
            inputs = {}
            if len(input_names) >= 2:
                inputs[input_names[0]] = self.template
                inputs[input_names[1]] = search
            else:
                inputs[input_names[0]] = search
            
            # Run inference
            outputs = self.session.run(output_names, inputs)
            
            # Parse output - assuming output is [cx, cy, w, h]
            if len(outputs) > 0:
                pred = outputs[0]
                if pred.ndim > 1:
                    pred = pred.flatten()
                return pred[:4].tolist()
            
        except Exception as e:
            print(f"[Error] Inference failed: {e}")
        
        return None


def process_single_image(tracker: AVTrackONNXHiSi, image_path: str, 
                         init_bbox: list, output_path: Optional[str] = None) -> list:
    """Process a single image with the tracker.
    
    Args:
        tracker: Initialized tracker
        image_path: Path to input image
        init_bbox: Initial bounding box [x, y, w, h]
        output_path: Optional path to save output image with visualization
        
    Returns:
        Final bounding box [x, y, w, h]
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    H, W = image.shape[:2]
    
    # Validate input size for HiSi mode
    if tracker.hisi_mode:
        is_valid, warnings = validate_hisi_constraints(W, H, "input image")
        for warning in warnings:
            print(f"[HiSi Warning] {warning}")
    
    # Initialize and track (for single image, init is the result)
    tracker.initialize(image, init_bbox)
    result = {"target_bbox": tracker.state}
    
    # Save visualization if requested
    if output_path:
        x, y, w, h = [int(v) for v in result["target_bbox"]]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(output_path, image)
        print(f"[Info] Output saved to {output_path}")
    
    return result["target_bbox"]


def process_video(tracker: AVTrackONNXHiSi, video_path: str, 
                  init_bbox: Optional[list] = None,
                  output_path: Optional[str] = None,
                  display: bool = False) -> List[list]:
    """Process a video with the tracker.
    
    Args:
        tracker: Tracker instance
        video_path: Path to input video
        init_bbox: Initial bounding box [x, y, w, h]. If None, user selects.
        output_path: Optional path to save output video
        display: Whether to display tracking in real-time
        
    Returns:
        List of bounding boxes for each frame
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[Info] Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Validate video size for HiSi mode
    if tracker.hisi_mode:
        is_valid, warnings = validate_hisi_constraints(width, height, "video frame")
        for warning in warnings:
            print(f"[HiSi Warning] {warning}")
    
    # Setup output video writer
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    results = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx == 0:
            # Initialize on first frame
            if init_bbox is None:
                # Let user select ROI
                print("[Info] Select target region and press ENTER")
                roi = cv2.selectROI("Select Target", frame, False)
                cv2.destroyWindow("Select Target")
                init_bbox = list(roi)  # (x, y, w, h)
            
            tracker.initialize(frame, init_bbox)
            result = {"target_bbox": tracker.state}
        else:
            # Track in subsequent frames
            result = tracker.track(frame)
        
        results.append(result["target_bbox"])
        
        # Visualization
        x, y, w, h = [int(v) for v in result["target_bbox"]]
        vis_frame = frame.copy()
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Frame {frame_idx}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if out:
            out.write(vis_frame)
        
        if display:
            cv2.imshow("Tracking", vis_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"[Info] Processed {frame_idx}/{total_frames} frames")
    
    cap.release()
    if out:
        out.release()
        print(f"[Info] Output video saved to {output_path}")
    
    if display:
        cv2.destroyAllWindows()
    
    print(f"[Info] Tracking complete: {len(results)} frames processed")
    return results


def main():
    """Main entry point for command line usage."""
    parser = argparse.ArgumentParser(
        description='AVTrack ONNX Inference with HiSi Platform Support'
    )
    
    # Required arguments
    parser.add_argument('--model', type=str, default=None,
                        help='Path to ONNX model file')
    
    # Input source (one of these required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str, help='Path to input video')
    input_group.add_argument('--image', type=str, help='Path to input image')
    
    # Bounding box
    parser.add_argument('--bbox', type=float, nargs=4, default=None,
                        metavar=('X', 'Y', 'W', 'H'),
                        help='Initial bounding box (x y w h)')
    
    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output')
    
    # Tracker parameters
    parser.add_argument('--template_size', type=int, default=128,
                        help='Template size (default: 128)')
    parser.add_argument('--search_size', type=int, default=256,
                        help='Search region size (default: 256)')
    parser.add_argument('--template_factor', type=float, default=2.0,
                        help='Template extraction factor (default: 2.0)')
    parser.add_argument('--search_factor', type=float, default=4.0,
                        help='Search region factor (default: 4.0)')
    
    # HiSi mode
    parser.add_argument('--hisi_mode', action='store_true',
                        help='Enable HiSi platform optimization mode')
    
    # Display
    parser.add_argument('--display', action='store_true',
                        help='Display tracking results in real-time')
    
    # Debug
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    
    args = parser.parse_args()
    
    # Create tracker
    tracker = AVTrackONNXHiSi(
        model_path=args.model,
        template_size=args.template_size,
        search_size=args.search_size,
        template_factor=args.template_factor,
        search_factor=args.search_factor,
        hisi_mode=args.hisi_mode
    )
    
    if args.video:
        # Process video
        process_video(
            tracker=tracker,
            video_path=args.video,
            init_bbox=args.bbox,
            output_path=args.output,
            display=args.display
        )
    else:
        # Process single image
        if args.bbox is None:
            print("[Error] --bbox is required for single image processing")
            return
        
        process_single_image(
            tracker=tracker,
            image_path=args.image,
            init_bbox=args.bbox,
            output_path=args.output
        )


if __name__ == '__main__':
    main()
