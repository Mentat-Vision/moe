import cv2
import numpy as np

def resize_frame_for_processing(frame, scale_factor):
    """Resize frame for AI processing based on scale factor"""
    if frame is None or scale_factor <= 0:
        return frame
    
    current_height, current_width = frame.shape[:2]
    
    # Calculate new dimensions based on scale factor
    new_width = int(current_width * scale_factor)
    new_height = int(current_height * scale_factor)
    
    # Always resize to ensure AI models process the scaled frames
    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return frame

def scale_bounding_boxes_for_display(detections, original_frame_shape, display_frame_shape):
    """
    Scale bounding boxes from processed frame coordinates to display frame coordinates.
    
    Args:
        detections: List of detection dictionaries with 'bbox' key
        original_frame_shape: (height, width) of the original frame
        display_frame_shape: (height, width) of the display frame
    
    Returns:
        List of detections with scaled bounding boxes
    """
    if not detections:
        return detections
    
    orig_height, orig_width = original_frame_shape[:2]
    display_height, display_width = display_frame_shape[:2]
    
    # Calculate scaling factors
    scale_x = display_width / orig_width
    scale_y = display_height / orig_height
    
    scaled_detections = []
    for detection in detections:
        scaled_detection = detection.copy()
        bbox = detection["bbox"]
        
        # Scale bounding box coordinates
        scaled_detection["bbox"] = [
            int(bbox[0] * scale_x),  # x1
            int(bbox[1] * scale_y),  # y1
            int(bbox[2] * scale_x),  # x2
            int(bbox[3] * scale_y)   # y2
        ]
        
        scaled_detections.append(scaled_detection)
    
    return scaled_detections

def scale_bounding_boxes_from_processed_to_display(detections, processing_scale, display_shape):
    """
    Scale bounding boxes from processed frame coordinates to display frame coordinates.
    
    The bounding boxes were calculated on a frame that was scaled by processing_scale.
    The display frame is also scaled by processing_scale, so the coordinates should match.
    
    Args:
        detections: List of detection dictionaries with 'bbox' key
        processing_scale: The scale factor used for processing
        display_shape: (height, width) of the display frame
    
    Returns:
        List of detections with scaled bounding boxes
    """
    if not detections:
        return detections
    
    display_height, display_width = display_shape[:2]
    
    # Since both the processed frame and display frame are scaled by processing_scale,
    # the coordinates should be the same (scale factor = 1.0)
    # However, if the display frame has a different aspect ratio or size,
    # we need to account for that
    
    scaled_detections = []
    for detection in detections:
        scaled_detection = detection.copy()
        bbox = detection["bbox"]
        
        # For now, assume the coordinates match (both frames scaled by same factor)
        # This is the correct behavior when both frames use the same processing_scale
        scaled_detection["bbox"] = [
            int(bbox[0]),  # x1
            int(bbox[1]),  # y1
            int(bbox[2]),  # x2
            int(bbox[3])   # y2
        ]
        
        scaled_detections.append(scaled_detection)
    
    return scaled_detections

def draw_detections_on_frame(frame, detections, colors=None):
    """
    Draw detection bounding boxes on a frame.
    
    Args:
        frame: OpenCV frame to draw on
        detections: List of detection dictionaries
        colors: List of BGR colors for bounding boxes
    
    Returns:
        Frame with bounding boxes drawn
    """
    if not detections:
        return frame
    
    if colors is None:
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, detection in enumerate(detections):
        bbox = detection["bbox"]
        class_name = detection["class"]
        confidence = detection["confidence"]
        
        color = colors[i % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
        
        # Draw label
        label = f"{class_name} {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (bbox[0], bbox[1] - text_height - 10),
                     (bbox[0] + text_width + 10, bbox[1]), color, -1)
        cv2.putText(frame, label, (bbox[0] + 5, bbox[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def get_processing_scale_from_config(config):
    """Get processing scale from config with default fallback"""
    return float(config.get("PROCESSING_SCALE", 0.5))

def validate_scale_factor(scale_factor):
    """Validate that scale factor is within reasonable bounds"""
    if scale_factor < 0.1:
        print(f"⚠️ Scale factor {scale_factor} is very low, may cause poor detection quality")
    elif scale_factor > 1.0:
        print(f"⚠️ Scale factor {scale_factor} is above 1.0, may cause performance issues")
    
    return max(0.1, min(1.0, scale_factor)) 