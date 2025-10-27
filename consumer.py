import time
import yaml
import torch
from ultralytics import YOLO
import numpy as np
import cv2
import logging
import os
from datetime import datetime
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchConsumer:
    def __init__(self, config_path="config.yaml", visualize_mode=None):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Load YOLOv8 model
        self.model = YOLO(self.config['model_path'])
        self.device = self.config['device']
        self.batch_size = self.config['batch_size']
        self.batch_timeout = self.config['batch_timeout']
        self.confidence_threshold = self.config['confidence_threshold']
        self.model_input_size = tuple(self.config['model_input_size'])
        
        # Visualization settings
        self.visualize_mode = visualize_mode  # None, 'display', or 'save'
        self.viz_config = self.config.get('visualization', {})
        self.video_config = self.config.get('video_save', {})
        self.video_writers = {}
        
        # Create output directory for saved videos
        if self.visualize_mode == 'save':
            output_dir = self.video_config.get('output_dir', './output_videos')
            os.makedirs(output_dir, exist_ok=True)
            self.output_dir = output_dir
            logger.info(f"Video output directory: {self.output_dir}")
        
        # Create class name to ID mapping for efficient filtering
        self.class_name_to_id = {name: idx for idx, name in self.model.names.items()}
        self.class_id_to_name = self.model.names
        
        self.pending_frames = []
        self.last_batch_time = time.time()
        self.running = threading.Event()
        self.running.clear()
        
        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Model input size: {self.model_input_size}")
        logger.info(f"Visualization mode: {self.visualize_mode if self.visualize_mode else 'disabled'}")
        logger.info(f"Available classes: {list(self.class_name_to_id.keys())}")
    
    def start(self, producer):
        """Start consuming frames from producer"""
        self.running.set()
        self.producer = producer
        
        while self.running.is_set():
            # Collect frames for batch
            self._collect_frames()
            
            # Process batch if ready
            if self._should_process_batch():
                self._process_batch()
            
            # Handle OpenCV window events if displaying
            if self.visualize_mode == 'display':
                cv2.waitKey(1)
            
            time.sleep(0.001)  # Small sleep to prevent busy waiting
    
    def stop(self):
        """Stop the consumer"""
        self.running.clear()
        
        # Process any remaining frames
        if self.pending_frames:
            self._process_batch()
        
        # Close all video writers
        if self.visualize_mode == 'save':
            for camera_id, writer in self.video_writers.items():
                writer.release()
                logger.info(f"Closed video writer for camera: {camera_id}")
            logger.info("All video writers closed")
        
        # Close all OpenCV windows
        if self.visualize_mode == 'display':
            cv2.destroyAllWindows()
            logger.info("All display windows closed")
    
    def _collect_frames(self):
        """Collect frames from producer"""
        frame_data = self.producer.get_frame()
        if frame_data:
            self.pending_frames.append(frame_data)
    
    def _should_process_batch(self):
        """Determine if batch should be processed"""
        if not self.pending_frames:
            return False
        
        # Process if batch is full or timeout reached
        batch_full = len(self.pending_frames) >= self.batch_size
        timeout_reached = (time.time() - self.last_batch_time) >= self.batch_timeout
        
        return batch_full or timeout_reached
    
    def _get_class_ids_for_camera(self, classes_to_detect):
        """Convert class names to class IDs for filtering"""
        if not classes_to_detect:  # Empty list means detect all classes
            return None
        
        class_ids = []
        invalid_classes = []
        
        for class_name in classes_to_detect:
            if class_name in self.class_name_to_id:
                class_ids.append(self.class_name_to_id[class_name])
            else:
                invalid_classes.append(class_name)
        
        if invalid_classes:
            logger.warning(f"Invalid class names found: {invalid_classes}. "
                          f"Available classes: {list(self.class_name_to_id.keys())}")
        
        return class_ids if class_ids else None
    
    def _filter_detections_by_class(self, detections, allowed_class_ids):
        """Filter detections by allowed class IDs"""
        if allowed_class_ids is None:  # None means all classes allowed
            return detections
        
        filtered_detections = []
        for detection in detections:
            if detection['class_id'] in allowed_class_ids:
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def _filter_detections_by_roi(self, detections, roi_config, frame_shape):
        """Filter detections to only include those within ROI"""
        if not roi_config['enabled'] or not roi_config['coordinates']:
            return detections  # Return all detections if ROI is disabled
        
        filtered_detections = []
        
        # Create ROI mask for checking detection centers
        mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
        roi_points = np.array(roi_config['coordinates'], dtype=np.int32)
        cv2.fillPoly(mask, [roi_points], 255)
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Calculate center point of detection
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Check if center point is within ROI
            if (0 <= center_y < mask.shape[0] and 
                0 <= center_x < mask.shape[1] and 
                mask[center_y, center_x] > 0):
                filtered_detections.append(detection)
            else:
                logger.debug(f"Detection filtered out - center ({center_x}, {center_y}) outside ROI")
        
        return filtered_detections
    
    def _draw_roi_on_frame(self, frame, roi_config):
        """Draw ROI polygon on frame"""
        if not roi_config['enabled'] or not roi_config['coordinates']:
            return frame
        
        if not self.viz_config.get('show_roi', True):
            return frame
        
        roi_points = np.array(roi_config['coordinates'], dtype=np.int32)
        roi_color = tuple(self.viz_config.get('roi_color', [255, 0, 0]))
        roi_thickness = self.viz_config.get('roi_thickness', 2)
        
        # Draw ROI polygon
        cv2.polylines(frame, [roi_points], True, roi_color, roi_thickness)
        
        return frame
    
    def _visualize_detections(self, frame, detections, camera_name, roi_config):
        """Draw bounding boxes and labels on frame"""
        vis_frame = frame.copy()
        
        # Draw ROI if enabled
        vis_frame = self._draw_roi_on_frame(vis_frame, roi_config)
        
        # Get visualization colors
        box_color = tuple(self.viz_config.get('box_color', [0, 255, 0]))
        text_color = tuple(self.viz_config.get('text_color', [0, 255, 0]))
        box_thickness = self.viz_config.get('box_thickness', 2)
        text_thickness = self.viz_config.get('text_thickness', 2)
        font_scale = self.viz_config.get('font_scale', 0.6)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class']}: {det['confidence']:.2f}"
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), box_color, box_thickness)
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
            cv2.rectangle(vis_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), box_color, -1)
            
            # Draw label text
            cv2.putText(vis_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness)
        
        # Add camera name and detection count
        info_text = f"{camera_name} | Detections: {len(detections)}"
        cv2.putText(vis_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_frame
    
    def _display_frame(self, frame, camera_name):
        """Display frame in OpenCV window"""
        cv2.imshow(camera_name, frame)
    
    def _initialize_video_writer(self, camera_id, camera_name, frame_shape):
        """Initialize video writer for a camera if not exists"""
        if camera_id not in self.video_writers:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{camera_id}_{camera_name.replace(' ', '_')}_{timestamp}.mp4"
            filepath = os.path.join(self.output_dir, filename)
            
            # Get video properties
            height, width = frame_shape[:2]
            fps = self.video_config.get('fps', 30)
            codec = self.video_config.get('codec', 'mp4v')
            fourcc = cv2.VideoWriter_fourcc(*codec)
            
            # Create video writer
            writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
            
            if not writer.isOpened():
                logger.error(f"Failed to create video writer for {camera_name}: {filepath}")
                return False
            
            self.video_writers[camera_id] = writer
            logger.info(f"Created video writer for {camera_name}: {filepath}")
            return True
        
        return True
    
    def _save_frame_to_video(self, frame, camera_id, camera_name):
        """Save frame to video file"""
        # Initialize video writer if needed
        if not self._initialize_video_writer(camera_id, camera_name, frame.shape):
            return
        
        # Write frame
        self.video_writers[camera_id].write(frame)
    
    def _process_batch(self):
        """Process batch of frames through YOLOv8"""
        if not self.pending_frames:
            return
        
        batch_start_time = time.time()
        
        # Prepare batch
        batch_frames = []
        frame_metadata = []
        original_frames = []
        
        for frame_data in self.pending_frames:
            batch_frames.append(frame_data['frame'])
            original_frames.append(frame_data['original_frame'])
            frame_metadata.append({
                'camera_id': frame_data['camera_id'],
                'camera_name': frame_data['camera_name'],
                'roi_config': frame_data['roi_config'],
                'classes_to_detect': frame_data['classes_to_detect'],
                'timestamp': frame_data['timestamp'],
                'frame_shape': frame_data['frame'].shape
            })
        
        try:
            # Run inference on batch (frames already have ROI applied)
            results = self.model(
                batch_frames,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )
            
            # Process results
            self._handle_results(results, frame_metadata, original_frames)
            
            processing_time = time.time() - batch_start_time
            logger.info(f"Processed batch of {len(batch_frames)} frames in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
        
        finally:
            # Clear batch
            self.pending_frames.clear()
            self.last_batch_time = time.time()
    
    def _handle_results(self, results, frame_metadata, original_frames):
        """Handle inference results with class and ROI filtering"""
        for i, (result, metadata, original_frame) in enumerate(zip(results, frame_metadata, original_frames)):
            detections = []
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = box
                    class_name = self.class_id_to_name[cls_id]
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class': class_name,
                        'class_id': int(cls_id)
                    }
                    
                    detections.append(detection)
            
            # Filter by allowed classes for this camera
            allowed_class_ids = self._get_class_ids_for_camera(metadata['classes_to_detect'])
            class_filtered_detections = self._filter_detections_by_class(detections, allowed_class_ids)
            
            # Apply ROI filtering
            final_detections = self._filter_detections_by_roi(
                class_filtered_detections, 
                metadata['roi_config'], 
                metadata['frame_shape']
            )
            
            # Log results with filtering information
            if detections:
                class_filter_info = f"classes: {metadata['classes_to_detect']}" if metadata['classes_to_detect'] else "all classes"
                roi_info = "with ROI" if metadata['roi_config']['enabled'] else "full frame"
                
                logger.info(f"{metadata['camera_name']} ({roi_info}, {class_filter_info}): "
                          f"Raw: {len(detections)}, Class filtered: {len(class_filtered_detections)}, "
                          f"Final: {len(final_detections)}")
                
                for det in final_detections:
                    logger.info(f"  -> {det['class']}: {det['confidence']:.2f} at {det['bbox']}")
            
            # Prepare frame for visualization/saving
            # Resize original frame to model input size for consistent visualization
            display_frame = cv2.resize(original_frame, self.model_input_size)
            
            # Draw detections and ROI on frame (even if no detections)
            annotated_frame = self._visualize_detections(
                display_frame, 
                final_detections, 
                metadata['camera_name'],
                metadata['roi_config']
            )
            
            # Handle visualization based on mode
            if self.visualize_mode == 'display':
                self._display_frame(annotated_frame, metadata['camera_name'])
            elif self.visualize_mode == 'save':
                # ALWAYS save frames when in save mode, regardless of detections
                self._save_frame_to_video(annotated_frame, metadata['camera_id'], metadata['camera_name'])
            
            # Send filtered results to downstream system
            self._send_results(metadata, final_detections, {
                'total_detections': len(detections),
                'class_filtered': len(class_filtered_detections),
                'final_detections': len(final_detections)
            })
    
    def _send_results(self, metadata, detections, filtering_stats):
        """Send results to downstream system"""
        result_summary = {
            'camera_id': metadata['camera_id'],
            'camera_name': metadata['camera_name'],
            'timestamp': metadata['timestamp'],
            'roi_enabled': metadata['roi_config']['enabled'],
            'roi_coordinates': metadata['roi_config']['coordinates'] if metadata['roi_config']['enabled'] else None,
            'classes_to_detect': metadata['classes_to_detect'],
            'filtering_stats': filtering_stats,
            'detection_count': len(detections),
            'detections': detections
        }
        
        # Here you can implement your result forwarding logic
        # For example: save to database, send to API, write to file, etc.
        pass