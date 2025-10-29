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
import json
from collections import defaultdict
from ObjectCount import ObjectCounter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CountingConsumer:
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
        
        # Counting settings
        self.counting_config = self.config.get('counting', {})
        self.log_interval = self.counting_config.get('log_interval', 900)
        self.reset_after_log = self.counting_config.get('reset_after_log', False)
        
        # Visualization settings
        self.visualize_mode = visualize_mode
        self.viz_config = self.config.get('visualization', {})
        self.video_config = self.config.get('video_save', {})
        self.video_writers = {}
        
        # Create output directory for saved videos
        if self.visualize_mode == 'save':
            output_dir = self.video_config.get('output_dir', './output_videos')
            os.makedirs(output_dir, exist_ok=True)
            self.output_dir = output_dir
            logger.info(f"Video output directory: {self.output_dir}")
        
        # Create class name to ID mapping
        self.class_name_to_id = {name: idx for idx, name in self.model.names.items()}
        self.class_id_to_name = self.model.names
        
        # Initialize object counters for each camera
        self.object_counters = {}
        
        # Logging control (must be initialized BEFORE _initialize_counters)
        self.last_log_time = {}
        self.count_logs_dir = "count_logs"
        os.makedirs(self.count_logs_dir, exist_ok=True)
        
        # Now initialize counters (uses self.last_log_time)
        self._initialize_counters()
        
        self.pending_frames = []
        self.last_batch_time = time.time()
        self.running = threading.Event()
        self.running.clear()
        
        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Model input size: {self.model_input_size}")
        logger.info(f"Counting log interval: {self.log_interval} seconds")
        logger.info(f"Visualization mode: {self.visualize_mode if self.visualize_mode else 'disabled'}")
        logger.info(f"Available classes: {list(self.class_name_to_id.keys())}")
    
    def _initialize_counters(self):
        """Initialize ObjectCounter for each camera"""
        for camera_id, camera_config in self.config['cameras'].items():
            counting_line_config = camera_config['counting_line']
            
            if not counting_line_config['enabled']:
                logger.warning(f"Counting line disabled for {camera_config['name']}")
                continue
            
            # Create ObjectCounter instance
            counter = ObjectCounter(
                allowed_classes=camera_config['classes_to_count'],
                show=False,  # We'll handle visualization ourselves
                region=counting_line_config['coordinates'],
                model=self.model,
                show_in=self.counting_config.get('show_in', True),
                show_out=self.counting_config.get('show_out', True)
            )
            
            self.object_counters[camera_id] = counter
            self.last_log_time[camera_id] = time.time()
            
            logger.info(f"Initialized counter for {camera_config['name']} with line: {counting_line_config['coordinates']}")
    
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
            
            # Check if we should log counts for any camera
            self._check_and_log_counts()
            
            # Handle OpenCV window events if displaying
            if self.visualize_mode == 'display':
                cv2.waitKey(1)
            
            time.sleep(0.001)
    
    def stop(self):
        """Stop the consumer"""
        self.running.clear()
        
        # Process any remaining frames
        if self.pending_frames:
            self._process_batch()
        
        # Log final counts for all cameras
        logger.info("===== FINAL COUNT SUMMARY =====")
        for camera_id in self.object_counters.keys():
            self._log_counts(camera_id, force=True)
        
        # Close all video writers
        if self.visualize_mode == 'save':
            for camera_id, writer in self.video_writers.items():
                writer.release()
                logger.info(f"Closed video writer for camera: {camera_id}")
        
        # Close all OpenCV windows
        if self.visualize_mode == 'display':
            cv2.destroyAllWindows()
    
    def _collect_frames(self):
        """Collect frames from producer"""
        frame_data = self.producer.get_frame()
        if frame_data:
            self.pending_frames.append(frame_data)
    
    def _should_process_batch(self):
        """Determine if batch should be processed"""
        if not self.pending_frames:
            return False
        
        batch_full = len(self.pending_frames) >= self.batch_size
        timeout_reached = (time.time() - self.last_batch_time) >= self.batch_timeout
        
        return batch_full or timeout_reached
    
    def _check_and_log_counts(self):
        """Check if it's time to log counts for any camera"""
        current_time = time.time()
        
        for camera_id in self.object_counters.keys():
            time_since_last_log = current_time - self.last_log_time[camera_id]
            
            if time_since_last_log >= self.log_interval:
                self._log_counts(camera_id)
    
    def _log_counts(self, camera_id, force=False):
        """Log counts for a specific camera"""
        if camera_id not in self.object_counters:
            return
        
        counter = self.object_counters[camera_id]
        camera_name = self.config['cameras'][camera_id]['name']
        current_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare count data
        count_data = {
            'timestamp': timestamp,
            'camera_id': camera_id,
            'camera_name': camera_name,
            'elapsed_seconds': current_time - self.last_log_time[camera_id],
            'counts': {
                'in_count': counter.in_count,
                'out_count': counter.out_count,
                'total': counter.in_count + counter.out_count
            },
            'classwise_counts': dict(counter.classwise_counts),
            'counted_ids': len(counter.counted_ids),
            'log_type': 'FINAL' if force else 'PERIODIC'
        }
        
        # Log to console
        log_prefix = "FINAL COUNT" if force else "PERIODIC COUNT"
        logger.info(f"{'='*80}")
        logger.info(f"{log_prefix} - {camera_name} ({camera_id})")
        logger.info(f"Timestamp: {timestamp}")
        logger.info(f"Elapsed Time: {count_data['elapsed_seconds']:.1f} seconds")
        logger.info(f"Total IN: {counter.in_count} | Total OUT: {counter.out_count}")
        logger.info(f"Net Count: {counter.in_count - counter.out_count}")
        
        if counter.classwise_counts:
            logger.info("Class-wise Counts:")
            for class_name, counts in counter.classwise_counts.items():
                logger.info(f"  {class_name}: IN={counts['IN']}, OUT={counts['OUT']}")
        
        logger.info(f"Unique Objects Counted: {len(counter.counted_ids)}")
        logger.info(f"{'='*80}")
        
        # Save to JSON file
        log_filename = f"{camera_id}_counts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        log_filepath = os.path.join(self.count_logs_dir, log_filename)
        
        try:
            with open(log_filepath, 'w') as f:
                json.dump(count_data, f, indent=2)
            logger.info(f"Count data saved to: {log_filepath}")
        except Exception as e:
            logger.error(f"Failed to save count data: {e}")
        
        # Update last log time
        self.last_log_time[camera_id] = current_time
        
        # Reset counts if configured
        if self.reset_after_log and not force:
            counter.reset_count()
            logger.info(f"Counts reset for {camera_name}")
    
    def _draw_counting_line(self, frame, counting_line_config):
        """Draw counting line on frame"""
        if not counting_line_config['enabled']:
            return frame
        
        if not self.viz_config.get('show_counting_line', True):
            return frame
        
        coordinates = counting_line_config['coordinates']
        line_color = tuple(self.viz_config.get('line_color', [255, 0, 0]))
        line_thickness = self.viz_config.get('line_thickness', 2)
        
        if len(coordinates) == 2:
            # Draw line
            cv2.line(frame, tuple(coordinates[0]), tuple(coordinates[1]), line_color, line_thickness)
        elif len(coordinates) > 2:
            # Draw polygon
            points = np.array(coordinates, dtype=np.int32)
            cv2.polylines(frame, [points], True, line_color, line_thickness)
        
        return frame
    
    def _visualize_tracking(self, frame, tracks, camera_name, counting_line_config, counts):
        """Draw tracking and counting visualization on frame"""
        vis_frame = frame.copy()
        
        # Draw counting line
        vis_frame = self._draw_counting_line(vis_frame, counting_line_config)
        
        # Get visualization colors
        box_color = tuple(self.viz_config.get('box_color', [0, 255, 0]))
        text_color = tuple(self.viz_config.get('text_color', [0, 255, 0]))
        box_thickness = self.viz_config.get('box_thickness', 2)
        text_thickness = self.viz_config.get('text_thickness', 2)
        font_scale = self.viz_config.get('font_scale', 0.6)
        
        # Draw tracks
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['track_id']
            class_name = track['class']
            conf = track['confidence']
            
            label = f"ID:{track_id} {class_name} {conf:.2f}"
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), box_color, box_thickness)
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
            cv2.rectangle(vis_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), box_color, -1)
            
            # Draw label text
            cv2.putText(vis_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness)
        
        # Add camera info and counts
        info_y = 30
        cv2.putText(vis_frame, f"{camera_name}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        info_y += 35
        cv2.putText(vis_frame, f"IN: {counts['in']} | OUT: {counts['out']} | Total: {counts['total']}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        info_y += 30
        cv2.putText(vis_frame, f"Active Tracks: {len(tracks)}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_frame
    
    def _initialize_video_writer(self, camera_id, camera_name, frame_shape):
        """Initialize video writer for a camera"""
        if camera_id not in self.video_writers:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{camera_id}_{camera_name.replace(' ', '_')}_{timestamp}.mp4"
            filepath = os.path.join(self.output_dir, filename)
            
            height, width = frame_shape[:2]
            fps = self.video_config.get('fps', 30)
            codec = self.video_config.get('codec', 'mp4v')
            fourcc = cv2.VideoWriter_fourcc(*codec)
            
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
        if not self._initialize_video_writer(camera_id, camera_name, frame.shape):
            return
        
        self.video_writers[camera_id].write(frame)
    
    def _process_batch(self):
        """Process batch of frames through tracking and counting"""
        if not self.pending_frames:
            return
        
        batch_start_time = time.time()
        
        # Group frames by camera for sequential processing
        camera_frames = defaultdict(list)
        for frame_data in self.pending_frames:
            camera_frames[frame_data['camera_id']].append(frame_data)
        
        # Process each camera's frames sequentially through its counter
        for camera_id, frames in camera_frames.items():
            if camera_id not in self.object_counters:
                continue
            
            counter = self.object_counters[camera_id]
            
            for frame_data in frames:
                # Run tracking on frame using the counter
                im0 = frame_data['frame']
                original_frame = frame_data['original_frame']
                
                # Process frame through counter (this updates counts internally)
                try:
                    _ = counter.count(im0)
                    
                    # Get current tracking info for visualization
                    tracks = []
                    if hasattr(counter, 'boxes') and counter.boxes is not None:
                        for box, track_id, cls, conf in zip(counter.boxes, counter.track_ids, 
                                                            counter.clss, counter.confs):
                            class_name = counter.names[cls]
                            
                            # Filter by allowed classes
                            if counter.allowed_classes and class_name not in counter.allowed_classes:
                                continue
                            
                            x1, y1, x2, y2 = map(int, box)
                            tracks.append({
                                'bbox': [x1, y1, x2, y2],
                                'track_id': int(track_id),
                                'class': class_name,
                                'class_id': int(cls),
                                'confidence': float(conf)
                            })
                    
                    # Get current counts
                    current_counts = {
                        'in': counter.in_count,
                        'out': counter.out_count,
                        'total': counter.in_count + counter.out_count
                    }
                    
                    # Visualize if needed
                    if self.visualize_mode:
                        display_frame = cv2.resize(original_frame, self.model_input_size)
                        annotated_frame = self._visualize_tracking(
                            display_frame,
                            tracks,
                            frame_data['camera_name'],
                            frame_data['counting_line'],
                            current_counts
                        )
                        
                        if self.visualize_mode == 'display':
                            cv2.imshow(frame_data['camera_name'], annotated_frame)
                        elif self.visualize_mode == 'save':
                            self._save_frame_to_video(annotated_frame, camera_id, frame_data['camera_name'])
                
                except Exception as e:
                    logger.error(f"Error processing frame for {camera_id}: {e}")
        
        processing_time = time.time() - batch_start_time
        logger.debug(f"Processed batch of {len(self.pending_frames)} frames in {processing_time:.3f}s")
        
        # Clear batch
        self.pending_frames.clear()
        self.last_batch_time = time.time()