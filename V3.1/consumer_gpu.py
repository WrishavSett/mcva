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
from shapely.geometry import LineString, Polygon, Point

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUOptimizedConsumer:
    """
    GPU-optimized consumer with true batched inference.
    Maintains all V2.1 functionality while maximizing GPU utilization.
    """

    def __init__(self, config, visualize_mode=None):
        self.config = config

        # Path configuration
        self.paths = self.config.get('paths', {})
        self.model_dir = self.paths.get('model_dir', './models')
        self.log_dir = self.paths.get('log_dir', './logs')
        self.count_log_dir = self.paths.get('count_log_dir', './count_logs')
        self.output_video_dir = self.paths.get('output_video_dir', './output_videos')
        
        # Create directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.count_log_dir, exist_ok=True)
        
        # Load YOLO model
        model_path = os.path.join(self.model_dir, self.config['model_name'])
        self.model = YOLO(model_path)
        self.device = self.config['device']
        self.batch_size = self.config['batch_size']
        self.batch_timeout = self.config['batch_timeout']
        self.confidence_threshold = self.config['confidence_threshold']
        self.model_input_size = tuple(self.config['model_input_size'])
        
        # Verify GPU availability
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
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
            os.makedirs(self.output_video_dir, exist_ok=True)
            logger.info(f"Video output directory: {self.output_video_dir}")
        
        # Create class name to ID mapping
        self.class_name_to_id = {name: idx for idx, name in self.model.names.items()}
        self.class_id_to_name = self.model.names
        
        # Initialize object counters for each camera
        self.object_counters = {}
        self.counter_locks = {}  # Thread locks for parallel processing
        
        # Logging control
        self.last_log_time = {}

        # Initialize counters
        self._initialize_counters()
        
        self.pending_frames = []
        self.last_batch_time = time.time()
        self.running = threading.Event()
        self.running.clear()
        
        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Model input size: {self.model_input_size}")
        logger.info(f"GPU-optimized batching: ENABLED (batch_size={self.batch_size})")
        logger.info(f"Counting log interval: {self.log_interval} seconds")
        logger.info(f"Visualization mode: {self.visualize_mode if self.visualize_mode else 'disabled'}")
        logger.info(f"Available classes: {list(self.class_name_to_id.keys())}")

    def _initialize_counters(self):
        """Initialize counter state for each camera"""
        for camera_id, camera_config in self.config['cameras'].items():
            counting_line_config = camera_config['counting_line']
            
            if not counting_line_config['enabled']:
                logger.warning(f"Counting line disabled for {camera_config['name']}")
                continue
            
            # Counter state structure
            counter_info = {
                'camera_id': camera_id,
                'camera_name': camera_config['name'],
                'allowed_classes': set(camera_config['classes_to_count']) if camera_config['classes_to_count'] else None,
                'counting_line': counting_line_config['coordinates'],
                'in_count': 0,
                'out_count': 0,
                'counted_ids': set(),
                'classwise_counts': defaultdict(lambda: {"IN": 0, "OUT": 0}),
                'track_history': defaultdict(list),
                'show_in': self.counting_config.get('show_in', True),
                'show_out': self.counting_config.get('show_out', True)
            }
            
            self.object_counters[camera_id] = counter_info
            self.counter_locks[camera_id] = threading.Lock()
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
                self._process_batch_gpu_optimized()
            
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
            self._process_batch_gpu_optimized()
        
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
    
    def _process_batch_gpu_optimized(self):
        """
        GPU-optimized batch processing with TRUE batched inference.
        Key optimization: Single YOLO call for all frames.
        """
        if not self.pending_frames:
            return
        
        batch_start_time = time.time()
        
        # Prepare batch for inference
        batch_frames = []
        frame_metadata = []
        
        for frame_data in self.pending_frames:
            batch_frames.append(frame_data['frame'])
            frame_metadata.append(frame_data)
        
        try:
            # =================================================================
            # KEY GPU OPTIMIZATION: Single batched inference call
            # =================================================================
            results = self.model.track(
                batch_frames,
                conf=self.confidence_threshold,
                device=self.device,
                persist=True,  # Enable tracking persistence
                verbose=False,
                imgsz=self.model_input_size[0]
            )
            
            # Process results for each frame
            for result, metadata in zip(results, frame_metadata):
                camera_id = metadata['camera_id']
                
                if camera_id not in self.object_counters:
                    continue
                
                # Thread-safe processing
                with self.counter_locks[camera_id]:
                    tracks = self._process_detections(result, metadata, camera_id)
                
                # Visualization (outside lock for performance)
                if self.visualize_mode and tracks:
                    self._handle_visualization(metadata, tracks, camera_id)
            
            processing_time = time.time() - batch_start_time
            logger.info(f"Processed batch of {len(batch_frames)} frames in {processing_time:.3f}s (GPU-optimized)")
            
        except Exception as e:
            logger.error(f"Error in GPU batch processing: {e}", exc_info=True)
        
        finally:
            self.pending_frames.clear()
            self.last_batch_time = time.time()
    
    def _process_detections(self, result, metadata, camera_id):
        """
        Process YOLO detections and update counts.
        Returns list of tracks for visualization.
        """
        counter = self.object_counters[camera_id]
        
        if result.boxes is None or len(result.boxes) == 0:
            return []
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # Get track IDs
        if hasattr(result.boxes, 'id') and result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
        else:
            # Fallback if tracking fails
            track_ids = np.arange(len(boxes))
        
        tracks = []
        
        for box, conf, cls_id, track_id in zip(boxes, confidences, class_ids, track_ids):
            class_name = self.class_id_to_name[cls_id]
            
            # Filter by allowed classes
            if counter['allowed_classes'] and class_name not in counter['allowed_classes']:
                continue
            
            x1, y1, x2, y2 = box
            current_centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # Store track history
            counter['track_history'][track_id].append(current_centroid)
            
            # Limit history size to prevent memory growth
            if len(counter['track_history'][track_id]) > 100:
                counter['track_history'][track_id].pop(0)
            
            # Get previous position
            prev_position = None
            if len(counter['track_history'][track_id]) > 1:
                prev_position = counter['track_history'][track_id][-2]
            
            # Count objects crossing the line
            self._count_crossing(
                counter, 
                current_centroid, 
                prev_position, 
                track_id, 
                class_name
            )
            
            # Prepare track for visualization
            tracks.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'track_id': int(track_id),
                'class': class_name,
                'class_id': int(cls_id),
                'confidence': float(conf)
            })
        
        return tracks
    
    def _count_crossing(self, counter, current_centroid, prev_position, track_id, class_name):
        """
        Check if object crossed counting line and update counts.
        Implements same logic as V2.1 ObjectCounter.
        """
        if prev_position is None or track_id in counter['counted_ids']:
            return
        
        # Initialize classwise count for this class
        if class_name not in counter['classwise_counts']:
            counter['classwise_counts'][class_name] = {"IN": 0, "OUT": 0}
        
        counting_line = counter['counting_line']
        
        if len(counting_line) == 2:  # Linear counting line
            line = LineString(counting_line)
            trajectory = LineString([prev_position, current_centroid])
            
            if line.intersects(trajectory):
                # Determine orientation (vertical or horizontal)
                dx = abs(counting_line[0][0] - counting_line[1][0])
                dy = abs(counting_line[0][1] - counting_line[1][1])
                
                if dx < dy:  # Vertical line
                    if current_centroid[0] > prev_position[0]:  # Moving right
                        counter['in_count'] += 1
                        counter['classwise_counts'][class_name]["IN"] += 1
                    else:  # Moving left
                        counter['out_count'] += 1
                        counter['classwise_counts'][class_name]["OUT"] += 1
                else:  # Horizontal line
                    if current_centroid[1] > prev_position[1]:  # Moving down
                        counter['in_count'] += 1
                        counter['classwise_counts'][class_name]["IN"] += 1
                    else:  # Moving up
                        counter['out_count'] += 1
                        counter['classwise_counts'][class_name]["OUT"] += 1
                
                counter['counted_ids'].add(track_id)
        
        elif len(counting_line) > 2:  # Polygonal region
            polygon = Polygon(counting_line)
            point = Point(current_centroid)
            
            if polygon.contains(point):
                # Calculate region dimensions
                xs = [p[0] for p in counting_line]
                ys = [p[1] for p in counting_line]
                region_width = max(xs) - min(xs)
                region_height = max(ys) - min(ys)
                
                # Determine direction based on region shape
                if (region_width < region_height and current_centroid[0] > prev_position[0]) or \
                   (region_width >= region_height and current_centroid[1] > prev_position[1]):
                    counter['in_count'] += 1
                    counter['classwise_counts'][class_name]["IN"] += 1
                else:
                    counter['out_count'] += 1
                    counter['classwise_counts'][class_name]["OUT"] += 1
                
                counter['counted_ids'].add(track_id)
    
    def _handle_visualization(self, metadata, tracks, camera_id):
        """Handle visualization with all V2.1 features"""
        counter = self.object_counters[camera_id]
        original_frame = metadata['original_frame']
        
        # Resize to model input size for consistent visualization
        display_frame = cv2.resize(original_frame, self.model_input_size)
        
        # Draw counting line
        annotated_frame = self._draw_counting_line(display_frame, metadata['counting_line'])
        
        # Draw tracks
        annotated_frame = self._draw_tracks(
            annotated_frame, 
            tracks, 
            metadata['camera_name'], 
            counter
        )
        
        if self.visualize_mode == 'display':
            cv2.imshow(metadata['camera_name'], annotated_frame)
        elif self.visualize_mode == 'save':
            self._save_frame_to_video(annotated_frame, camera_id, metadata['camera_name'])
    
    def _draw_counting_line(self, frame, counting_line_config):
        """Draw counting line on frame (V2.1 compatible)"""
        if not counting_line_config['enabled']:
            return frame
        
        if not self.viz_config.get('show_counting_line', True):
            return frame
        
        coordinates = counting_line_config['coordinates']
        line_color = tuple(self.viz_config.get('line_color', [255, 0, 0]))
        line_thickness = self.viz_config.get('line_thickness', 2)
        
        if len(coordinates) == 2:
            cv2.line(frame, tuple(coordinates[0]), tuple(coordinates[1]), line_color, line_thickness)
        elif len(coordinates) > 2:
            points = np.array(coordinates, dtype=np.int32)
            cv2.polylines(frame, [points], True, line_color, line_thickness)
        
        return frame
    
    def _draw_tracks(self, frame, tracks, camera_name, counter):
        """Draw tracking boxes and info (V2.1 compatible)"""
        box_color = tuple(self.viz_config.get('box_color', [0, 255, 0]))
        text_color = tuple(self.viz_config.get('text_color', [0, 255, 0]))
        box_thickness = self.viz_config.get('box_thickness', 2)
        text_thickness = self.viz_config.get('text_thickness', 2)
        font_scale = self.viz_config.get('font_scale', 0.6)
        
        # Draw bounding boxes
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['track_id']
            class_name = track['class']
            conf = track['confidence']
            
            label = f"ID:{track_id} {class_name} {conf:.2f}"
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), box_color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness)
        
        # Add camera info
        info_y = 30
        cv2.putText(frame, f"{camera_name}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        info_y += 35
        cv2.putText(frame, f"IN: {counter['in_count']} | OUT: {counter['out_count']} | Total: {counter['in_count'] + counter['out_count']}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        info_y += 30
        cv2.putText(frame, f"Active Tracks: {len(tracks)}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _save_frame_to_video(self, frame, camera_id, camera_name):
        """Save frame to video file (V2.1 compatible)"""
        if camera_id not in self.video_writers:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{camera_id}_{camera_name.replace(' ', '_')}_{timestamp}.mp4"
            filepath = os.path.join(self.output_video_dir, filename)
            
            height, width = frame.shape[:2]
            fps = self.video_config.get('fps', 30)
            codec = self.video_config.get('codec', 'mp4v')
            fourcc = cv2.VideoWriter_fourcc(*codec)
            
            writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
            
            if not writer.isOpened():
                logger.error(f"Failed to create video writer for {camera_name}: {filepath}")
                return
            
            self.video_writers[camera_id] = writer
            logger.info(f"Created video writer for {camera_name}: {filepath}")
        
        self.video_writers[camera_id].write(frame)
    
    def _check_and_log_counts(self):
        """Check if it's time to log counts (V2.1 compatible)"""
        current_time = time.time()
        
        for camera_id in self.object_counters.keys():
            time_since_last_log = current_time - self.last_log_time[camera_id]
            
            if time_since_last_log >= self.log_interval:
                self._log_counts(camera_id)
    
    def _log_counts(self, camera_id, force=False):
        """Log counts for a camera (V2.1 compatible)"""
        if camera_id not in self.object_counters:
            return
        
        with self.counter_locks[camera_id]:
            counter = self.object_counters[camera_id]
            current_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            count_data = {
                'timestamp': timestamp,
                'camera_id': camera_id,
                'camera_name': counter['camera_name'],
                'elapsed_seconds': current_time - self.last_log_time[camera_id],
                'counts': {
                    'in_count': counter['in_count'],
                    'out_count': counter['out_count'],
                    'total': counter['in_count'] + counter['out_count']
                },
                'classwise_counts': dict(counter['classwise_counts']),
                'counted_ids': len(counter['counted_ids']),
                'log_type': 'FINAL' if force else 'PERIODIC'
            }
        
        # Log to console
        log_prefix = "FINAL COUNT" if force else "PERIODIC COUNT"
        logger.info(f"{'='*80}")
        logger.info(f"{log_prefix} - {counter['camera_name']} ({camera_id})")
        logger.info(f"Timestamp: {timestamp}")
        logger.info(f"Elapsed Time: {count_data['elapsed_seconds']:.1f} seconds")
        logger.info(f"Total IN: {count_data['counts']['in_count']} | Total OUT: {count_data['counts']['out_count']}")
        logger.info(f"Net Count: {count_data['counts']['in_count'] - count_data['counts']['out_count']}")
        
        if count_data['classwise_counts']:
            logger.info("Class-wise Counts:")
            for class_name, counts in count_data['classwise_counts'].items():
                logger.info(f"  {class_name}: IN={counts['IN']}, OUT={counts['OUT']}")
        
        logger.info(f"Unique Objects Counted: {count_data['counted_ids']}")
        logger.info(f"{'='*80}")
        
        # Save to JSON
        log_filename = f"{camera_id}_counts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        log_filepath = os.path.join(self.count_log_dir, log_filename)
        
        try:
            with open(log_filepath, 'w', encoding='utf-8') as f:
                json.dump(count_data, f, indent=2)
            logger.info(f"Count data saved to: {log_filepath}")
        except (IOError, PermissionError) as e:
            logger.error(f"Failed to save count data to {log_filepath}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error saving count data: {e}", exc_info=True)
        
        # Update last log time
        self.last_log_time[camera_id] = current_time
        
        # Reset counts if configured
        if self.reset_after_log and not force:
            counter['in_count'] = 0
            counter['out_count'] = 0
            counter['counted_ids'] = set()
            counter['classwise_counts'] = defaultdict(lambda: {"IN": 0, "OUT": 0})
            logger.info(f"Counts reset for {counter['camera_name']}")
    
    def get_status(self):
        """Get current system status (V2.1 compatible)"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'cameras': {}
        }
        
        for camera_id, counter in self.object_counters.items():
            with self.counter_locks[camera_id]:
                time_since_log = time.time() - self.last_log_time[camera_id]
                
                status['cameras'][camera_id] = {
                    'name': counter['camera_name'],
                    'in_count': counter['in_count'],
                    'out_count': counter['out_count'],
                    'total_count': counter['in_count'] + counter['out_count'],
                    'unique_tracks': len(counter['counted_ids']),
                    'time_since_last_log': time_since_log,
                    'next_log_in': max(0, self.log_interval - time_since_log)
                }
        
        return status