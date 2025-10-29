import cv2
import time
import threading
from queue import Queue, Empty
import yaml
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameProducer:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.frame_queue = Queue(maxsize=self.config['frame_queue_size'])
        self.cameras = {}
        self.capture_threads = {}
        self.running = threading.Event()
        self.running.clear()
        self.model_input_size = tuple(self.config['model_input_size'])
        
    def start(self):
        """Start all camera capture threads"""
        self.running.set()
        
        for camera_id, camera_config in self.config['cameras'].items():
            thread = threading.Thread(
                target=self._capture_frames,
                args=(camera_id, camera_config)
            )
            self.capture_threads[camera_id] = thread
            thread.start()
            
            classes_info = "all classes" if not camera_config['classes_to_count'] else str(camera_config['classes_to_count'])
            logger.info(f"Started capture for {camera_config['name']} - Counting Line: {camera_config['counting_line']['enabled']}, Classes: {classes_info}")
    
    def stop(self):
        """Stop all capture threads"""
        self.running.clear()

        # Wait for all threads to finish
        for thread in self.capture_threads.values():
            thread.join()
        
        # Close all captures
        for cap in self.cameras.values():
            if cap.isOpened():
                cap.release()
        
        logger.info("All cameras stopped")
    
    def _preprocess_frame(self, frame):
        """Preprocess frame: resize to model input size"""
        resized_frame = cv2.resize(frame, self.model_input_size)
        return resized_frame
    
    def _capture_frames(self, camera_id, camera_config):
        """Capture frames from a single camera"""
        cap = cv2.VideoCapture(camera_config['rtsp_url'])
        self.cameras[camera_id] = cap
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}: {camera_config['name']}")
            return
        
        # Set FPS
        cap.set(cv2.CAP_PROP_FPS, self.config['target_fps'])
        
        frame_interval = 1.0 / self.config['target_fps']
        last_time = time.time()
        
        logger.info(f"Camera {camera_id} - Counting Line: {camera_config['counting_line']['enabled']}, "
                   f"Classes: {camera_config['classes_to_count'] if camera_config['classes_to_count'] else 'all'}")
        
        try:
            while self.running.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read from {camera_id}")
                    time.sleep(0.1)
                    continue
                
                current_time = time.time()
                
                # Frame rate control
                if current_time - last_time >= frame_interval:
                    # Preprocess frame (resize only, no ROI masking for counting)
                    processed_frame = self._preprocess_frame(frame)
                    
                    frame_data = {
                        'camera_id': camera_id,
                        'camera_name': camera_config['name'],
                        'frame': processed_frame,
                        'original_frame': frame,
                        'counting_line': camera_config['counting_line'],
                        'classes_to_count': camera_config['classes_to_count'],
                        'timestamp': current_time
                    }
                    
                    try:
                        # Non-blocking put, drop frame if queue is full
                        self.frame_queue.put_nowait(frame_data)
                    except:
                        pass  # Drop frame if queue is full
                    
                    last_time = current_time

                time.sleep(0.001)

        finally:
            cap.release()
    
    def get_frame(self):
        """Get next frame from queue"""
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None
    
    def queue_size(self):
        """Get current queue size"""
        return self.frame_queue.qsize()