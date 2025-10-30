import cv2
import time
import threading
from queue import Queue, Empty, Full
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
        
        # Reconnection settings
        self.reconnect_config = self.config.get('producer', {})
        self.max_retries = self.reconnect_config.get('reconnect_max_retries', 10)
        self.base_delay = self.reconnect_config.get('reconnect_base_delay', 1.0)
        self.max_delay = self.reconnect_config.get('reconnect_max_delay', 60.0)
        self.backoff_factor = self.reconnect_config.get('reconnect_backoff_factor', 2.0)
        
    def start(self):
        """Start all camera capture threads"""
        self.running.set()
        
        for camera_id, camera_config in self.config['cameras'].items():
            thread = threading.Thread(
                target=self._capture_frames,
                args=(camera_id, camera_config),
                daemon=True
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
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        # Close all captures
        for cap in self.cameras.values():
            if cap and cap.isOpened():
                cap.release()
        
        logger.info("All cameras stopped")
    
    def _preprocess_frame(self, frame):
        """Preprocess frame: resize to model input size"""
        resized_frame = cv2.resize(frame, self.model_input_size)
        return resized_frame
    
    def _connect_camera(self, camera_id, rtsp_url, retry_count=0):
        """Attempt to connect to camera with exponential backoff"""
        if retry_count > 0:
            # Calculate backoff delay
            delay = min(
                self.base_delay * (self.backoff_factor ** (retry_count - 1)),
                self.max_delay
            )
            logger.warning(f"{camera_id}: Retry {retry_count}/{self.max_retries} after {delay:.1f}s delay")
            time.sleep(delay)
        
        try:
            cap = cv2.VideoCapture(rtsp_url)
            
            if cap.isOpened():
                # Set FPS
                cap.set(cv2.CAP_PROP_FPS, self.config['target_fps'])
                logger.info(f"{camera_id}: Successfully connected")
                return cap
            else:
                cap.release()
                return None
                
        except Exception as e:
            logger.error(f"{camera_id}: Connection error: {e}")
            return None
    
    def _capture_frames(self, camera_id, camera_config):
        """Capture frames from a single camera with reconnection logic"""
        retry_count = 0
        cap = None
        
        frame_interval = 1.0 / self.config['target_fps']
        last_time = time.time()
        consecutive_failures = 0
        
        logger.info(f"Camera {camera_id} - Counting Line: {camera_config['counting_line']['enabled']}, "
                   f"Classes: {camera_config['classes_to_count'] if camera_config['classes_to_count'] else 'all'}")
        
        try:
            while self.running.is_set():
                # Try to connect/reconnect if needed
                if cap is None or not cap.isOpened():
                    if retry_count >= self.max_retries:
                        logger.error(f"{camera_id}: Max retries ({self.max_retries}) reached. Giving up.")
                        break
                    
                    cap = self._connect_camera(camera_id, camera_config['rtsp_url'], retry_count)
                    
                    if cap is None:
                        retry_count += 1
                        continue
                    else:
                        # Successful connection, reset retry counter
                        retry_count = 0
                        consecutive_failures = 0
                        self.cameras[camera_id] = cap
                
                # Try to read frame
                ret, frame = cap.read()
                
                if not ret:
                    consecutive_failures += 1
                    logger.warning(f"{camera_id}: Failed to read frame (consecutive failures: {consecutive_failures})")
                    
                    # If too many consecutive failures, trigger reconnection
                    if consecutive_failures >= 10:
                        logger.error(f"{camera_id}: Too many consecutive failures. Attempting reconnection.")
                        if cap:
                            cap.release()
                        cap = None
                        retry_count += 1
                        consecutive_failures = 0
                    
                    time.sleep(0.1)
                    continue
                
                # Successfully read frame
                consecutive_failures = 0
                current_time = time.time()
                
                # Frame rate control
                if current_time - last_time >= frame_interval:
                    # Preprocess frame
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
                    except Full:
                        logger.debug(f"{camera_id}: Dropped frame - queue full")
                    
                    last_time = current_time

                time.sleep(0.001)

        except Exception as e:
            logger.error(f"{camera_id}: Unexpected error in capture thread: {e}", exc_info=True)
        finally:
            if cap and cap.isOpened():
                cap.release()
            logger.info(f"{camera_id}: Capture thread stopped")
    
    def get_frame(self):
        """Get next frame from queue"""
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None
    
    def queue_size(self):
        """Get current queue size"""
        return self.frame_queue.qsize()
    
    def get_camera_status(self):
        """Get status of all cameras"""
        status = {}
        for camera_id, cap in self.cameras.items():
            status[camera_id] = {
                'connected': cap.isOpened() if cap else False,
                'thread_alive': self.capture_threads[camera_id].is_alive() if camera_id in self.capture_threads else False
            }
        return status