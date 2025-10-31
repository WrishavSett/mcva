#!/usr/bin/env python3
import signal
import sys
import time
import os
import argparse
import yaml
import torch
from producer import FrameProducer
import logging
import threading
from datetime import datetime

def setup_logging(config):
    """Configure logging to write to both file and console"""
    paths = config.get('paths', {})
    log_dir = paths.get('log_dir', './logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"counting_{timestamp}.log")
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    
    root_logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    return log_file

logger = logging.getLogger(__name__)

def select_consumer(config):
    """
    Automatically select the appropriate consumer based on configuration.
    Returns the Consumer class and mode information.
    """
    device = config.get('device', 'cpu')
    consumer_config = config.get('consumer', {})
    mode = consumer_config.get('mode', 'auto')
    num_cameras = len(config.get('cameras', {}))
    
    # Auto-detect best mode
    if mode == 'auto':
        if device == 'cuda' and torch.cuda.is_available():
            if num_cameras >= 3:
                mode = 'gpu'
            else:
                # For 1-2 cameras, CPU parallel might be competitive
                mode = 'cpu'
                logger.info("Auto-selected CPU mode (optimal for 1-2 cameras)")
        else:
            mode = 'cpu'
    
    # Import appropriate consumer
    if mode == 'gpu' and device == 'cuda':
        try:
            from consumer_gpu import GPUOptimizedConsumer
            logger.info("Using GPU-optimized consumer")
            return GPUOptimizedConsumer, 'gpu'
        except ImportError as e:
            logger.warning(f"Failed to import GPU consumer: {e}, falling back to standard consumer")
            mode = 'cpu'
    
    if mode == 'cpu' or device == 'cpu':
        from consumer_cpu import CountingConsumer
        logger.info("Using standard CPU consumer")
        return CountingConsumer, 'cpu'

class HealthAPI:
    """Optional Flask-based health monitoring API"""
    def __init__(self, app_instance, config):
        self.app_instance = app_instance
        self.config = config
        self.flask_app = None
        self.api_thread = None
        
        health_config = config.get('health_api', {})
        self.enabled = health_config.get('enabled', False)
        self.host = health_config.get('host', '0.0.0.0')
        self.port = health_config.get('port', 8080)
        
        if self.enabled:
            try:
                from flask import Flask, jsonify
                self.Flask = Flask
                self.jsonify = jsonify
                self._setup_api()
            except ImportError:
                logger.error("Flask not installed. Install with: pip install flask")
                self.enabled = False
    
    def _setup_api(self):
        """Setup Flask routes"""
        self.flask_app = self.Flask(__name__)
        
        @self.flask_app.route('/status')
        def status():
            try:
                status_data = {
                    'system': 'running' if self.app_instance.running else 'stopped',
                    'queue_size': self.app_instance.producer.queue_size(),
                    'camera_status': self.app_instance.producer.get_camera_status(),
                    'counting_status': self.app_instance.consumer.get_status()
                }
                return self.jsonify(status_data), 200
            except Exception as e:
                return self.jsonify({'error': str(e)}), 500
        
        @self.flask_app.route('/health')
        def health():
            return self.jsonify({'status': 'healthy'}), 200
        
        @self.flask_app.route('/cameras')
        def cameras():
            try:
                camera_info = {}
                for camera_id, config in self.config['cameras'].items():
                    camera_info[camera_id] = {
                        'name': config['name'],
                        'counting_line_enabled': config['counting_line']['enabled'],
                        'classes_to_count': config['classes_to_count']
                    }
                return self.jsonify(camera_info), 200
            except Exception as e:
                return self.jsonify({'error': str(e)}), 500
    
    def start(self):
        """Start the Flask API in a separate thread"""
        if not self.enabled or not self.flask_app:
            return
        
        def run_api():
            logger.info(f"Starting health API on http://{self.host}:{self.port}")
            self.flask_app.run(host=self.host, port=self.port, debug=False, use_reloader=False)
        
        self.api_thread = threading.Thread(target=run_api, daemon=True)
        self.api_thread.start()
        logger.info(f"Health API endpoints:")
        logger.info(f"  - Status: http://{self.host}:{self.port}/status")
        logger.info(f"  - Health: http://{self.host}:{self.port}/health")
        logger.info(f"  - Cameras: http://{self.host}:{self.port}/cameras")
    
    def stop(self):
        """Stop the Flask API"""
        if self.api_thread and self.api_thread.is_alive():
            logger.info("Health API stopped")

class MultiCameraCounting:
    def __init__(self, config_path="config_cpu.yaml", visualize_mode=None):
        self.config_path = config_path
        self.visualize_mode = visualize_mode
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create necessary directories
        paths = self.config.get('paths', {})
        for path_key, path_value in paths.items():
            os.makedirs(path_value, exist_ok=True)
        
        # Select appropriate consumer
        ConsumerClass, self.consumer_mode = select_consumer(self.config)
        
        # Initialize components
        self.producer = FrameProducer(config_path)
        self.consumer = ConsumerClass(config_path, visualize_mode=visualize_mode)
        self.consumer_thread = threading.Thread(target=self.consumer.start, args=(self.producer,))
        self.running = True
        
        # Health API
        self.health_api = HealthAPI(self, self.config)
    
    def start(self):
        """Start the counting system"""
        viz_info = f" with {self.visualize_mode} mode" if self.visualize_mode else ""
        logger.info(f"Starting Multi-Camera Object Counting System{viz_info}")
        logger.info(f"Consumer mode: {self.consumer_mode.upper()}")
        logger.info(f"Count logging interval: {self.consumer.log_interval} seconds")
        
        if hasattr(self.consumer, 'enable_parallel'):
            if self.consumer.enable_parallel:
                logger.info(f"Parallel processing: ENABLED ({self.consumer.max_workers} workers)")
            else:
                logger.info("Parallel processing: DISABLED")
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Start health API if enabled
        self.health_api.start()
        
        logger.info("Starting Producer.")
        self.producer.start()

        logger.info("Starting Consumer thread.")
        self.consumer_thread.start()

        logger.info("Pipeline running. Press Ctrl+C to stop.")
        
        # Status update thread
        last_status_time = time.time()
        status_interval = 60  # Print status every 60 seconds
        
        while self.running:
            time.sleep(0.5)
            
            # Print periodic status
            if time.time() - last_status_time >= status_interval:
                self._print_status()
                last_status_time = time.time()

        logger.info("Pipeline stopping...")
        
        # Stop health API
        self.health_api.stop()
        
        self.consumer.stop()
        self.consumer_thread.join()
        self.producer.stop()
        logger.info("Pipeline stopped cleanly.")
    
    def _signal_handler(self, signum, frame):
        if self.running:
            logger.info(f"Signal {signum} received. Stopping...")
            self.running = False
    
    def _print_status(self):
        """Print current system status"""
        queue_size = self.producer.queue_size()
        camera_status = self.producer.get_camera_status()
        
        logger.info("="*80)
        logger.info("SYSTEM STATUS")
        logger.info(f"Consumer mode: {self.consumer_mode.upper()}")
        logger.info(f"Frame queue size: {queue_size}")
        
        # Print camera connection status
        logger.info("Camera Status:")
        for camera_id, status in camera_status.items():
            status_str = "Connected" if status['connected'] else "Disconnected"
            thread_str = "Running" if status['thread_alive'] else "Stopped"
            logger.info(f"  {camera_id}: {status_str} | Thread: {thread_str}")
        
        # Print current counts
        logger.info("Counting Status:")
        for camera_id, counter in self.consumer.object_counters.items():
            camera_name = self.consumer.config['cameras'][camera_id]['name']
            time_since_log = time.time() - self.consumer.last_log_time[camera_id]
            next_log_in = max(0, self.consumer.log_interval - time_since_log)
            
            with self.consumer.counter_locks[camera_id]:
                if hasattr(counter, 'get'):
                    # GPU consumer uses a dictionary
                    in_count = counter.get('in_count', 0)
                    out_count = counter.get('out_count', 0)
                    counted_ids = len(counter.get('counted_ids', []))
                else:
                    # CPU consumer uses an ObjectCounter instance
                    in_count = counter.in_count
                    out_count = counter.out_count
                    counted_ids = len(counter.counted_ids)

                logger.info(f"  {camera_name} ({camera_id}):")
                logger.info(f"    Current IN: {in_count} | OUT: {out_count}")
                logger.info(f"    Unique objects tracked: {counted_ids}")
                logger.info(f"    Next log in: {next_log_in:.0f} seconds")
        
        logger.info("="*80)

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Camera Object Counting System with Adaptive GPU/CPU Selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Auto-select CPU/GPU mode
  python main.py --config config_gpu.yaml     # Use GPU-optimized config
  python main.py --visualize                  # Display in OpenCV windows
  python main.py --save                       # Save annotated videos
  
Consumer Selection:
  - Auto mode: Automatically selects best consumer based on hardware
  - GPU mode: Used when device='cuda' and 3+ cameras
  - CPU mode: Used for CPU-only or 1-2 cameras
  
Config files:
  - config_cpu.yaml: Standard CPU configuration (V2.1 baseline)
  - config_gpu.yaml: GPU-optimized configuration

Health API (if enabled in config):
  http://localhost:8080/status      # System status
  http://localhost:8080/health      # Health check
  http://localhost:8080/cameras     # Camera information
        """
    )
    
    parser.add_argument(
        '--config',
        default='config_cpu.yaml',
        help='Path to configuration file (default: config_cpu.yaml)'
    )
    
    viz_group = parser.add_mutually_exclusive_group()
    viz_group.add_argument(
        '--visualize',
        action='store_true',
        help='Display counting results in real-time OpenCV windows'
    )
    viz_group.add_argument(
        '--save',
        action='store_true',
        help='Save counting visualization as video files'
    )
    
    args = parser.parse_args()
    
    # Load config for logging setup
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    log_file = setup_logging(config)
    logger.info(f"Logging to file: {log_file}")
    logger.info(f"Logging to console: enabled")
    logger.info(f"Using config file: {args.config}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.info("CUDA not available, using CPU")
    
    visualize_mode = None
    if args.visualize:
        visualize_mode = 'display'
        logger.info("Visualization mode: Real-time display")
    elif args.save:
        visualize_mode = 'save'
        logger.info("Visualization mode: Save to video files")
    else:
        logger.info("Visualization mode: Disabled (logging only)")
    
    try:
        app = MultiCameraCounting(config_path=args.config, visualize_mode=visualize_mode)
        app.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()