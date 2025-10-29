#!/usr/bin/env python3
import signal
import sys
import time
import os
import argparse
from producer import FrameProducer
from consumer import CountingConsumer
import logging
import threading
from datetime import datetime

def setup_logging():
    """Configure logging to write to both file and console"""
    os.makedirs("logs", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("logs", f"counting_{timestamp}.log")
    
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

class MultiCameraCounting:
    def __init__(self, visualize_mode=None):
        self.producer = FrameProducer()
        self.consumer = CountingConsumer(visualize_mode=visualize_mode)
        self.consumer_thread = threading.Thread(target=self.consumer.start, args=(self.producer,))
        self.running = True
        self.visualize_mode = visualize_mode
        
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("count_logs", exist_ok=True)
    
    def start(self):
        """Start the counting system"""
        viz_info = f" with {self.visualize_mode} mode" if self.visualize_mode else ""
        logger.info(f"Starting Multi-Camera Object Counting System{viz_info}")
        logger.info(f"Count logging interval: {self.consumer.log_interval} seconds")
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
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
        
        logger.info("="*80)
        logger.info("SYSTEM STATUS")
        logger.info(f"Frame queue size: {queue_size}")
        
        # Print current counts for each camera
        for camera_id, counter in self.consumer.object_counters.items():
            camera_name = self.consumer.config['cameras'][camera_id]['name']
            time_since_log = time.time() - self.consumer.last_log_time[camera_id]
            next_log_in = max(0, self.consumer.log_interval - time_since_log)
            
            logger.info(f"{camera_name} ({camera_id}):")
            logger.info(f"  Current IN: {counter.in_count} | OUT: {counter.out_count}")
            logger.info(f"  Unique objects tracked: {len(counter.counted_ids)}")
            logger.info(f"  Next log in: {next_log_in:.0f} seconds")
        
        logger.info("="*80)

def main():
    log_file = setup_logging()
    logger.info(f"Logging to file: {log_file}")
    logger.info(f"Logging to console: enabled")
    
    parser = argparse.ArgumentParser(
        description='Multi-Camera Object Counting System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run without visualization
  python main.py --visualize        # Display counting in OpenCV windows
  python main.py --save             # Save annotated videos to disk
  
The system will log counts every X seconds (configured in config.yaml).
Count logs are saved to: ./count_logs/
        """
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
    
    visualize_mode = None
    if args.visualize:
        visualize_mode = 'display'
        logger.info("Visualization mode: Real-time display")
    elif args.save:
        visualize_mode = 'save'
        logger.info("Visualization mode: Save to video files")
    else:
        logger.info("Visualization mode: Disabled (logging only)")
    
    app = MultiCameraCounting(visualize_mode=visualize_mode)
    app.start()

if __name__ == "__main__":
    main()