#!/usr/bin/env python3
import signal
import sys
import time
import os
import argparse
from producer import FrameProducer
from consumer import BatchConsumer
import logging
import threading
from datetime import datetime

# Setup logging to both file and console
def setup_logging():
    """Configure logging to write to both file and console"""
    os.makedirs("logs", exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("logs", f"analytics_{timestamp}.log")
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler with UTF-8 encoding for Windows compatibility
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Force UTF-8 encoding for console output on Windows
    if sys.platform == 'win32':
        import codecs
        sys.stdout.reconfigure(encoding='utf-8')
    
    root_logger.addHandler(console_handler)
    
    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    return log_file

logger = logging.getLogger(__name__)

class MultiCameraAnalytics:
    def __init__(self, visualize_mode=None):
        self.producer = FrameProducer()
        self.consumer = BatchConsumer(visualize_mode=visualize_mode)
        self.consumer_thread = threading.Thread(target=self.consumer.start, args=(self.producer,))
        self.running = True
        self.visualize_mode = visualize_mode
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
    
    def start(self):
        """Start the analytics system"""
        viz_info = f" with {self.visualize_mode} mode" if self.visualize_mode else ""
        logger.info(f"Starting Multi-Camera Analytics System with ROI and Class Filtering{viz_info}")
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Starting Producer.")
        self.producer.start()

        logger.info("Starting Consumer thread.")
        self.consumer_thread.start()

        logger.info("Pipeline running. Press Ctrl+C to stop.")
        while self.running:
            time.sleep(0.5)

        logger.info("Pipeline stopping...")
        self.consumer.stop()
        self.consumer_thread.join()
        self.producer.stop()
        logger.info("Pipeline stopped cleanly.")
    
    def _signal_handler(self, signum, frame):
        if self.running:
            logger.info(f"Signal {signum} received. Stopping...")
            self.running = False
    
    def status(self):
        """Print system status"""
        queue_size = self.producer.queue_size()
        logger.info(f"Frame queue size: {queue_size}")

def main():
    # Setup logging first (before any other operations)
    log_file = setup_logging()
    logger.info(f"Logging to file: {log_file}")
    logger.info(f"Logging to console: enabled")
    
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Multi-Camera Video Analytics System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run without visualization
  python main.py --visualize        # Display detections in OpenCV windows
  python main.py --save             # Save annotated videos to disk
        """
    )
    
    viz_group = parser.add_mutually_exclusive_group()
    viz_group.add_argument(
        '--visualize',
        action='store_true',
        help='Display detection results in real-time OpenCV windows'
    )
    viz_group.add_argument(
        '--save',
        action='store_true',
        help='Save detection results as video files'
    )
    
    args = parser.parse_args()
    
    # Determine visualization mode
    visualize_mode = None
    if args.visualize:
        visualize_mode = 'display'
        logger.info("Visualization mode: Real-time display")
    elif args.save:
        visualize_mode = 'save'
        logger.info("Visualization mode: Save to video files")
    else:
        logger.info("Visualization mode: Disabled (logging only)")
    
    # Initialize and start the system
    app = MultiCameraAnalytics(visualize_mode=visualize_mode)
    app.start()

if __name__ == "__main__":
    main()