import cv2
import os
import time
import logging
from datetime import datetime, timedelta

# --- Configuration ---
RTSP_URL = "rtsp://kumar:Kumar%23123@116.73.21.116:554/Streaming/Channels/101"
RECORD_DURATION_SECONDS = 3600  # 1 hour
OUTPUT_VIDEO_TARGET_HEIGHT = 480 # Target height for the output video
OUTPUT_VIDEO_FPS = 4 # Target frame rate
RECONNECT_DELAY_SECONDS = 10
MAX_RECONNECT_ATTEMPTS = 5

# --- Logging Setup ---
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, 'recorder.log')

logging.basicConfig(level=logging.INFO,
                    format=LOG_FORMAT,
                    handlers=[
                        logging.FileHandler(LOG_FILE),
                        logging.StreamHandler()
                    ])

def record_rtsp_stream(rtsp_url, output_path, target_height, fps, duration):
    """
    Records a video stream from an RTSP URL to a file for a specified duration.

    Includes logic for automatic reconnection if the stream temporarily disconnects.

    Args:
        rtsp_url (str): The URL of the RTSP stream to record.
        output_path (str): The absolute path where the recorded video will be saved.
        target_height (int): The desired height of the output video in pixels.
                             The width will be calculated to maintain the original aspect ratio.
        fps (int): The desired frame rate (frames per second) of the output video.
        duration (int): The total duration to record in seconds.
    """
    logging.info(f"Starting recording for {duration} seconds to {output_path}")
    
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        logging.error(f"Initial connection to RTSP stream failed at {rtsp_url}")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate aspect ratio, with a fallback if original dimensions are zero
    aspect_ratio = (original_height / original_width) if original_width > 0 and original_height > 0 else (9/16) 
    
    # Calculate width based on target_height and aspect_ratio
    width = int(target_height / aspect_ratio)
    height = target_height # The actual height used

    # Use H.264 codec ('avc1')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
    if not out.isOpened():
        logging.error(f"Could not open video writer for path {output_path}. "
                      f"Ensure OpenCV is built with H.264 support (e.g., x264).")
        cap.release()
        return

    logging.info(f"Recording at {width}x{height} @ {fps}fps using H.264 codec.")

    start_time = time.time()
    reconnect_attempts = 0
    
    try:
        while (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if ret:
                if reconnect_attempts > 0:
                    logging.info("Reconnection successful. Resuming recording.")
                    reconnect_attempts = 0  # Reset on success

                resized_frame = cv2.resize(frame, (width, height))
                out.write(resized_frame)
            else:
                logging.warning("Stream disconnected. Attempting to reconnect...")
                reconnect_attempts += 1
                if reconnect_attempts > MAX_RECONNECT_ATTEMPTS:
                    logging.error("Max reconnect attempts reached. Ending recording for this hour.")
                    break

                cap.release()
                time.sleep(RECONNECT_DELAY_SECONDS)
                cap = cv2.VideoCapture(rtsp_url)
                if not cap.isOpened():
                    logging.warning(f"Reconnect attempt {reconnect_attempts} failed.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during recording: {e}", exc_info=True)
    finally:
        logging.info("Closing video stream and writer.")
        cap.release()
        out.release()

def main():
    """
    Main function to schedule and manage hourly RTSP stream recordings.

    This function runs indefinitely, waiting for the beginning of each hour
    to start a new recording session. It handles potential errors in the
    recording process and attempts to continue scheduling subsequent recordings.
    The recording duration is configurable via RECORD_DURATION_SECONDS.
    """
    while True:
        try:
            now = datetime.now()
            next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            wait_seconds = (next_hour - now).total_seconds()
            
            logging.info(f"Waiting for {wait_seconds:.2f} seconds until {next_hour} to start recording.")
            time.sleep(wait_seconds)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}.mp4"
            output_dir = os.path.join(SCRIPT_DIR, "vids")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = os.path.join(output_dir, filename)

            record_rtsp_stream(
                rtsp_url=RTSP_URL,
                output_path=output_file,
                target_height=OUTPUT_VIDEO_TARGET_HEIGHT, # Pass the target height
                fps=OUTPUT_VIDEO_FPS,
                duration=RECORD_DURATION_SECONDS
            )
            logging.info(f"Finished hourly recording. Video saved to {output_file}")

        except KeyboardInterrupt:
            logging.info("Script interrupted by user. Exiting.")
            break
        except Exception as e:
            logging.critical(f"A critical error occurred in the main loop: {e}", exc_info=True)
            logging.info("Restarting main loop after 60 seconds...")
            time.sleep(60)

if __name__ == "__main__":
    main()