from flask import Flask, request, jsonify, send_from_directory, send_file, make_response
import json
import os
import torch
import cv2
import numpy as np
from flask_cors import CORS
import uuid
import logging
from logging.handlers import RotatingFileHandler
import itertools
import random
import shutil
from datetime import datetime, timedelta
import re
import signal
import sys
import traceback
from werkzeug.serving import WSGIRequestHandler
import pickle

app = Flask(__name__)
CORS(app, resources={
    r"/*": {  # Allow CORS for all routes
        "origins": [
            "http://0.0.0.0:5500",
            "http://localhost:5500",
            "http://127.0.0.1:5500",
            "http://localhost:3000",
            "http://localhost:8000",
            "http://localhost:5501",
            "https://aliang8.github.io",  # Add GitHub Pages domain
            "https://snoopy1.usc.edu:8443"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin"],
        "expose_headers": ["Content-Type", "Content-Length", "Accept-Ranges", "Content-Range"],
        "supports_credentials": True
    }
})

# Configure logging more robustly
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure file logging
file_handler = RotatingFileHandler('logs/app.log', maxBytes=10000000, backupCount=5)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)

# Configure console logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s'
))
console_handler.setLevel(logging.INFO)

# Set up root logger
app.logger.addHandler(file_handler)
app.logger.addHandler(console_handler)
app.logger.setLevel(logging.INFO)

# Log uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    app.logger.critical("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

# Handle SIGTERM gracefully
def handle_sigterm(signum, frame):
    app.logger.info('Received SIGTERM signal. Performing cleanup...')
    cleanup_old_videos()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)

@app.errorhandler(Exception)
def handle_error(error):
    app.logger.error(f"Unhandled error: {error}", exc_info=True)
    return jsonify({
        'error': str(error),
        'type': error.__class__.__name__
    }), 500

class CustomRequestHandler(WSGIRequestHandler):
    def log_request(self, code='-', size='-'):
        if str(code).startswith('2') or str(code).startswith('3'):
            app.logger.info(f"{self.address_string()} - - [{self.log_date_time_string()}] {self.requestline} {code} {size}")
        else:
            app.logger.error(f"{self.address_string()} - - [{self.log_date_time_string()}] {self.requestline} {code} {size}")

    def log_error(self, format, *args):
        app.logger.error(f"{self.address_string()} - - [{self.log_date_time_string()}] {format%args}")

# Get the absolute path of the current directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = "/scr/aliang80/robot_pref_site"

# Folder to store generated videos - use absolute path
VIDEO_DIR = os.path.join(ROOT_DIR, 'temp_videos')  # Point to /scr/aliang80/robot_pref_site/temp_videos
os.makedirs(VIDEO_DIR, exist_ok=True)

app.logger.info(f"Video directory: {VIDEO_DIR}")
app.logger.info(f"Base directory: {BASE_DIR}")
app.logger.info(f"Root directory: {ROOT_DIR}")

# Set maximum file upload size to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Video cleanup settings
VIDEO_MAX_AGE = timedelta(minutes=30)  # Delete videos older than 30 minutes

def cleanup_old_videos():
    """Delete videos older than VIDEO_MAX_AGE"""
    try:
        now = datetime.now()
        for filename in os.listdir(VIDEO_DIR):
            filepath = os.path.join(VIDEO_DIR, filename)
            file_modified = datetime.fromtimestamp(os.path.getmtime(filepath))
            if now - file_modified > VIDEO_MAX_AGE:
                os.remove(filepath)
                app.logger.info(f"Cleaned up old video: {filename}")
    except Exception as e:
        app.logger.error(f"Error cleaning up videos: {str(e)}")

@app.before_request
def log_request_info():
    app.logger.info('Headers: %s', dict(request.headers))
    app.logger.info('Body: %s', request.get_data())
    app.logger.info('Args: %s', dict(request.args))

@app.after_request
def after_request(response):
    app.logger.info('Response: %s', response.get_data())
    origin = request.headers.get('Origin')
    allowed_origins = [
        "http://0.0.0.0:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:5501",
        "https://aliang8.github.io",
        "https://snoopy1.usc.edu:8443"
    ]
    
    if origin in allowed_origins:
        response.headers.add('Access-Control-Allow-Origin', origin)
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,Origin')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Expose-Headers', 'Content-Type,Content-Length,Accept-Ranges,Content-Range')
        
    # Handle preflight requests
    if request.method == 'OPTIONS':
        response.headers['Access-Control-Max-Age'] = '3600'
        response.status_code = 204
        return response
        
    return response

@app.route('/api/get-trajectory-pair', methods=['GET'])
def get_trajectory_pair():
    # Clean up old videos first
    cleanup_old_videos()
    
    app.logger.info("Loading trajectory pair")
    
    try:
        # Get the requested pair index and dataset from query params
        pair_index = request.args.get('pair_index')
        dataset = request.args.get('dataset', 'assembly-v2')  # Default to assembly-v2
        
        if pair_index is None:
            return jsonify({'error': 'Missing pair_index parameter'}), 400
            
        try:
            pair_index = int(pair_index)
            app.logger.info(f"Using pair_index: {pair_index}")
        except (ValueError, AttributeError) as e:
            app.logger.error(f"Invalid pair_index format: {pair_index}")
            return jsonify({'error': 'Invalid pair_index format'}), 400
        
        # Construct data path based on dataset
        dataset_dir = os.path.join('data', dataset)
        app.logger.info(f"Looking for data in directory: {dataset_dir}")
        
        if not os.path.exists(dataset_dir):
            app.logger.error(f"Dataset directory not found: {dataset_dir}")
            return jsonify({'error': f'Dataset directory not found: {dataset}'}), 404
            
        # Find the trajectory data file
        data_files = [f for f in os.listdir(dataset_dir) if f.endswith('.pt')]
        app.logger.info(f"Found .pt files: {data_files}")
        
        if not data_files:
            app.logger.error(f"No .pt file found in dataset directory: {dataset_dir}")
            return jsonify({'error': f'No .pt file found in dataset {dataset}'}), 404
            
        data_path = os.path.join(dataset_dir, data_files[0])
        app.logger.info(f"Loading trajectory data from {data_path}")
        
        if not os.path.exists(data_path):
            app.logger.error(f"Data file not found: {data_path}")
            return jsonify({'error': f'Data file not found: {data_path}'}), 404

        trajectory_data = torch.load(data_path)
        app.logger.info(f"Loaded trajectory data keys: {list(trajectory_data.keys())}")
        app.logger.info(f"Image data shape: {trajectory_data['image'].shape}")
        
        # Convert tensors to numpy
        trajectory_data = {
            k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
            for k, v in trajectory_data.items()
        }
        app.logger.info("Converted tensors to numpy arrays")
        
        # Load segment indices from the dataset directory
        segment_indices_path = os.path.join(dataset_dir, 'segment_start_end_indices.npy')
        segment_pairs_path = os.path.join(dataset_dir, 'segment_pairs.npy')
        
        app.logger.info(f"Loading segment data from: {segment_indices_path} and {segment_pairs_path}")
        
        if not os.path.exists(segment_indices_path):
            app.logger.error(f"Segment indices file not found: {segment_indices_path}")
            return jsonify({'error': 'Segment indices file not found'}), 404
            
        if not os.path.exists(segment_pairs_path):
            app.logger.error(f"Segment pairs file not found: {segment_pairs_path}")
            return jsonify({'error': 'Segment pairs file not found'}), 404

        segment_indices = np.load(segment_indices_path)
        segment_indices = segment_indices.reshape(-1, 2)
        segment_pairs = np.load(segment_pairs_path)
        total_pairs = len(segment_pairs)
        
        app.logger.info(f"Total pairs: {total_pairs}")
        
        # Validate pair_index and get segments
        pair_idx = pair_index % total_pairs  # Wrap around if pair_index exceeds total pairs
        segment_a, segment_b = segment_pairs[pair_idx][0], segment_pairs[pair_idx][1]
        
        app.logger.info(f"Pair {pair_idx}: segment_a={segment_a}, segment_b={segment_b}")
        app.logger.info(f"segment_indices bounds: min={segment_indices.min()}, max={segment_indices.max()}")
        app.logger.info(f"Valid segment range: 0 to {len(segment_indices)-1}")
        
        # Validate segment indices are within bounds
        if segment_a >= len(segment_indices) or segment_b >= len(segment_indices):
            return jsonify({'error': f'Invalid segment indices: {segment_a}, {segment_b}. Max valid index: {len(segment_indices)-1}'}), 400

        app.logger.info(f"Using segments {segment_a} and {segment_b} from pair_index {pair_index} (wrapped to {pair_idx})")
        
        # Get trajectory frames
        try:
            # Each row in segment_indices contains [start, end] for a segment
            trajectory_a_start = segment_indices[segment_a][0]
            trajectory_a_end = segment_indices[segment_a][1]
            trajectory_b_start = segment_indices[segment_b][0]
            trajectory_b_end = segment_indices[segment_b][1]
            
            app.logger.info(f"Extracting frames - A: {trajectory_a_start}:{trajectory_a_end}, B: {trajectory_b_start}:{trajectory_b_end}")
            
            # Get rewards for each segment
            reward_a = float(trajectory_data['reward'][trajectory_a_start:trajectory_a_end].mean())
            reward_b = float(trajectory_data['reward'][trajectory_b_start:trajectory_b_end].mean())
            app.logger.info(f"Segment rewards - A: {reward_a}, B: {reward_b}")

            # Validate indices are within bounds of trajectory data
            total_frames = len(trajectory_data['image'])
            if (trajectory_a_end > total_frames or trajectory_b_end > total_frames or
                trajectory_a_start < 0 or trajectory_b_start < 0):
                return jsonify({'error': f'Invalid frame indices. Total frames: {total_frames}, Requested A: {trajectory_a_start}:{trajectory_a_end}, B: {trajectory_b_start}:{trajectory_b_end}'}), 400
            
            trajectory_a_frames = trajectory_data['image'][trajectory_a_start:trajectory_a_end]
            trajectory_b_frames = trajectory_data['image'][trajectory_b_start:trajectory_b_end]

            # Log frame information
            app.logger.info(f"Trajectory A frames (indices {trajectory_a_start}:{trajectory_a_end}) shape: {trajectory_a_frames.shape}")
            app.logger.info(f"Trajectory B frames (indices {trajectory_b_start}:{trajectory_b_end}) shape: {trajectory_b_frames.shape}")

            # Ensure frames are numpy arrays
            trajectory_a_frames = np.array(trajectory_a_frames)
            trajectory_b_frames = np.array(trajectory_b_frames)
            
            # Ensure frames are in the correct format (uint8, 0-255 range)
            if trajectory_a_frames.dtype != np.uint8:
                if trajectory_a_frames.max() <= 1.0:
                    trajectory_a_frames = (trajectory_a_frames * 255).astype(np.uint8)
                else:
                    trajectory_a_frames = trajectory_a_frames.astype(np.uint8)
            
            if trajectory_b_frames.dtype != np.uint8:
                if trajectory_b_frames.max() <= 1.0:
                    trajectory_b_frames = (trajectory_b_frames * 255).astype(np.uint8)
                else:
                    trajectory_b_frames = trajectory_b_frames.astype(np.uint8)
            
            app.logger.info(f"After conversion - A shape: {trajectory_a_frames.shape}, dtype: {trajectory_a_frames.dtype}")
            app.logger.info(f"After conversion - B shape: {trajectory_b_frames.shape}, dtype: {trajectory_b_frames.dtype}")
            
        except Exception as e:
            app.logger.error(f"Error extracting frames: {str(e)}", exc_info=True)
            raise

        # Generate filenames based on segment indices and dataset
        video_a_path = os.path.join(VIDEO_DIR, f"{dataset}_segment_{segment_a}.mp4")
        video_b_path = os.path.join(VIDEO_DIR, f"{dataset}_segment_{segment_b}.mp4")

        # Generate videos only if they don't exist
        if not os.path.exists(video_a_path):
            app.logger.info(f"Generating video A for segment {segment_a}...")
            generate_video(trajectory_a_frames, video_a_path)
        else:
            app.logger.info(f"Reusing existing video for segment {segment_a}")

        if not os.path.exists(video_b_path):
            app.logger.info(f"Generating video B for segment {segment_b}...")
            generate_video(trajectory_b_frames, video_b_path)
        else:
            app.logger.info(f"Reusing existing video for segment {segment_b}")
        
        # Return full URLs
        base_url = request.host_url.rstrip('/')
        response = {
            'segments': [int(segment_a), int(segment_b)],
            'trajectory_a_url': f"{base_url}/videos/{os.path.basename(video_a_path)}",
            'trajectory_b_url': f"{base_url}/videos/{os.path.basename(video_b_path)}",
            'reward_a': reward_a,
            'reward_b': reward_b
        }
        app.logger.info(f"Successfully generated trajectory pair: {response}")
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error getting trajectory pair: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-available-datasets', methods=['GET'])
def get_available_datasets():
    try:
        datasets = []
        data_dir = 'data'
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                # Check if directory contains any .pt files
                if any(f.endswith('.pt') for f in os.listdir(item_path)):
                    datasets.append(item)
        return jsonify({'datasets': sorted(datasets)})
    except Exception as e:
        app.logger.error(f"Error getting available datasets: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-dataset-info', methods=['GET'])
def get_dataset_info():
    try:
        dataset = request.args.get('dataset', 'assembly-v2')
        dataset_dir = os.path.join('data', dataset)
        
        if not os.path.isdir(dataset_dir):
            return jsonify({'error': f'Dataset directory not found: {dataset}'}), 404
            
        # Load segment pairs
        segment_pairs = np.load(os.path.join(dataset_dir, 'segment_pairs.npy'))
        total_pairs = len(segment_pairs)
        
        # Count labeled pairs
        dataset_pref_dir = os.path.join(dataset_dir, 'preferences')
        os.makedirs(dataset_pref_dir, exist_ok=True)
        
        labeled_pairs = set()
        for file in os.listdir(dataset_pref_dir):
            if file.endswith('.json'):
                with open(os.path.join(dataset_pref_dir, file), 'r') as f:
                    prefs = json.load(f)
                    for pref in prefs.get('preferences', []):
                        labeled_pairs.add(pref.get('pair_index'))
        
        return jsonify({
            'total_pairs': total_pairs,
            'labeled_pairs': len(labeled_pairs),
            'unlabeled_pairs': total_pairs - len(labeled_pairs)
        })
    except Exception as e:
        app.logger.error(f"Error getting dataset info: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-preferences', methods=['POST'])
def save_preferences():
    try:
        data = request.json
        dataset = data.get('dataset', 'assembly-v2')
        app.logger.info(f"Saving preferences for session {data.get('session_id')} in dataset {dataset}")
        
        # Save preferences in dataset-specific directory
        dataset_pref_dir = os.path.join('data', dataset, 'preferences')
        os.makedirs(dataset_pref_dir, exist_ok=True)
        
        filename = f"preferences_{data['session_id']}.json"
        filepath = os.path.join(dataset_pref_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        app.logger.info(f"Successfully saved preferences to {filepath}")
        return jsonify({"success": True})
    except Exception as e:
        app.logger.error(f"Error saving preferences: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/videos/<path:filename>')
def serve_video(filename):
    """Serve video files with proper headers and CORS"""
    # Ensure the file exists
    video_path = os.path.join(VIDEO_DIR, filename)
    app.logger.info(f"Attempting to serve video: {video_path}")
    
    if not os.path.exists(video_path):
        app.logger.error(f"Video file not found: {video_path}")
        return jsonify({'error': 'Video file not found'}), 404

    # Get file size and check if it's valid
    file_size = os.path.getsize(video_path)
    app.logger.info(f"Video file size: {file_size} bytes")
    
    if file_size == 0:
        app.logger.error(f"Video file is empty: {video_path}")
        return jsonify({'error': 'Video file is empty'}), 500

    # Handle range requests
    range_header = request.headers.get('Range', None)
    app.logger.info(f"Range header: {range_header}")
    
    if range_header:
        byte1, byte2 = 0, None
        match = re.search('(\d+)-(\d*)', range_header)
        if match:
            groups = match.groups()
            if groups[0]: byte1 = int(groups[0])
            if groups[1]: byte2 = int(groups[1])
        
        if byte2 is None:
            byte2 = file_size - 1
        
        length = byte2 - byte1 + 1
        app.logger.info(f"Serving bytes {byte1}-{byte2}/{file_size}")

        try:
            with open(video_path, 'rb') as f:
                f.seek(byte1)
                data = f.read(length)
                
            response = make_response(bytes(data))
            response.headers.set('Content-Range', f'bytes {byte1}-{byte2}/{file_size}')
            response.headers.set('Accept-Ranges', 'bytes')
            response.headers.set('Content-Length', str(length))
            response.headers.set('Content-Type', 'video/mp4')
            response.status_code = 206
        except Exception as e:
            app.logger.error(f"Error reading file: {e}", exc_info=True)
            return jsonify({'error': 'Error reading video file'}), 500
    else:
        # Normal request - serve entire file
        try:
            with open(video_path, 'rb') as f:
                data = f.read()
                
            response = make_response(bytes(data))
            response.headers.set('Content-Type', 'video/mp4')
            response.headers.set('Content-Length', str(file_size))
            response.headers.set('Accept-Ranges', 'bytes')
        except Exception as e:
            app.logger.error(f"Error reading file: {e}", exc_info=True)
            return jsonify({'error': 'Error reading video file'}), 500
    
    # Add CORS headers
    response.headers.set('Access-Control-Allow-Origin', '*')
    response.headers.set('Access-Control-Allow-Methods', 'GET, OPTIONS')
    response.headers.set('Access-Control-Allow-Headers', 'Range, Content-Type, Authorization, Accept')
    response.headers.set('Access-Control-Expose-Headers', 'Content-Type, Content-Length, Accept-Ranges, Content-Range')
    
    # Add cache control headers
    response.headers.set('Cache-Control', 'no-cache, no-store, must-revalidate')
    response.headers.set('Pragma', 'no-cache')
    response.headers.set('Expires', '0')
    
    app.logger.info(f"Successfully serving video with headers: {dict(response.headers)}")
    return response


def generate_video(frames, output_path):
    """Generate video from trajectory frames"""
    if len(frames.shape) != 4:  # Expected shape: (T, H, W, C)
        raise ValueError(f"Invalid frame shape: {frames.shape}, expected 4 dimensions (T, H, W, C)")
    
    app.logger.info(f"Generating video with shape {frames.shape} to {output_path}")
    
    # Get video dimensions from first frame
    height, width = frames.shape[1:3]
    app.logger.info(f"Video dimensions: {width}x{height}")
    
    # Ensure frames are in the correct format
    if frames.dtype != np.uint8:
        app.logger.info(f"Converting frames from {frames.dtype} to uint8")
        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)
    
    # Ensure frames are in RGB format
    if frames.shape[-1] != 3:
        raise ValueError(f"Invalid number of channels: {frames.shape[-1]}, expected 3 (RGB)")
    
    # Initialize video writer with MJPG codec (more widely supported)
    temp_output = output_path.replace('.mp4', '_temp.avi')
    app.logger.info(f"Using temporary file: {temp_output}")
    
    # Use MJPG codec for initial video creation
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(
        temp_output,
        fourcc,
        10,  # Reduced FPS from 30 to 10 for slower playback
        (width, height),
        isColor=True
    )
    
    if not out.isOpened():
        app.logger.error("Failed to initialize VideoWriter")
        raise ValueError(f"Failed to initialize video writer with MJPG codec. Shape: {frames.shape}")
    
    app.logger.info("VideoWriter initialized successfully")
    
    # Write frames
    for i, frame in enumerate(frames):
        try:
            # Convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Verify frame format
            if frame_bgr.shape != (height, width, 3):
                raise ValueError(f"Invalid frame shape at index {i}: {frame_bgr.shape}")
            if frame_bgr.dtype != np.uint8:
                raise ValueError(f"Invalid frame dtype at index {i}: {frame_bgr.dtype}")
            
            out.write(frame_bgr)
            
            if i % 10 == 0:  # Log every 10th frame
                app.logger.debug(f"Wrote frame {i}/{len(frames)}")
        except Exception as e:
            app.logger.error(f"Error writing frame {i}: {str(e)}")
            raise
    
    out.release()
    app.logger.info(f"Temporary video written to {temp_output}")
    
    # Verify temp file exists and has size
    if not os.path.exists(temp_output):
        raise ValueError(f"Temporary file {temp_output} was not created")
    
    temp_size = os.path.getsize(temp_output)
    app.logger.info(f"Temporary file size: {temp_size} bytes")
    
    if temp_size == 0:
        raise ValueError(f"Temporary file {temp_output} is empty")
    
    # Use ffmpeg to convert to web-compatible MP4
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-i', temp_output,  # Input file
        '-vcodec', 'libx264',  # Force H.264 codec
        '-profile:v', 'baseline',  # Use baseline profile for maximum compatibility
        '-level', '3.0',
        '-preset', 'medium',  # Better quality encoding
        '-movflags', '+faststart',  # Enable fast start for web playback
        '-pix_fmt', 'yuv420p',  # Pixel format for maximum compatibility
        '-an',  # No audio
        '-crf', '23',  # Constant rate factor (quality)
        '-r', '10',  # Set output FPS to 10
        '-f', 'mp4',  # Force MP4 format
        output_path  # Output file
    ]
    
    app.logger.info(f"Running ffmpeg command: {' '.join(ffmpeg_cmd)}")
    
    import subprocess
    result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
    app.logger.info(f"FFmpeg stdout: {result.stdout}")
    
    if os.path.exists(output_path):
        final_size = os.path.getsize(output_path)
        app.logger.info(f"Final MP4 size: {final_size} bytes")
        
        # Verify the output video
        probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
                    '-show_entries', 'stream=codec_name,width,height', 
                    '-of', 'json', output_path]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        app.logger.info(f"FFprobe output: {probe_result.stdout}")
    else:
        app.logger.error(f"Final MP4 file {output_path} was not created")
    
    os.remove(temp_output)  # Clean up temporary file
    app.logger.info(f"Successfully generated video: {output_path}")
    return output_path
   

@app.route('/api/get-similar-segments', methods=['GET'])
def get_similar_segments():
    try:
        # Get parameters
        dataset = request.args.get('dataset', 'assembly-v2')
        segment_index = request.args.get('segment_index')
        k = int(request.args.get('k', '5'))  # Number of similar/dissimilar segments to return
        
        if segment_index is None:
            return jsonify({'error': 'Missing segment_index parameter'}), 400
            
        try:
            segment_index = int(segment_index)
        except ValueError:
            return jsonify({'error': 'Invalid segment_index format'}), 400
            
        # Load dataset
        dataset_dir = os.path.join('data', dataset)
        data_files = [f for f in os.listdir(dataset_dir) if f.endswith('.pt')]
        
        if not data_files:
            return jsonify({'error': f'No .pt file found in dataset {dataset}'}), 404
            
        data_path = os.path.join(dataset_dir, data_files[0])
        trajectory_data = torch.load(data_path)
        
        # Load segment indices
        segment_indices = np.load(os.path.join(dataset_dir, 'segment_start_end_indices.npy'))
        segment_indices = segment_indices.reshape(-1, 2)
        
        if segment_index >= len(segment_indices):
            return jsonify({'error': 'Segment index out of bounds'}), 400

        # Load DTW distance matrix
        dtw_matrix_path = os.path.join(dataset_dir, 'dtw_matrix_32.pkl')
        if not os.path.exists(dtw_matrix_path):
            return jsonify({'error': 'DTW distance matrix not found'}), 404
            
        with open(dtw_matrix_path, 'rb') as f:
            dtw_distances, _  = pickle.load(f)
        
        # Get distances for the target segment
        target_distances = dtw_distances[segment_index]
        print(target_distances.shape)
        
        # Get indices sorted by distance (excluding the target segment itself)
        all_indices = np.argsort(target_distances)
        # Remove the target segment (which would have distance 0 to itself)
        all_indices = all_indices[all_indices != segment_index]
        
        # Get k most similar (lowest DTW) and k most dissimilar (highest DTW)
        similar_indices = all_indices[:k]  # First k are most similar
        dissimilar_indices = all_indices[-k:][::-1]  # Last k are most dissimilar (reverse to get highest DTW first)

        print("Similar indices:", similar_indices)
        print("Dissimilar indices:", dissimilar_indices)
        
        # Generate videos for all segments
        base_url = request.host_url.rstrip('/')
        results = {
            'target': None,
            'similar': [],
            'dissimilar': []
        }
        
        # First add the target segment
        target_start, target_end = segment_indices[segment_index]
        target_frames = trajectory_data['image'][target_start:target_end]
        target_video_path = os.path.join(VIDEO_DIR, f"{dataset}_segment_{segment_index}.mp4")
        
        if not os.path.exists(target_video_path):
            generate_video(target_frames.cpu().numpy(), target_video_path)
            
        target_reward = float(trajectory_data['reward'][target_start:target_end].mean())
        results['target'] = {
            'segment_index': int(segment_index),
            'video_url': f"{base_url}/videos/{os.path.basename(target_video_path)}",
            'similarity': 1.0,  # Self-similarity is 1
            'reward': target_reward,
            'dtw_distance': 0.0  # Distance to self is 0
        }
        
        # Calculate scale for similarity scores
        scale = np.mean(target_distances[target_distances > 0])
        
        # Add similar segments
        for sim_idx in similar_indices:
            start, end = segment_indices[sim_idx]
            video_path = os.path.join(VIDEO_DIR, f"{dataset}_segment_{sim_idx}.mp4")
            
            if not os.path.exists(video_path):
                frames = trajectory_data['image'][start:end]
                generate_video(frames.cpu().numpy(), video_path)
                
            reward = float(trajectory_data['reward'][start:end].mean())
            dtw_dist = float(target_distances[sim_idx])
            similarity = np.exp(-dtw_dist/scale)
            
            results['similar'].append({
                'segment_index': int(sim_idx),
                'video_url': f"{base_url}/videos/{os.path.basename(video_path)}",
                'similarity': similarity,
                'reward': reward,
                'dtw_distance': dtw_dist
            })
            
        # Add dissimilar segments
        for dis_idx in dissimilar_indices:
            start, end = segment_indices[dis_idx]
            video_path = os.path.join(VIDEO_DIR, f"{dataset}_segment_{dis_idx}.mp4")
            
            if not os.path.exists(video_path):
                frames = trajectory_data['image'][start:end]
                generate_video(frames.cpu().numpy(), video_path)
                
            reward = float(trajectory_data['reward'][start:end].mean())
            dtw_dist = float(target_distances[dis_idx])
            similarity = np.exp(-dtw_dist/scale)
            
            results['dissimilar'].append({
                'segment_index': int(dis_idx),
                'video_url': f"{base_url}/videos/{os.path.basename(video_path)}",
                'similarity': similarity,
                'reward': reward,
                'dtw_distance': dtw_dist
            })
        
        return jsonify(results)
        
    except Exception as e:
        app.logger.error(f"Error finding similar segments: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-next-pair', methods=['GET'])
def get_next_pair():
    try:
        dataset = request.args.get('dataset', 'assembly-v2')
        acquisition = request.args.get('acquisition', 'random')  # random, entropy, or disagreement
        dataset_dir = os.path.join('data', dataset)
        
        # Load segment pairs to get total number of pairs
        segment_pairs = np.load(os.path.join(dataset_dir, 'segment_pairs.npy'))
        total_pairs = len(segment_pairs)
        
        # Get labeled pairs
        dataset_pref_dir = os.path.join(dataset_dir, 'preferences')
        os.makedirs(dataset_pref_dir, exist_ok=True)
        
        labeled_pairs = set()
        for file in os.listdir(dataset_pref_dir):
            if file.endswith('.json'):
                with open(os.path.join(dataset_pref_dir, file), 'r') as f:
                    prefs = json.load(f)
                    for pref in prefs.get('preferences', []):
                        labeled_pairs.add(pref.get('pair_index'))
        
        # Get all unlabeled pair indices
        all_indices = set(range(total_pairs))
        unlabeled_indices = list(all_indices - labeled_pairs)
        
        if not unlabeled_indices:
            return jsonify({'message': 'All pairs have been labeled', 'pair_index': 0})
        
        # Select next pair based on acquisition method
        next_index = 0
        if acquisition == 'random':
            next_index = random.choice(unlabeled_indices)
        elif acquisition == 'entropy':
            # Load entropy scores if they exist
            entropy_file = os.path.join(dataset_dir, 'entropy_scores.pkl')
            if os.path.exists(entropy_file):
                with open(entropy_file, 'rb') as f:
                    entropy_scores = pickle.load(f)
                # Filter for unlabeled pairs and get the one with highest entropy
                unlabeled_scores = [(i, entropy_scores[i]) for i in unlabeled_indices]
                next_index = max(unlabeled_scores, key=lambda x: x[1])[0]
            else:
                # Fall back to random if no entropy scores
                next_index = random.choice(unlabeled_indices)
        elif acquisition == 'disagreement':
            # Load disagreement scores if they exist
            disagreement_file = os.path.join(dataset_dir, 'disagreement_scores.pkl')
            if os.path.exists(disagreement_file):
                with open(disagreement_file, 'rb') as f:
                    disagreement_scores = pickle.load(f)
                # Filter for unlabeled pairs and get the one with highest disagreement
                unlabeled_scores = [(i, disagreement_scores[i]) for i in unlabeled_indices]
                next_index = max(unlabeled_scores, key=lambda x: x[1])[0]
            else:
                # Fall back to random if no disagreement scores
                next_index = random.choice(unlabeled_indices)
        else:
            next_index = random.choice(unlabeled_indices)
        
        # Get the segments for this pair
        segments = segment_pairs[next_index]
        
        # Get acquisition scores if available
        acquisition_score = None
        if acquisition == 'entropy':
            entropy_file = os.path.join(dataset_dir, 'entropy_scores.pkl')
            if os.path.exists(entropy_file):
                with open(entropy_file, 'rb') as f:
                    entropy_scores = pickle.load(f)
                acquisition_score = float(entropy_scores[next_index])
        elif acquisition == 'disagreement':
            disagreement_file = os.path.join(dataset_dir, 'disagreement_scores.pkl')
            if os.path.exists(disagreement_file):
                with open(disagreement_file, 'rb') as f:
                    disagreement_scores = pickle.load(f)
                acquisition_score = float(disagreement_scores[next_index])
        
        return jsonify({
            'pair_index': next_index,
            'segments': segments.tolist(),
            'total_pairs': total_pairs,
            'labeled_pairs': len(labeled_pairs),
            'unlabeled_pairs': len(unlabeled_indices),
            'acquisition_method': acquisition,
            'acquisition_score': acquisition_score
        })
        
    except Exception as e:
        app.logger.error(f"Error getting next pair: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # Check if running in production
        in_production = os.environ.get('FLASK_ENV') == 'production'
        
        # Get port from environment or use default
        port = int(os.environ.get('PORT', 8443))
        
        # SSL context for HTTPS
        ssl_context = None
        cert_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ssl', 'cert.pem')
        key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ssl', 'key.pem')
        
        # Check if SSL certificates exist
        if os.path.exists(cert_path) and os.path.exists(key_path):
            ssl_context = (cert_path, key_path)
            app.logger.info("SSL certificates found, running with HTTPS")
        else:
            app.logger.warning("SSL certificates not found, running with HTTP only")
        
        if in_production:
            # Production settings
            app.logger.info("Running in production mode")
            if ssl_context:
                app.run(host='0.0.0.0', port=port, ssl_context=ssl_context, request_handler=CustomRequestHandler)
            else:
                app.logger.error("SSL certificates required for production mode")
                sys.exit(1)
        else:
            # Development settings
            app.logger.info("Running in development mode")
            app.run(host='0.0.0.0', port=port, ssl_context=ssl_context, debug=True, request_handler=CustomRequestHandler)
            
    except Exception as e:
        app.logger.critical(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1)