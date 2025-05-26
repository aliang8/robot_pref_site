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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
sys.path.append('/scr/aliang80/robot_pref')
from models.reward_models import EnsembleRewardModel
from select_next_pair import compute_disagreement, compute_entropy, select_next_pair
from train_reward_model import train_model
from utils.dataset import PreferenceDataset, create_data_loaders
from omegaconf import OmegaConf

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
            "https://snoopy1.usc.edu:8443",
            "http://10.136.20.183:5500"  # Add your IP address
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
DATA_DIR = "/scr/shared/datasets/robot_pref"
# Folder to store generated videos - use absolute path
VIDEO_DIR = os.path.join(ROOT_DIR, 'temp_videos')  # Point to /scr/aliang80/robot_pref_site/temp_videos
os.makedirs(VIDEO_DIR, exist_ok=True)

app.logger.info(f"Video directory: {VIDEO_DIR}")
app.logger.info(f"Base directory: {BASE_DIR}")
app.logger.info(f"Root directory: {ROOT_DIR}")
app.logger.info(f"Data directory: {DATA_DIR}")

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
        "https://snoopy1.usc.edu:8443",
        "http://10.136.20.183:5500"
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
        dataset_dir = os.path.join(DATA_DIR, dataset)
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
            if "reward" in trajectory_data:
                reward_a = float(trajectory_data['reward'][trajectory_a_start:trajectory_a_end].mean())
                reward_b = float(trajectory_data['reward'][trajectory_b_start:trajectory_b_end].mean())
            else:
                reward_a = 0.0
                reward_b = 0.0
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
    """Get list of available datasets that have the required files."""
    try:
        app.logger.info(f"Checking for datasets in: {DATA_DIR}")
        if not os.path.exists(DATA_DIR):
            app.logger.error(f"Data directory not found: {DATA_DIR}")
            return jsonify({'error': 'Data directory not found'}), 500
            
        datasets = []
        for item in os.listdir(DATA_DIR):
            item_path = os.path.join(DATA_DIR, item)
            if os.path.isdir(item_path):
                app.logger.debug(f"Checking directory: {item_path}")
                
                # Check for required files
                pt_files = [f for f in os.listdir(item_path) if f.endswith('.pt')]
                has_pt_file = len(pt_files) > 0
                has_segment_pairs = os.path.exists(os.path.join(item_path, 'segment_pairs.npy'))
                has_segment_indices = os.path.exists(os.path.join(item_path, 'segment_start_end_indices.npy'))
                
                app.logger.debug(f"Directory {item} - PT files: {has_pt_file}, Segment pairs: {has_segment_pairs}, Segment indices: {has_segment_indices}")
                
                if has_pt_file and has_segment_pairs and has_segment_indices:
                    datasets.append(item)
                    app.logger.info(f"Found valid dataset: {item}")
                else:
                    app.logger.warning(f"Directory {item} missing required files")
                    
        if not datasets:
            app.logger.warning("No valid datasets found")
            return jsonify({'datasets': [], 'warning': 'No valid datasets found'}), 200
            
        app.logger.info(f"Found {len(datasets)} valid datasets: {datasets}")
        return jsonify({'datasets': sorted(datasets)})
        
    except Exception as e:
        app.logger.error(f"Error getting available datasets: {str(e)}", exc_info=True)
        return jsonify({'error': f'Failed to get datasets: {str(e)}'}), 500

@app.route('/api/get-dataset-info', methods=['GET'])
def get_dataset_info():
    """Get information about the dataset including total pairs and labeled pairs."""
    try:
        dataset = request.args.get('dataset', 'assembly-v2')
        dataset_dir = os.path.join(DATA_DIR, dataset)
        
        app.logger.info(f"Getting info for dataset: {dataset} in directory: {dataset_dir}")
        
        if not os.path.isdir(dataset_dir):
            app.logger.error(f"Dataset directory not found: {dataset_dir}")
            return jsonify({'error': f'Dataset directory not found: {dataset}'}), 404
            
        # Load segment pairs
        segment_pairs_path = os.path.join(dataset_dir, 'segment_pairs.npy')
        if not os.path.exists(segment_pairs_path):
            app.logger.error(f"Segment pairs file not found: {segment_pairs_path}")
            return jsonify({'error': 'Segment pairs file not found'}), 404
            
        segment_pairs = np.load(segment_pairs_path)
        total_pairs = len(segment_pairs)
        app.logger.info(f"Total pairs: {total_pairs}")
        
        # Count labeled pairs from preferences directory
        dataset_pref_dir = os.path.join(dataset_dir, 'preferences')
        labeled_pairs = set()
        
        if os.path.exists(dataset_pref_dir):
            for file in os.listdir(dataset_pref_dir):
                if file.endswith('.json'):
                    try:
                        with open(os.path.join(dataset_pref_dir, file), 'r') as f:
                            prefs = json.load(f)
                            for pref in prefs.get('preferences', []):
                                if pref.get('pair_index') is not None:
                                    labeled_pairs.add(pref.get('pair_index'))
                    except (json.JSONDecodeError, IOError) as e:
                        app.logger.warning(f"Error reading preference file {file}: {str(e)}")
                        continue
        
        num_labeled = len(labeled_pairs)
        app.logger.info(f"Labeled pairs: {num_labeled}")
        
        return jsonify({
            'total_pairs': total_pairs,
            'labeled_pairs': num_labeled,
            'unlabeled_pairs': total_pairs - num_labeled
        })
    
    except Exception as e:
        app.logger.error(f"Error getting dataset info: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-preferences', methods=['POST'])
def save_preferences():
    try:
        data = request.json
        if not data:
            app.logger.error("No JSON data received in request")
            return jsonify({'error': 'No JSON data received'}), 400
            
        dataset = data.get('dataset', 'assembly-v2')
        session_id = data.get('session_id')
        
        if not session_id:
            app.logger.error("No session_id provided in request data")
            return jsonify({'error': 'session_id is required'}), 400
            
        app.logger.info(f"Saving preferences for session {session_id} in dataset {dataset}")
        
        # Save preferences in dataset-specific directory
        dataset_pref_dir = os.path.join(DATA_DIR, dataset, 'preferences')
        os.makedirs(dataset_pref_dir, exist_ok=True)
        
        filename = f"preferences_{session_id}.json"
        filepath = os.path.join(dataset_pref_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        app.logger.info(f"Successfully saved preferences to {filepath}")
        return jsonify({"success": True})
        
    except Exception as e:
        app.logger.error(f"Error saving preferences: {str(e)}", exc_info=True)
        return jsonify({'error': f'Failed to save preferences: {str(e)}'}), 500

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
    """Get similar segments based on DTW distance."""
    try:
        app.logger.info("Received request for similar segments")
        # Get parameters
        dataset = request.args.get('dataset', 'assembly-v2')
        segment_index = request.args.get('segment_index')
        k = int(request.args.get('k', '5'))  # Number of similar/dissimilar segments to return
        
        app.logger.info(f"Parameters - dataset: {dataset}, segment_index: {segment_index}, k: {k}")
        
        if segment_index is None:
            return jsonify({'error': 'Missing segment_index parameter'}), 400
            
        try:
            segment_index = int(segment_index)
        except ValueError:
            return jsonify({'error': 'Invalid segment_index format'}), 400
            
        # Load dataset
        dataset_dir = os.path.join(DATA_DIR, dataset)
        if not os.path.exists(dataset_dir):
            return jsonify({'error': f'Dataset {dataset} not found'}), 404
            
        dtw_matrix_path_pkl = os.path.join(dataset_dir, 'dtw_matrix_32.pkl')
        
        dtw_matrix = None
        if os.path.exists(dtw_matrix_path_pkl):
            import pickle
            with open(dtw_matrix_path_pkl, 'rb') as f:
                dtw_matrix, _ = pickle.load(f)
        else:
            return jsonify({'error': 'DTW matrix not found. Looked for dtw_matrix.npy, dtw_matrix_32.pkl, or dtw_matrix.pkl'}), 404
        
        # Validate segment_index is within bounds
        if segment_index >= len(dtw_matrix) or segment_index < 0:
            return jsonify({'error': f'Segment index {segment_index} out of bounds. Valid range: 0-{len(dtw_matrix)-1}'}), 400
            
        # Get distances for the target segment
        distances = dtw_matrix[segment_index]
        app.logger.info(f"DTW matrix shape: {dtw_matrix.shape}, distances shape: {distances.shape if hasattr(distances, 'shape') else type(distances)}")
        
        # Get indices of k most similar and k most dissimilar segments
        similar_indices = np.argsort(distances)[1:k+1]  # Skip first (self)
        dissimilar_indices = np.argsort(distances)[-k:][::-1]
        
        app.logger.info(f"Similar indices: {similar_indices}, Dissimilar indices: {dissimilar_indices}")
        
        # Load rewards if available, otherwise compute from trajectory data
        rewards_path = os.path.join(dataset_dir, 'rewards.npy')
        rewards = None
        if os.path.exists(rewards_path):
            rewards = np.load(rewards_path)
        else:
            # Try to compute rewards from trajectory data
            data_path = os.path.join(dataset_dir, f"{dataset}.pt")
            if os.path.exists(data_path):
                try:
                    trajectory_data = torch.load(data_path)
                    if "reward" in trajectory_data:
                        segment_indices_path = os.path.join(dataset_dir, 'segment_start_end_indices.npy')
                        if os.path.exists(segment_indices_path):
                            segment_indices = np.load(segment_indices_path)
                            segment_indices = segment_indices.reshape(-1, 2)
                            # Compute mean reward for each segment
                            rewards = []
                            for start, end in segment_indices:
                                segment_rewards = trajectory_data['reward'][start:end]
                                if isinstance(segment_rewards, torch.Tensor):
                                    segment_rewards = segment_rewards.cpu().numpy()
                                rewards.append(float(np.mean(segment_rewards)))
                            rewards = np.array(rewards)
                            app.logger.info(f"Computed rewards for {len(rewards)} segments")
                except Exception as e:
                    app.logger.warning(f"Could not compute rewards from trajectory data: {str(e)}")
        
        # Load trajectory data and segment indices for video generation
        data_path = os.path.join(dataset_dir, f"{dataset}.pt")
        segment_indices_path = os.path.join(dataset_dir, 'segment_start_end_indices.npy')
        
        trajectory_data = None
        segment_indices = None
        if os.path.exists(data_path) and os.path.exists(segment_indices_path):
            trajectory_data = torch.load(data_path)
            # Convert tensors to numpy
            trajectory_data = {
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                for k, v in trajectory_data.items()
            }
            segment_indices = np.load(segment_indices_path)
            segment_indices = segment_indices.reshape(-1, 2)
        
        # Function to get video URL for a segment
        def get_segment_info(idx):
            base_url = request.host_url.rstrip('/')
            video_filename = f"{dataset}_segment_{idx}.mp4"
            video_path = os.path.join(VIDEO_DIR, video_filename)
            
            # Generate video if it doesn't exist
            if not os.path.exists(video_path) and trajectory_data is not None and segment_indices is not None:
                try:
                    if idx < len(segment_indices):
                        start, end = segment_indices[idx]
                        segment_frames = trajectory_data['image'][start:end]
                        
                        # Ensure frames are in the correct format
                        if segment_frames.dtype != np.uint8:
                            if segment_frames.max() <= 1.0:
                                segment_frames = (segment_frames * 255).astype(np.uint8)
                            else:
                                segment_frames = segment_frames.astype(np.uint8)
                        
                        generate_video(segment_frames, video_path)
                        app.logger.info(f"Generated video for segment {idx}")
                except Exception as e:
                    app.logger.error(f"Failed to generate video for segment {idx}: {str(e)}")
            
            # Handle distance conversion - it might be a tuple, array, or scalar
            distance_value = distances[idx]
            if isinstance(distance_value, (tuple, list, np.ndarray)):
                # If it's a tuple/array, take the first element or flatten it
                distance_value = np.array(distance_value).flatten()[0]
            
            info = {
                'segment_index': int(idx),
                'video_url': f'{base_url}/videos/{video_filename}',
                'distance': float(distance_value)
            }
            if rewards is not None and idx < len(rewards):
                reward_value = rewards[idx]
                if isinstance(reward_value, (tuple, list, np.ndarray)):
                    reward_value = np.array(reward_value).flatten()[0]
                info['reward'] = float(reward_value)
            else:
                info['reward'] = 0.0  # Default reward if not available
            return info
        
        # Prepare response
        response = {
            'target': get_segment_info(segment_index),
            'similar': [get_segment_info(idx) for idx in similar_indices],
            'dissimilar': [get_segment_info(idx) for idx in dissimilar_indices]
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/active-preferences', methods=['POST'])
def handle_active_preference():
    """Handle active preference submission and model training."""
    try:
        data = request.json
        dataset = data.get('taskId', 'assembly-v2')
        pair_index = data.get('pairIndex')
        preference = data.get('preference')
        acquisition_method = data.get('acquisitionMethod', 'disagreement')
        timestamp = data.get('timestamp')
        session_id = data.get('sessionId')  # Get session ID from request
        max_iterations = int(data.get('maxIterations', 10))  # Ensure integer
        current_iteration = int(data.get('currentIteration', 0))  # Ensure integer
        
        app.logger.info(f"Received active preference - Dataset: {dataset}, Pair: {pair_index}, Preference: {preference}, Session: {session_id}, Iteration: {current_iteration}/{max_iterations}")
        
        # Get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        app.logger.info(f"Using device: {device}")

        # Save preference in session-specific file
        dataset_dir = os.path.join(DATA_DIR, dataset)
        active_pref_dir = os.path.join(dataset_dir, 'active_preferences')
        os.makedirs(active_pref_dir, exist_ok=True)
        
        app.logger.info(f"Saving preference for session {session_id}...")
        
        # Use session-specific preference file
        active_pref_file = os.path.join(active_pref_dir, f'active_preferences_{session_id}.json')
        try:
            with open(active_pref_file, 'r') as f:
                active_prefs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            active_prefs = {
                'session_id': session_id,
                'max_iterations': max_iterations,
                'preferences': []
            }
            
        # Add new preference (will be updated with training metrics later)
        new_preference = {
            'pair_index': pair_index,
            'preference': preference,
            'acquisition_method': acquisition_method,
            'timestamp': timestamp,
            'iteration': current_iteration
        }
        active_prefs['preferences'].append(new_preference)
        
        with open(active_pref_file, 'w') as f:
            json.dump(active_prefs, f, indent=2)

        # Check if we've reached max iterations
        if current_iteration >= max_iterations:
            return jsonify({
                "success": True,
                "session_complete": True,
                "total_iterations": current_iteration,
                "max_iterations": max_iterations
            })
        
        # Train reward model
        app.logger.info(f"Starting model training with {len(active_prefs['preferences'])} preferences...")
        
        # Get paths
        data_path = os.path.join(dataset_dir, f"{dataset}.pt")
        segment_pairs_path = os.path.join(dataset_dir, 'segment_pairs.npy')
        segment_indices_path = os.path.join(dataset_dir, 'segment_start_end_indices.npy')
        
        app.logger.info("Loading trajectory data and segments...")
        # Load data and segments
        data = torch.load(data_path)
        # Move data to CPU first to avoid CUDA issues in dataloader workers
        data = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        
        segment_pairs = np.load(segment_pairs_path)
        segment_indices = np.load(segment_indices_path)
        segment_indices = segment_indices.reshape(-1, 2)
        
        # Convert preferences to numerical values
        preferences = []
        for pref in active_prefs['preferences']:
            if pref['preference'] == 'A':
                preferences.append(1)
            elif pref['preference'] == 'B':
                preferences.append(0)
            else:  # equal
                preferences.append(0.5)
                
        # Get pairs that have preferences
        labeled_pairs = []
        labeled_preferences = []
        for i, pref in enumerate(active_prefs['preferences']):
            labeled_pairs.append(segment_pairs[pref['pair_index']])
            labeled_preferences.append(preferences[i])
            
        labeled_pairs = np.array(labeled_pairs)
        labeled_preferences = np.array(labeled_preferences)
        
        app.logger.info("Creating preference dataset...")
        # Create dataset with data on CPU
        preference_dataset = PreferenceDataset(
            data,
            labeled_pairs,
            segment_indices,
            labeled_preferences,
            normalize_obs=True,
            norm_method="standard",
            use_images=False
        )
        
        # Use single worker to avoid CUDA issues
        # For small datasets, use all data for training and validation
        dataset_size = len(preference_dataset)
        if dataset_size < 10:
            # For very small datasets, use all data for both train and validation
            train_ratio = 1.0
            val_ratio = 0.0
        else:
            train_ratio = 0.8
            val_ratio = 0.1
            
        dataloaders = create_data_loaders(
            preference_dataset,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            batch_size=min(32, dataset_size),  # Adjust batch size for small datasets
        )
        
        app.logger.info("Initializing ensemble reward model...")
        # Initialize model
        state_dim = data['obs'].shape[1]
        action_dim = data['action'].shape[1]
        
        model = EnsembleRewardModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[256, 256],
            num_models=5,
            device=device
        ).to(device)
        
        # Train model using quick training settings
        model_dir = os.path.join(dataset_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'model_iter_{len(active_prefs["preferences"])}.pt')
        
        # Quick training settings
        num_epochs = 10
        lr = 1e-3
        
        # Use train loader for validation if val loader is empty
        val_loader = dataloaders['val'] if len(dataloaders['val'].dataset) > 0 else dataloaders['train']
        
        app.logger.info(f"Training model for {num_epochs} epochs...")
        model, train_losses, val_losses = train_model(
            model,
            dataloaders['train'],
            val_loader,
            device,
            num_epochs=num_epochs,
            lr=lr,
            wandb_run=None,
            is_ensemble=True,
            output_path=None
        )
        
        # Save model checkpoint
        app.logger.info(f"Saving model checkpoint to {model_path}...")
        torch.save(model.state_dict(), model_path)
        
        # Update the preference with training metrics
        new_preference['train_loss'] = float(train_losses[-1])
        new_preference['val_loss'] = float(val_losses[-1])
        
        # Save updated preferences
        with open(active_pref_file, 'w') as f:
            json.dump(active_prefs, f, indent=2)
        
        app.logger.info("Computing acquisition scores...")
        # Process data in smaller batches to avoid memory issues
        batch_size = 100
        num_pairs = len(segment_pairs)
        all_logits = []  # Store all logits
        
        for i in range(0, num_pairs, batch_size):
            batch_pairs = segment_pairs[i:i + batch_size]
            batch_obs = []
            batch_actions = []
            
            for pair in batch_pairs:
                seg1, seg2 = pair
                start1, end1 = segment_indices[seg1]
                start2, end2 = segment_indices[seg2]
                
                # Get observations and actions for both segments
                obs1 = data['obs'][start1:end1]
                obs2 = data['obs'][start2:end2]
                action1 = data['action'][start1:end1]
                action2 = data['action'][start2:end2]
                
                # Concatenate segments
                obs = torch.cat([obs1, obs2], dim=0)
                action = torch.cat([action1, action2], dim=0)
                
                batch_obs.append(obs)
                batch_actions.append(action)
            
            # Stack batch and move to device
            batch_obs = torch.stack(batch_obs).to(device)
            batch_actions = torch.stack(batch_actions).to(device)
            
            # Get model predictions for this batch
            with torch.no_grad():
                batch_logits = model(batch_obs, batch_actions)
                all_logits.append(batch_logits)  # Store batch predictions
            
        # Concatenate all logits
        logits = torch.cat(all_logits, dim=1)  # Concatenate along the pairs dimension
            
        # Compute acquisition scores
        if acquisition_method == 'disagreement':
            scores = compute_disagreement(logits)
            score_path = os.path.join(dataset_dir, 'disagreement_scores.pkl')
        else:  # entropy
            scores = compute_entropy(logits)
            score_path = os.path.join(dataset_dir, 'entropy_scores.pkl')
            
        # Save scores
        with open(score_path, 'wb') as f:
            pickle.dump(scores.cpu().numpy(), f)
        
        app.logger.info("Active learning workflow completed successfully")
        return jsonify({
            "success": True,
            "model_path": model_path,
            "train_loss": float(train_losses[-1]),
            "val_loss": float(val_losses[-1]),
            "message": f"Training completed successfully! Train loss: {train_losses[-1]:.4f}"
        })
        
    except Exception as e:
        app.logger.error(f"Error in active learning workflow: {str(e)}", exc_info=True)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-next-pair', methods=['GET'])
def get_next_pair():
    """Get next pair using active learning."""
    try:
        dataset = request.args.get('dataset', 'assembly-v2')
        acquisition = request.args.get('acquisition', 'disagreement')
        session_id = request.args.get('session_id')  # Get session ID
        dataset_dir = os.path.join(DATA_DIR, dataset)
        
        app.logger.info(f"Getting next pair for dataset={dataset}, acquisition={acquisition}, session={session_id}")
        
        # Load segment pairs
        segment_pairs = np.load(os.path.join(dataset_dir, 'segment_pairs.npy'))
        total_pairs = len(segment_pairs)
        
        # Get labeled pairs from session-specific active preferences
        labeled_pairs = set()
        if session_id:
            active_pref_file = os.path.join(dataset_dir, 'active_preferences', f'active_preferences_{session_id}.json')
            try:
                with open(active_pref_file, 'r') as f:
                    active_prefs = json.load(f)
                    for pref in active_prefs.get('preferences', []):
                        labeled_pairs.add(pref.get('pair_index'))
                    app.logger.info(f"Found {len(labeled_pairs)} labeled pairs in session {session_id}")
            except (FileNotFoundError, json.JSONDecodeError):
                app.logger.info(f"No active preferences file found for session {session_id}")
            
        # Get unlabeled pairs
        unlabeled_indices = list(set(range(total_pairs)) - labeled_pairs)
        
        if not unlabeled_indices:
            return jsonify({'message': 'All pairs have been labeled'}), 200
            
        # Select next pair based on acquisition method
        next_index = None
        acquisition_score = None
        
        if acquisition == 'disagreement':
            # Load disagreement scores
            try:
                with open(os.path.join(dataset_dir, 'disagreement_scores.pkl'), 'rb') as f:
                    disagreement_scores = pickle.load(f)
                # Get pair with highest disagreement among unlabeled pairs
                # For ensemble scores, take mean over all dimensions to get a single scalar
                unlabeled_scores = [(i, np.mean(disagreement_scores[i]).item()) for i in unlabeled_indices]
                next_index, acquisition_score = max(unlabeled_scores, key=lambda x: x[1])
            except (FileNotFoundError, pickle.UnpicklingError):
                # Fall back to random selection if scores not available
                next_index = random.choice(unlabeled_indices)
                
        elif acquisition == 'entropy':
            # Load entropy scores
            try:
                with open(os.path.join(dataset_dir, 'entropy_scores.pkl'), 'rb') as f:
                    entropy_scores = pickle.load(f)
                # Get pair with highest entropy among unlabeled pairs
                unlabeled_scores = [(i, entropy_scores[i]) for i in unlabeled_indices]
                next_index, acquisition_score = max(unlabeled_scores, key=lambda x: x[1])
            except (FileNotFoundError, pickle.UnpicklingError):
                # Fall back to random selection if scores not available
                next_index = random.choice(unlabeled_indices)
        else:
            # Random selection
            next_index = random.choice(unlabeled_indices)
            
        return jsonify({
            'pair_index': next_index,
            'segments': segment_pairs[next_index].tolist() if next_index is not None else None,
            'total_pairs': total_pairs,
            'labeled_pairs': len(labeled_pairs),
            'unlabeled_pairs': len(unlabeled_indices),
            'acquisition_method': acquisition,
            'acquisition_score': float(acquisition_score) if acquisition_score is not None else None
        })
        
    except Exception as e:
        app.logger.error(f"Error getting next pair: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-acquisition-stats', methods=['GET'])
def get_acquisition_stats():
    """Get acquisition score statistics for plotting."""
    try:
        dataset = request.args.get('dataset', 'assembly-v2')
        acquisition = request.args.get('acquisition', 'disagreement')
        session_id = request.args.get('session_id')
        dataset_dir = os.path.join(DATA_DIR, dataset)
        
        # Load acquisition scores
        score_file = f'{acquisition}_scores.pkl'
        score_path = os.path.join(dataset_dir, score_file)
        
        if not os.path.exists(score_path):
            return jsonify({'error': f'No {acquisition} scores available'}), 404
            
        with open(score_path, 'rb') as f:
            scores = pickle.load(f)
            
        # Handle ensemble scores (take mean if multidimensional)
        if len(scores.shape) > 1:
            scores = np.mean(scores, axis=1)
            
        # Get labeled pairs from session
        labeled_pairs = set()
        if session_id:
            active_pref_file = os.path.join(dataset_dir, 'active_preferences', f'active_preferences_{session_id}.json')
            try:
                with open(active_pref_file, 'r') as f:
                    active_prefs = json.load(f)
                    for pref in active_prefs.get('preferences', []):
                        labeled_pairs.add(pref.get('pair_index'))
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        
        # Create statistics
        stats = {
            'total_pairs': len(scores),
            'labeled_pairs': len(labeled_pairs),
            'scores': {
                'all': scores.tolist(),
                'labeled': [scores[i] for i in labeled_pairs],
                'unlabeled': [scores[i] for i in range(len(scores)) if i not in labeled_pairs]
            },
            'statistics': {
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'median': float(np.median(scores))
            },
            'acquisition_method': acquisition
        }
        
        return jsonify(stats)
        
    except Exception as e:
        app.logger.error(f"Error getting acquisition stats: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/acquisition-plot', methods=['GET'])
def get_acquisition_plot():
    """Generate acquisition score distribution plot."""
    try:
        dataset = request.args.get('dataset', 'assembly-v2')
        acquisition = request.args.get('acquisition', 'disagreement')
        session_id = request.args.get('session_id')
        dataset_dir = os.path.join(DATA_DIR, dataset)
        
        # Load acquisition scores
        score_file = f'{acquisition}_scores.pkl'
        score_path = os.path.join(dataset_dir, score_file)
        
        if not os.path.exists(score_path):
            return jsonify({'error': f'No {acquisition} scores available'}), 404
            
        with open(score_path, 'rb') as f:
            scores = pickle.load(f)
            
        # Handle ensemble scores (take mean if multidimensional)
        if len(scores.shape) > 1:
            scores = np.mean(scores, axis=1)
            
        # Get labeled pairs from session
        labeled_pairs = set()
        if session_id:
            active_pref_file = os.path.join(dataset_dir, 'active_preferences', f'active_preferences_{session_id}.json')
            try:
                with open(active_pref_file, 'r') as f:
                    active_prefs = json.load(f)
                    for pref in active_prefs.get('preferences', []):
                        labeled_pairs.add(pref.get('pair_index'))
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Get unlabeled and labeled scores
        unlabeled_scores = [scores[i] for i in range(len(scores)) if i not in labeled_pairs]
        labeled_scores = [scores[i] for i in labeled_pairs] if labeled_pairs else []
        
        # Plot histogram of unlabeled scores
        plt.hist(unlabeled_scores, bins=30, alpha=0.7, color='skyblue', label=f'Unlabeled ({len(unlabeled_scores)})', edgecolor='black')
        
        # Overlay labeled scores if any
        if labeled_scores:
            plt.hist(labeled_scores, bins=30, alpha=0.8, color='orange', label=f'Labeled ({len(labeled_scores)})', edgecolor='black')
        
        plt.xlabel('Acquisition Score')
        plt.ylabel('Frequency')
        plt.title(f'{acquisition.capitalize()} Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Mean: {np.mean(scores):.4f}\nStd: {np.std(scores):.4f}\nMin: {np.min(scores):.4f}\nMax: {np.max(scores):.4f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save to BytesIO
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return send_file(img_buffer, mimetype='image/png')
        
    except Exception as e:
        app.logger.error(f"Error generating acquisition plot: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/training-plot', methods=['GET'])
def get_training_plot():
    """Generate training progress plot."""
    try:
        dataset = request.args.get('dataset', 'assembly-v2')
        session_id = request.args.get('session_id')
        dataset_dir = os.path.join(DATA_DIR, dataset)
        
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
        
        # Load training history from session file
        active_pref_file = os.path.join(dataset_dir, 'active_preferences', f'active_preferences_{session_id}.json')
        
        try:
            with open(active_pref_file, 'r') as f:
                session_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return jsonify({'error': 'No session data found'}), 404
        
        # Extract training metrics if available
        preferences = session_data.get('preferences', [])
        iterations = []
        train_losses = []
        val_losses = []
        
        for i, pref in enumerate(preferences):
            if 'train_loss' in pref and 'val_loss' in pref:
                iterations.append(i + 1)
                train_losses.append(pref['train_loss'])
                val_losses.append(pref['val_loss'])
        
        if not iterations:
            return jsonify({'error': 'No training data available'}), 404
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, train_losses, 'o-', color='red', label='Train Loss', linewidth=2, markersize=6)
        plt.plot(iterations, val_losses, 's-', color='blue', label='Validation Loss', linewidth=2, markersize=6)
        
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save to BytesIO
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return send_file(img_buffer, mimetype='image/png')
        
    except Exception as e:
        app.logger.error(f"Error generating training plot: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/start-active-session', methods=['POST'])
def start_active_session():
    """Start a new active learning session."""
    try:
        data = request.json
        dataset = data.get('dataset', 'assembly-v2')
        max_iterations = int(data.get('maxIterations', 10))
        session_id = data.get('sessionId')
        
        # Validate max_iterations
        if max_iterations not in [10, 25, 50, 100]:
            return jsonify({'error': 'Invalid max_iterations value. Must be 10, 25, 50, or 100'}), 400
            
        # Create new session file
        dataset_dir = os.path.join(DATA_DIR, dataset)
        active_pref_dir = os.path.join(dataset_dir, 'active_preferences')
        os.makedirs(active_pref_dir, exist_ok=True)
        
        session_file = os.path.join(active_pref_dir, f'active_preferences_{session_id}.json')
        session_data = {
            'session_id': session_id,
            'dataset': dataset,
            'max_iterations': max_iterations,
            'start_time': datetime.now().isoformat(),
            'preferences': []
        }
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
            
        # Clear any existing acquisition scores for this dataset
        for score_file in ['disagreement_scores.pkl', 'entropy_scores.pkl']:
            file_path = os.path.join(dataset_dir, score_file)
            if os.path.exists(file_path):
                os.remove(file_path)
                
        return jsonify({
            'success': True,
            'session_id': session_id,
            'max_iterations': max_iterations,
            'message': f'Started new active learning session with {max_iterations} iterations'
        })
        
    except Exception as e:
        app.logger.error(f"Error starting active learning session: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# Add static file serving
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

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