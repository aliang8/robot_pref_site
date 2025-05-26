#!/usr/bin/env python3

import os
import torch
import pickle
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import glob
from tqdm import tqdm

def load_trajectory(traj_dir):
    """Load a single trajectory's data"""
    # Load policy output
    with open(os.path.join(traj_dir, 'policy_out.pkl'), 'rb') as f:
        policy_data = pickle.load(f)
    
    # Load observation dictionary
    with open(os.path.join(traj_dir, 'obs_dict.pkl'), 'rb') as f:
        obs_dict = pickle.load(f)
    
    # Load images
    img_dir = os.path.join(traj_dir, 'external_imgs')
    images = []
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')), key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
    for img_path in img_files:
        img = Image.open(img_path)
        img = np.array(img)
        images.append(img)
    images = np.stack(images)

    actions = [d["actions"] for d in policy_data]
    
    return {
        'obs': obs_dict["full_state"][:-1],
        'action': actions,
        'images': images[:-1],
    }

def process_task_data(task_name):
    """Process all trajectories for a given task"""
    source_dir = f"/scr/shared/clam/datasets/robot/{task_name}"
    target_dir = f"/scr/aliang80/robot_pref_site/data/{task_name}"
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Processing trajectories for task: {task_name}")
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")
    
    # Get all trajectory directories
    traj_dirs = sorted(glob.glob(os.path.join(source_dir, "traj*")))
    if not traj_dirs:
        raise ValueError(f"No trajectory directories found in {source_dir}")
    
    print(f"Found {len(traj_dirs)} trajectories")
    
    # Process each trajectory
    all_obs = []
    all_actions = []
    all_images = []
    all_episodes = []
    
    for episode_idx, traj_dir in enumerate(tqdm(traj_dirs)):
        try:
            traj_data = load_trajectory(traj_dir)
            
            # Convert to tensors
            obs_tensor = torch.tensor(traj_data['obs'])
            action_tensor = torch.tensor(traj_data['action'])
            image_tensor = torch.tensor(traj_data['images'])
            episode_tensor = torch.full((len(traj_data['action']),), episode_idx)
            
            all_obs.append(obs_tensor)
            all_actions.append(action_tensor)
            all_images.append(image_tensor)
            all_episodes.append(episode_tensor)
            
        except Exception as e:
            print(f"Error processing trajectory {traj_dir}: {e}")
            continue
    
    # Concatenate all data
    tensordict = {
        'obs': torch.cat(all_obs),
        'action': torch.cat(all_actions),
        'image': torch.cat(all_images),
        'episode': torch.cat(all_episodes)
    }
    
    # Save the tensordict
    output_path = os.path.join(target_dir, f"{task_name}.pt")
    torch.save(tensordict, output_path)
    print(f"Saved tensordict to {output_path}")
    print(f"Tensordict shapes:")
    for key, value in tensordict.items():
        print(f"  {key}: {value.shape}")

def main():
    parser = argparse.ArgumentParser(description='Process robot trajectory data into tensordict format')
    parser.add_argument('-t', '--task_name', type=str, help='Name of the task to process')
    args = parser.parse_args()
    
    process_task_data(args.task_name)

if __name__ == "__main__":
    main() 