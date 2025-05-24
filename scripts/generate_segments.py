import sys

sys.path.append('/scr/aliang80/robot_pref')
import itertools

import numpy as np
import torch
from utils.data import load_tensordict, segment_episodes, segment_episodes_dynamic
import os

if __name__ == "__main__":
    BASE_DIR = '/scr/aliang80/robot_pref_site/data'
    task_name = "pick_up_green_and_slide_pot_right"
    data_path = os.path.join(BASE_DIR, task_name, f"{task_name}.pt")
    data = load_tensordict(data_path)
    data = {
        k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }
    segment_length = 32 
    print(f"Segmenting data with segment length {segment_length}")
    segments, segment_indices = segment_episodes_dynamic(data, segment_length)

    segment_start_end_indices = np.array(segment_indices)
    np.save(os.path.join(BASE_DIR, task_name, 'segment_start_end_indices.npy'), segment_start_end_indices)

    segment_indices = np.arange(len(segments))
    segment_pairs = list(itertools.combinations(segment_indices, 2))
    segment_pairs = np.array(segment_pairs)
    np.save(os.path.join(BASE_DIR, task_name, 'segment_pairs.npy'), segment_pairs)