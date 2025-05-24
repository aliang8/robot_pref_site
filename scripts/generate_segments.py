import json 
import sys 
sys.path.append('/scr/aliang80/robot_pref')
import torch 
import itertools
import numpy as np 

from utils.data import load_tensordict
from utils.data import segment_episodes


if __name__ == "__main__":
    data_path = '/scr/aliang80/robot_pref_site/data/buffer_assembly-v2_balanced.pt'
    data = load_tensordict(data_path)
    data = {
        k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }
    segment_length = 32 
    print(f"Segmenting data with segment length {segment_length}")
    segments, segment_indices = segment_episodes(data, segment_length)

    segment_start_end_indices = np.array(segment_indices)
    np.save('/scr/aliang80/robot_pref_site/data/segment_start_end_indices.npy', segment_start_end_indices)

    segment_indices = np.arange(len(segments))
    segment_pairs = list(itertools.combinations(segment_indices, 2))
    segment_pairs = np.array(segment_pairs)
    np.save('/scr/aliang80/robot_pref_site/data/segment_pairs.npy', segment_pairs)