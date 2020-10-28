import os
import numpy as np


def flow2rgb(flow_map, max_flow):
    flow_map_np = flow_map.numpy()
    flow_map_np = flow_map_np.squeeze()
    flow_map_np = flow_map_np.swapaxes(0,2)
    flow_map_np = flow_map_np.swapaxes(1,2)
    _, h, w = flow_map_np.shape
    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    if max_flow is not None:
        normalized_flow_map = flow_map_np / max_flow
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)


def apply_transform(image):
    # Transformation applied to the input image
    image = image/255
    image = image - [0.411,0.432,0.45]
    return image