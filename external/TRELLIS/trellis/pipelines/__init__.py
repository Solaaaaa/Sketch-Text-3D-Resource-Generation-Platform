from . import samplers
from .trellis_image_to_3d import TrellisImageTo3DPipeline


def from_pretrained(path: str):
    """
    Load a pipeline from a local model folder.

    Args:
        path: The local path to the model directory.
    """
    import os
    import json
    config_file = f"{path}/pipeline.json"
    if not os.path.isfile(config_file):
        raise FileNotFoundError(
            f"TRELLIS pipeline.json is missing from local path: {path}"
        )

    with open(config_file, 'r') as f:
        config = json.load(f)
    return globals()[config['name']].from_pretrained(path)
