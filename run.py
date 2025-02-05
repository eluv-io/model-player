
import argparse
import os
import json
from typing import List
import cv2
from common_ml.utils import nested_update
from common_ml.utils.files import get_file_type
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict

from src.model import PlayerModel
from config import config

def extract_xmp_as_dict(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    
    # Locate XMP metadata in the binary file
    xmp_start = data.find(b"<x:xmpmeta")
    xmp_end = data.find(b"</x:xmpmeta>") + len(b"</x:xmpmeta>")
    
    if xmp_start == -1 or xmp_end == -1:
        return {"error": "No XMP metadata found"}

    xmp_raw = data[xmp_start:xmp_end].decode("utf-8", errors="ignore")

    # Parse the XMP XML
    xmp_dict = parse_xmp_to_dict(xmp_raw)
    return xmp_dict

def parse_xmp_to_dict(xmp_raw):
    """Convert XMP XML string into a structured dictionary without namespaces."""
    try:
        root = ET.fromstring(xmp_raw)
        xmp_data = {}

        for elem in root.iter():
            # Remove namespace from tag names
            tag = elem.tag
            if "}" in tag:
                tag = tag.split("}")[-1]

            # Store text content or attributes
            if elem.text and elem.text.strip():
                xmp_data[tag] = elem.text.strip()
            for attr, value in elem.attrib.items():
                attr_name = attr.split("}")[-1] if "}" in attr else attr
                xmp_data[attr_name] = value

        return xmp_data
    except ET.ParseError:
        return {"error": "Failed to parse XMP XML"}

# Generate tag files from a list of video/image files and a runtime config
# Runtime config follows the schema found in celeb.model.RuntimeConfig
def run(file_paths: List[str], runtime_config: str=None):
    if runtime_config is None:
        cfg = config["model"]
    else:
        cfg = json.loads(runtime_config)
        cfg = nested_update(config["model"], cfg)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tags')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    model = PlayerModel(config["container"]["weights"], runtime_config=cfg)
    for fname in file_paths:
        if not os.path.exists(fname):
            raise FileNotFoundError(f"File {fname} not found")
        if not get_file_type(fname) == 'image':
            raise ValueError(f"File {fname} is not an image. Player Detection currently only supports image inputs.")
        img = cv2.imread(fname)
        # change color space to RGB
        img = img[:, :, ::-1]
         # get xmp data from image
        headline = extract_xmp_as_dict(fname).get('Headline', '')
        model.set_headline(headline)
        frametags = model.tag(img)
        with open(os.path.join(out_path, f"{os.path.basename(fname)}_imagetags.json"), 'w') as fout:
            fout.write(json.dumps([asdict(tag) for tag in frametags]))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_paths', nargs='+', type=str)
    parser.add_argument('--config', type=str, required=False)
    args = parser.parse_args()
    run(args.file_paths, args.config)