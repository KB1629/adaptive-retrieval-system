#!/usr/bin/env python3
"""Quick test to see how router classifies specific pages."""

import sys
import os
os.chdir('adaptive-retrieval-system')
sys.path.insert(0, 'src')

from PIL import Image
import numpy as np
from models import Page
from router.heuristic import HeuristicRouter
from router.base import RouterConfig

# Initialize router
config = RouterConfig(text_threshold=0.5, min_confidence=0.4)
router = HeuristicRouter(config=config)

# Test pages 7, 8, 9
for page_num in [7, 8, 9]:
    page_path = f"../data/docvqa_sample/page_{page_num:03d}.png"
    
    # Load image
    img = Image.open(page_path).convert("RGB")
    img_array = np.array(img)
    
    # Create Page object
    page = Page(
        page_id=f"page_{page_num:03d}",
        image=img_array,
        metadata={"source": page_path}
    )
    
    # Classify
    result = router.classify(page)
    
    print(f"\nPage {page_num:03d}:")
    print(f"  Modality: {result.modality}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Features:")
    for key, value in result.features.items():
        print(f"    {key}: {value:.3f}")
