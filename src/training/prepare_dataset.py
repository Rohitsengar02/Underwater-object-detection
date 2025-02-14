import os
import json
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import shutil
from pathlib import Path
import yaml

def create_dataset_structure():
    """Create the basic dataset directory structure"""
    dataset_dir = Path('datasets/underwater_objects')
    train_dir = dataset_dir / 'train'
    val_dir = dataset_dir / 'val'
    
    # Create main directories
    for d in [train_dir, val_dir]:
        (d / 'images').mkdir(parents=True, exist_ok=True)
        (d / 'labels').mkdir(parents=True, exist_ok=True)
    
    return dataset_dir, train_dir, val_dir

def create_class_definitions():
    """Define the comprehensive underwater object classes"""
    classes = {
        # Marine Life - Fish Species (100+ types)
        'fish': {
            'common_fish': [
                'tuna', 'salmon', 'cod', 'trout', 'bass', 'angelfish', 'clownfish',
                'butterflyfish', 'grouper', 'parrotfish', 'lionfish', 'surgeonfish',
                'triggerfish', 'damselfish', 'wrasse', 'goby', 'blenny', 'snapper',
                'barracuda', 'mackerel', 'sardine', 'anchovy', 'herring', 'mullet'
            ],
            'reef_fish': [
                'moorish_idol', 'tang', 'chromis', 'anthias', 'cardinalfish',
                'hawkfish', 'dartfish', 'filefish', 'pufferfish', 'boxfish',
                'trumpetfish', 'cornetfish', 'frogfish', 'scorpionfish'
            ],
            'deep_sea_fish': [
                'anglerfish', 'viperfish', 'lanternfish', 'gulper_eel', 'oarfish',
                'dragonfish', 'fangtooth', 'hatchetfish', 'pelican_eel'
            ]
        },
        
        # Large Marine Life (50+ types)
        'large_marine_life': {
            'sharks': [
                'great_white', 'tiger_shark', 'hammerhead', 'reef_shark', 'nurse_shark',
                'whale_shark', 'bull_shark', 'mako_shark', 'thresher_shark', 'blue_shark',
                'blacktip_reef_shark', 'lemon_shark', 'sand_tiger_shark'
            ],
            'rays': [
                'manta_ray', 'stingray', 'eagle_ray', 'electric_ray', 'devil_ray',
                'butterfly_ray', 'cownose_ray', 'spotted_ray', 'blue_spotted_ray'
            ],
            'marine_mammals': [
                'dolphin', 'whale', 'seal', 'sea_lion', 'dugong', 'manatee', 'orca',
                'humpback_whale', 'blue_whale', 'sperm_whale', 'beluga_whale'
            ]
        },
        
        # Invertebrates (60+ types)
        'invertebrates': {
            'cephalopods': [
                'octopus', 'squid', 'cuttlefish', 'nautilus', 'blue_ringed_octopus',
                'giant_squid', 'vampire_squid', 'reef_squid', 'bobtail_squid'
            ],
            'crustaceans': [
                'crab', 'lobster', 'shrimp', 'hermit_crab', 'mantis_shrimp',
                'barnacle', 'krill', 'copepod', 'spider_crab', 'king_crab',
                'decorator_crab', 'cleaner_shrimp', 'pistol_shrimp'
            ],
            'other_invertebrates': [
                'jellyfish', 'sea_anemone', 'sea_cucumber', 'sea_urchin', 'starfish',
                'brittle_star', 'sea_lily', 'nudibranch', 'sea_slug', 'sea_hare',
                'feather_star', 'christmas_tree_worm', 'tube_worm', 'sea_pen'
            ]
        },
        
        # Coral and Marine Plants (40+ types)
        'marine_plants': {
            'coral_types': [
                'brain_coral', 'staghorn_coral', 'table_coral', 'mushroom_coral',
                'soft_coral', 'sea_fan', 'fire_coral', 'pillar_coral', 'elkhorn_coral',
                'bubble_coral', 'plate_coral', 'finger_coral', 'lettuce_coral'
            ],
            'seaweed_and_grass': [
                'kelp', 'sea_grass', 'sargassum', 'red_algae', 'green_algae',
                'brown_algae', 'coralline_algae', 'sea_lettuce', 'dead_mans_fingers'
            ]
        },
        
        # Underwater Structures (30+ types)
        'structures': {
            'natural': [
                'reef', 'rock', 'cave', 'sand', 'mud', 'seamount', 'trench',
                'underwater_volcano', 'thermal_vent', 'coral_wall', 'sinkhole',
                'underwater_canyon', 'limestone_formation'
            ],
            'artificial': [
                'shipwreck', 'artificial_reef', 'pipeline', 'cable', 'anchor',
                'buoy', 'underwater_structure', 'marine_debris', 'underwater_statue',
                'diving_platform', 'underwater_habitat', 'research_equipment'
            ]
        },
        
        # Human Activity (50+ types)
        'human_activity': {
            'divers': [
                'scuba_diver', 'free_diver', 'technical_diver', 'rescue_diver',
                'underwater_photographer', 'marine_researcher', 'cave_diver'
            ],
            'equipment': [
                'diving_gear', 'oxygen_tank', 'diving_mask', 'diving_fins',
                'wetsuit', 'underwater_camera', 'diving_light', 'dive_computer',
                'regulator', 'bcd', 'dive_flag', 'safety_sausage'
            ],
            'vessels': [
                'submarine', 'ship', 'boat', 'yacht', 'fishing_boat', 'cargo_ship',
                'cruise_ship', 'sailboat', 'speedboat', 'research_vessel', 'dive_boat'
            ],
            'fishing_gear': [
                'fishing_net', 'fishing_line', 'fishing_hook', 'fishing_trap',
                'fishing_rod', 'marker_buoy', 'trawl_net', 'long_line'
            ]
        },
        
        # Marine Debris (20+ types)
        'pollution': {
            'debris_types': [
                'plastic_bag', 'bottle', 'fishing_net', 'rope', 'metal_debris',
                'tire', 'electronic_waste', 'construction_debris', 'abandoned_gear',
                'microplastic', 'oil_spill', 'chemical_pollution'
            ]
        }
    }
    
    return classes

def create_class_mapping(classes):
    """Create a flat mapping of all classes to numeric IDs"""
    flat_classes = []
    
    def flatten_dict(d):
        for v in d.values():
            if isinstance(v, list):
                flat_classes.extend(v)
            elif isinstance(v, dict):
                flatten_dict(v)
    
    flatten_dict(classes)
    flat_classes = sorted(list(set(flat_classes)))
    
    return {cls: idx for idx, cls in enumerate(flat_classes)}

def create_dataset():
    """Create the underwater object detection dataset"""
    print("Creating dataset structure...")
    dataset_dir, train_dir, val_dir = create_dataset_structure()
    
    print("Defining classes...")
    classes = create_class_definitions()
    
    print("Creating class mapping...")
    class_mapping = create_class_mapping(classes)
    
    # Save class mapping
    with open(dataset_dir / 'class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=4)
    
    # Create data.yaml for training
    data_yaml = {
        'path': str(dataset_dir.absolute()),
        'train': str((train_dir / 'images').absolute()),
        'val': str((val_dir / 'images').absolute()),
        'names': class_mapping,
        'nc': len(class_mapping)
    }
    
    with open(dataset_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    
    # Create README with instructions
    readme_content = """# Underwater Object Detection Dataset

## Dataset Structure
- train/
  - images/     # Training images
  - labels/     # Training labels (YOLO format)
- val/
  - images/     # Validation images
  - labels/     # Validation labels (YOLO format)

## Adding Images
1. Place your underwater images in the appropriate directory:
   - Training images in: train/images/
   - Validation images in: val/images/

2. Create corresponding label files in YOLO format:
   - One .txt file per image
   - Same name as the image file
   - Place in train/labels/ or val/labels/
   - Format: <class_id> <x_center> <y_center> <width> <height>
   - All values normalized to [0, 1]

## Class Mapping
The class mapping is stored in class_mapping.json
"""
    
    with open(dataset_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    
    print(f"\nCreated dataset structure with {len(class_mapping)} classes")
    print(f"Total categories:")
    for category, subcats in classes.items():
        subcat_count = sum(len(items) if isinstance(items, list) else len(items.values()) for items in subcats.values())
        print(f"- {category}: {subcat_count} types")
    
    return class_mapping

if __name__ == '__main__':
    print("Creating dataset...")
    class_mapping = create_dataset()
    
    print("\nDataset preparation complete!")
    print(f"Total number of classes: {len(class_mapping)}")
    print("\nClass mapping saved to datasets/underwater_objects/class_mapping.json")
    print("Dataset configuration saved to datasets/underwater_objects/data.yaml")
    print("\nNext steps:")
    print("1. Add your underwater images to datasets/underwater_objects/train/images/")
    print("2. Create corresponding label files in datasets/underwater_objects/train/labels/")
    print("3. Run the training script: python src/training/train_underwater_model.py")
