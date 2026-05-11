import os
from pathlib import Path
from ultralytics import YOLO
from config import TRAIN_CONFIG, MODELS_DIR

def main():
    print("=== Starting Smart Safety Inspector Training Pipeline ===")
    
    # Ensure the models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Check if the dataset yaml exists before starting
    dataset_yaml = Path(TRAIN_CONFIG["data"])
    if not dataset_yaml.exists():
        print(f"Error: Dataset configuration file not found at {dataset_yaml}")
        print("Please run 'python src/prepare_data.py' first.")
        return

    print(f"Loading base model: {TRAIN_CONFIG['model']}")
    try:
        model = YOLO(TRAIN_CONFIG["model"])
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    print("\nTraining configuration:")
    for key, value in TRAIN_CONFIG.items():
        print(f"  {key}: {value}")
        
    print("\nInitiating YOLOv8 training...")
    try:
        # Start training using the configuration dict unpacked
        results = model.train(**TRAIN_CONFIG)
        
        print("\n=== Training Completed Successfully ===")
        print(f"Best model weights saved in: {MODELS_DIR}/{TRAIN_CONFIG['name']}/weights/best.pt")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")

if __name__ == "__main__":
    main()
