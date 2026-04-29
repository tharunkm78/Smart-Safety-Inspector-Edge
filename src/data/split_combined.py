import os
import random
import shutil
import yaml
from pathlib import Path

random.seed(42)

def main():
    print("=== Splitting Combined Dataset ===")
    
    data_dir = Path(r"c:\Users\tk896\OneDrive\Desktop\Prac_AI\data\combined")
    
    img_train_dir = data_dir / "images" / "train"
    lbl_train_dir = data_dir / "labels" / "train"
    
    img_val_dir = data_dir / "images" / "val"
    lbl_val_dir = data_dir / "labels" / "val"
    
    img_test_dir = data_dir / "images" / "test"
    lbl_test_dir = data_dir / "labels" / "test"
    
    # Create output directories
    img_val_dir.mkdir(parents=True, exist_ok=True)
    lbl_val_dir.mkdir(parents=True, exist_ok=True)
    img_test_dir.mkdir(parents=True, exist_ok=True)
    lbl_test_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all current training images
    print("Gathering files...")
    images = list(img_train_dir.glob("*.jpg"))
    
    # Check if already split
    if len(images) < 10000:
        print("Warning: Seems like dataset is already split or missing!")
        print(f"Found only {len(images)} images in train.")
    
    # Shuffle for random split
    random.shuffle(images)
    
    total = len(images)
    print(f"Total images found: {total}")
    
    val_count = int(total * 0.10)
    test_count = int(total * 0.10)
    train_count = total - val_count - test_count
    
    print(f"Target splits -> Train: {train_count}, Val: {val_count}, Test: {test_count}")
    
    val_images = images[:val_count]
    test_images = images[val_count:val_count+test_count]
    # The rest stay in train
    
    def move_files(files, target_img_dir, target_lbl_dir):
        count = 0
        for img_path in files:
            lbl_path = lbl_train_dir / (img_path.stem + ".txt")
            
            # Move image
            shutil.move(str(img_path), str(target_img_dir / img_path.name))
            
            # Move label if it exists
            if lbl_path.exists():
                shutil.move(str(lbl_path), str(target_lbl_dir / lbl_path.name))
            count += 1
        return count
    
    print("\nMoving files to Validation split...")
    m_val = move_files(val_images, img_val_dir, lbl_val_dir)
    print(f"Moved {m_val} files to val.")
    
    print("\nMoving files to Test split...")
    m_test = move_files(test_images, img_test_dir, lbl_test_dir)
    print(f"Moved {m_test} files to test.")
    
    print("\nUpdating dataset.yaml...")
    yaml_path = data_dir / "dataset.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            yaml_content = yaml.safe_load(f)
            
        yaml_content["train"] = "images/train"
        yaml_content["val"] = "images/val"
        yaml_content["test"] = "images/test"
        
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False)
            
        print("dataset.yaml updated successfully.")
    else:
        print("Warning: dataset.yaml not found!")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
