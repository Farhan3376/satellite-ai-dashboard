import os
import cv2
import csv
import random

def get_class_mapping():
    """
    Returns a dictionary mapping the original EuroSAT classes to the 4 target classes:
    Forest, Water, Urban, and Agriculture.
    """
    return {
        "AnnualCrop": "Agriculture",
        "HerbaceousVegetation": "Agriculture",
        "Pasture": "Agriculture",
        "PermanentCrop": "Agriculture",
        "Forest": "Forest",
        "River": "Water",
        "SeaLake": "Water",
        "Highway": "Urban",
        "Industrial": "Urban",
        "Residential": "Urban"
    }

def collect_image_paths_and_labels(input_dir, class_mapping):
    """
    Scans the input directory for all images and maps their original classes to the target classes.
    
    Args:
        input_dir (str): Path to the original dataset directory containing class subdirectories.
        class_mapping (dict): Mapping from original class to target class.
        
    Returns:
        list: Tuples of (image_path, original_class, target_class)
    """
    data_records = []
    
    # Iterate over original class folders in the dataset directory
    for original_class in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, original_class)
        # Skip if not a directory or if not in mapping (e.g., label_map.json, CSVs)
        if not os.path.isdir(class_dir) or original_class not in class_mapping:
            continue
            
        target_class = class_mapping[original_class]
        
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, filename)
                data_records.append({
                    'original_path': img_path,
                    'original_class': original_class,
                    'target_class': target_class
                })
                
    return data_records

def preprocess_and_save_images(data_records, output_base_dir, split_name, target_size=(64, 64)):
    """
    Reads images, resizes them, and saves them to the structured output directory.
    Updates the records with the new paths.
    
    Args:
        data_records (list): List of dictionaries containing image information.
        output_base_dir (str): Base directory for the processed dataset.
        split_name (str): The data split name (e.g., 'train', 'val', 'test').
        target_size (tuple): Target (width, height) for resizing.
        
    Returns:
        list: Updated data records containing the new processed paths.
    """
    processed_records = []
    
    for record in data_records:
        original_path = record['original_path']
        target_class = record['target_class']
        
        # Create output directory structure: output_base_dir/split_name/target_class/
        output_dir = os.path.join(output_base_dir, split_name, target_class)
        os.makedirs(output_dir, exist_ok=True)
        
        # Define new filename and path structure for clarity
        filename = os.path.basename(original_path)
        # Prefix the filename with its original class to prevent potential name collisions
        new_filename = f"{record['original_class']}_{filename}"
        new_path = os.path.join(output_dir, new_filename)
        
        # Read, resize, and save image
        img = cv2.imread(original_path)
        if img is not None:
            resized_img = cv2.resize(img, target_size)
            cv2.imwrite(new_path, resized_img)
            
            # Update record for the mapping file
            record_copy = record.copy()
            record_copy['processed_path'] = new_path
            record_copy['split'] = split_name
            processed_records.append(record_copy)
            
    return processed_records

def prepare_dataset(input_dir, output_dir, mapping_csv_path, target_size=(64, 64)):
    """
    Main function to orchestrate the dataset loading, preprocessing, splitting, and saving.
    
    Args:
        input_dir (str): Path to original dataset.
        output_dir (str): Path to processed structured dataset.
        mapping_csv_path (str): Path for the output mapping CSV.
        target_size (tuple): Width and height to resize images.
    """
    # 1. Gather configured class mappings
    class_mapping = get_class_mapping()
    
    # 2. Collect files
    data_records = collect_image_paths_and_labels(input_dir, class_mapping)
    
    if not data_records:
        print("No images found to process.")
        return
        
    # 3. Stratified split (70% train, 20% test, 10% validation)
    class_groups = {}
    for record in data_records:
        t_class = record['target_class']
        if t_class not in class_groups:
            class_groups[t_class] = []
        class_groups[t_class].append(record)
        
    train_records = []
    val_records = []
    test_records = []
    
    # Use fixed seed for reproducibility
    random.seed(42)
    
    for t_class, records in class_groups.items():
        random.shuffle(records)
        n = len(records)
        train_end = int(n * 0.7)
        val_end = train_end + int(n * 0.1)
        
        train_records.extend(records[:train_end])
        val_records.extend(records[train_end:val_end])
        test_records.extend(records[val_end:])
    
    # 4. Preprocess and save for each split
    all_processed_records = []
    all_processed_records.extend(preprocess_and_save_images(train_records, output_dir, 'train', target_size))
    all_processed_records.extend(preprocess_and_save_images(val_records, output_dir, 'val', target_size))
    all_processed_records.extend(preprocess_and_save_images(test_records, output_dir, 'test', target_size))
    
    # 5. Save the final mapping tracking original and processed paths, alongside classes
    if all_processed_records:
        keys = all_processed_records[0].keys()
        with open(mapping_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_processed_records)

if __name__ == "__main__":
    # Define paths
    # Setting INPUT path to EuroSAT base folder containing the images
    INPUT_DATA_DIR = "/home/romi/farhan/Research paper/EuroSat/EuroSAT"
    
    # Settings derived from current Project workspace structure
    OUTPUT_DATA_DIR = "/home/romi/farhan/ML project/processed_dataset"
    MAPPING_CSV_FILE = "/home/romi/farhan/ML project/dataset_mapping.csv"
    
    # Prepare and process the dataset
    prepare_dataset(INPUT_DATA_DIR, OUTPUT_DATA_DIR, MAPPING_CSV_FILE)
