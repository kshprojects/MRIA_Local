"""
Background storage utilities for handling image saving without blocking main flow
"""

import os
import threading
import concurrent.futures
import datetime as dt
from typing import List, Dict, Any, Tuple
from rich.console import Console

console = Console()

def create_timestamped_folder(base_path: str = "assets/retrieved_images") -> str:
    """
    Create a timestamped folder for storing images
    
    Args:
        base_path: Base directory path for image storage
        
    Returns:
        Full path to created folder
    """
    current_time = dt.datetime.now()
    folder_name = current_time.strftime("%Y%m%d_%H%M%S")
    full_folder_path = os.path.join(base_path, folder_name)
    
    try:
        os.makedirs(full_folder_path, exist_ok=True)
        console.print(f"[Storage Setup] 📁 Created folder: {full_folder_path}")
        return full_folder_path
    except Exception as e:
        console.print(f"[Storage Setup] ❌ Failed to create directory {full_folder_path}: {e}")
        return None

def save_single_image(image_info: Tuple[bytes, str, str, float]) -> bool:
    """
    Save a single image to local storage
    
    Args:
        image_info: Tuple of (image_data, gcs_uri, folder_path, score)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        image_data, gcs_uri, folder_path, score = image_info
        
        # Extract filename from GCS URI
        if "/" in gcs_uri:
            original_filename = gcs_uri.split("/")[-1]
            # Ensure it has proper extension
            if not original_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                original_filename += '.jpg'
        else:
            # Create generic filename with timestamp
            timestamp = dt.datetime.now().strftime("%H%M%S_%f")[:12]  # Include microseconds
            original_filename = f"image_{timestamp}.jpg"
        
        local_image_path = os.path.join(folder_path, original_filename)
        
        # Write image data to file
        with open(local_image_path, 'wb') as f:
            f.write(image_data)
        
        console.print(f"[Background Storage] ✅ Saved: {local_image_path} (score: {score:.2f})")
        return True
        
    except Exception as e:
        console.print(f"[Background Storage] ❌ Failed to save image: {e}")
        return False

def save_images_batch_background(images_data: List[Tuple[bytes, str, str, float]]) -> None:
    """
    Save multiple images concurrently in background
    
    Args:
        images_data: List of tuples (image_data, gcs_uri, folder_path, score)
    """
    if not images_data:
        console.print("[Background Storage] No images to save")
        return
    
    successful_saves = 0
    
    try:
        # Use ThreadPoolExecutor for concurrent file I/O
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all save tasks
            future_to_image = {executor.submit(save_single_image, img_info): img_info 
                              for img_info in images_data}
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_image):
                if future.result():
                    successful_saves += 1
        
        console.print(f"[Background Storage] 🎯 Completed: {successful_saves}/{len(images_data)} images saved")
        
    except Exception as e:
        console.print(f"[Background Storage] ❌ Batch save error: {e}")

def start_background_image_storage(images_data: List[Tuple[bytes, str, str, float]]) -> threading.Thread:
    """
    Start background image storage in a separate thread
    
    Args:
        images_data: List of tuples (image_data, gcs_uri, folder_path, score)
        
    Returns:
        The background thread (for monitoring if needed)
    """
    def run_background_storage():
        try:
            save_images_batch_background(images_data)
        except Exception as e:
            console.print(f"[Background Storage] ❌ Thread error: {e}")
    
    # Start daemon thread (dies when main program exits)
    storage_thread = threading.Thread(
        target=run_background_storage,
        daemon=True,
        name="ImageStorageThread"
    )
    storage_thread.start()
    
    console.print(f"[Background Storage] 🚀 Started saving {len(images_data)} images in background")
    return storage_thread

def prepare_images_for_background_storage(image_downloads: List[Dict[str, Any]], 
                                        folder_path: str) -> List[Tuple[bytes, str, str, float]]:
    """
    Prepare image data for background storage
    
    Args:
        image_downloads: List of dicts with 'image_data', 'gcs_uri', 'score'
        folder_path: Destination folder path
        
    Returns:
        List of tuples ready for background storage
    """
    storage_data = []
    
    for download in image_downloads:
        if download.get('image_data') and isinstance(download['image_data'], bytes):
            storage_data.append((
                download['image_data'],
                download['gcs_uri'],
                folder_path,
                download.get('score', 0.0)
            ))
    
    return storage_data


# Example usage functions for different scenarios
def quick_background_save(image_downloads: List[Dict[str, Any]], 
                         base_path: str = "assets/retrieved_images") -> threading.Thread:
    """
    Convenience function for quick background saving
    
    Args:
        image_downloads: List of image download results
        base_path: Base storage path
        
    Returns:
        Background thread
    """
    # Create folder
    folder_path = create_timestamped_folder(base_path)
    if not folder_path:
        console.print("[Quick Save] ❌ Failed to create storage folder")
        return None
    
    # Prepare data
    storage_data = prepare_images_for_background_storage(image_downloads, folder_path)
    
    # Start background storage
    return start_background_image_storage(storage_data)