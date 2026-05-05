import cv2
import os

def blur_image(image_path, output_dir=None):
    """
    Apply a blur overlay to an image and save both original and blurred versions.
    This is a visual trick for demonstration purposes.
    
    Args:
        image_path: Path to the original image
        output_dir: Directory to save the blurred and original images (defaults to same directory)
    
    Returns:
        Tuple of (original_path, blurred_path)
    """
    # Load original image
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Original image not found at: {image_path}")

    # Create blur overlay (Gaussian blur)
    blur = cv2.GaussianBlur(original, (21, 21), 0)

    # Create overlay by combining original + blur with alpha
    alpha = 0.7
    overlay = cv2.addWeighted(original, 1 - alpha, blur, alpha, 0)

    # Get filenames
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    
    # Use same directory if output_dir is not specified
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save original and blurred overlay
    original_save_path = os.path.join(output_dir, f"{name}_original{ext}")
    blurred_save_path = os.path.join(output_dir, f"{name}_blurred{ext}")

    cv2.imwrite(original_save_path, original)
    cv2.imwrite(blurred_save_path, overlay)

    print(f"Saved original to: {original_save_path}")
    print(f"Saved blurred overlay to: {blurred_save_path}")
    
    return original_save_path, blurred_save_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply a blur overlay to an image")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output-dir", help="Directory to save output images")
    
    args = parser.parse_args()
    
    blur_image(args.input, args.output_dir)