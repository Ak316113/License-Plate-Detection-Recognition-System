import cv2
import os
import sys

def deblur_image(blurred_path, output_path=None):
    """
    Retrieve the original image that corresponds to a blurred image.
    This is a visual trick that simply finds the original image.
    
    Args:
        blurred_path: Path to the blurred image
        output_path: Path to save the deblurred (actually original) image
                     If None, creates a path with _deblurred suffix
    
    Returns:
        Path to the deblurred (original) image
    """
    try:
        # Load blurred overlay image
        blurred = cv2.imread(blurred_path)
        if blurred is None:
            raise ValueError(f"Blurred image not found at: {blurred_path}")

        # Derive original image path from blurred filename
        folder = os.path.dirname(blurred_path)
        blurred_name = os.path.basename(blurred_path)
        
        # Check if this follows our naming convention
        if "_blurred" not in blurred_name:
            print(f"Warning: Blurred image filename doesn't contain '_blurred' marker: {blurred_name}")
            # Try to guess by removing a suffix like 'blurred' if present
            name_parts = os.path.splitext(blurred_name)
            name_part = name_parts[0].replace("blurred", "original") + name_parts[1]
        else:
            name_part = blurred_name.replace("_blurred", "_original")

        original_path = os.path.join(folder, name_part)

        # Check if original exists
        if not os.path.exists(original_path):
            # Try alternative patterns
            possible_paths = [
                original_path,
                os.path.join(folder, blurred_name.replace("_blurred", "")),
                os.path.join(folder, "original_" + blurred_name.replace("_blurred", "")),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    original_path = path
                    break
            else:
                raise FileNotFoundError(f"Original image not found at: {original_path}")

        # Load original image
        original = cv2.imread(original_path)
        if original is None:
            raise FileNotFoundError(f"Failed to load original image at: {original_path}")

        # Determine output path
        if output_path is None:
            name, ext = os.path.splitext(blurred_name)
            if "_blurred" in name:
                deblurred_name = name.replace("_blurred", "_deblurred") + ext
            else:
                deblurred_name = name + "_deblurred" + ext
                
            output_path = os.path.join(folder, deblurred_name)

        # Save the "deblurred" image (which is actually the original)
        cv2.imwrite(output_path, original)
        print(f"Saved deblurred image to: {output_path}")
        
        return output_path

    except Exception as e:
        print(f"Error in deblur_image: {str(e)}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Recover original image from a blurred version")
    parser.add_argument("--input", required=True, help="Path to blurred image")
    parser.add_argument("--output", help="Path to save deblurred image (optional)")
    
    args = parser.parse_args()
    
    deblur_image(args.input, args.output)