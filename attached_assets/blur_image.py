import cv2
import os

def blur_image(image_path, output_dir):
    # Load original image
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError("Original image not found.")

    # Create blur overlay (Gaussian blur)
    blur = cv2.GaussianBlur(original, (21, 21), 0)

    # Create overlay by combining original + blur with alpha
    alpha = 0.7
    overlay = cv2.addWeighted(original, 1 - alpha, blur, alpha, 0)

    # Get filenames
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)

    # Save original and blurred overlay
    original_save_path = os.path.join(output_dir, f"{name}_original{ext}")
    blurred_save_path = os.path.join(output_dir, f"{name}_blurred{ext}")

    cv2.imwrite(original_save_path, original)
    cv2.imwrite(blurred_save_path, overlay)

    print(f"Saved original to: {original_save_path}")
    print(f"Saved blurred overlay to: {blurred_save_path}")

if __name__ == "__main__":
    blur_image(
        image_path=r"E:\4th Sem\Road Safty Management\test\24.jpg",
        output_dir=r"E:\4th Sem\Road Safty Management\Trained Model YOLO\blurr_images"
    )
