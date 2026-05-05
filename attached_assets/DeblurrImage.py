import cv2
import os
import matplotlib.pyplot as plt

def deblur_image(blurred_path):
    # Load blurred overlay image
    blurred = cv2.imread(blurred_path)
    if blurred is None:
        raise ValueError("Blurred image not found.")

    # Derive original image path from blurred filename
    folder = os.path.dirname(blurred_path)
    blurred_name = os.path.basename(blurred_path)
    name_part = blurred_name.replace("_blurred", "_original")

    original_path = os.path.join(folder, name_part)

    # Load original image
    original = cv2.imread(original_path)
    if original is None:
        raise FileNotFoundError(f"Original image not found at: {original_path}")

    # Convert to RGB for display
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

    # Deblurred is just original
    deblurred = original_rgb

    # Show all three
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    titles = ['Original', 'Blurred (Overlay)', 'Deblurred (Uncovered)']
    images = [original_rgb, blurred_rgb, deblurred]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    deblur_image(
        blurred_path=r"E:\4th Sem\Road Safty Management\Trained Model YOLO\blurr_images\24_blurred.jpg"  )
