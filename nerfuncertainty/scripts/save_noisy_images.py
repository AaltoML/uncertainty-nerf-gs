import cv2
import numpy as np
import os
import argparse

np.random.seed(42)

def add_gaussian_noise(image, mean=0, std_dev=0.1):
    """
    Add Gaussian noise to an image after normalizing it to [0, 1].
    
    :param image: Input image
    :param mean: Mean of the Gaussian noise
    :param std_dev: Standard deviation of the Gaussian noise (as a continuous value)
    :return: Image with Gaussian noise added
    """
    # Normalize image to [0, 1]
    normalized_image = image / 255.0
    
    # Generate Gaussian noise
    gaussian_noise = np.random.normal(mean, std_dev, normalized_image.shape)
    
    # Add Gaussian noise and clip to keep the values in the range [0, 1]
    noisy_image = np.clip(normalized_image + gaussian_noise, 0, 1)
    
    # Rescale image back to [0, 255]
    noisy_image = (noisy_image * 255).astype('uint8')
    
    return noisy_image

def add_gaussian_blur(image, kernel_size=5):
    """
    Apply Gaussian blur to an image.
    
    :param image: Input image
    :param kernel_size: Size of the Gaussian kernel (must be an odd number)
    :return: Blurred image
    """
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image

def process_images(input_folder, output_folder, operation, mean=0, std_dev=0.1, kernel_size=5):
    """
    Process all JPG images in the input folder by applying Gaussian noise or Gaussian blur,
    and save them to the output folder.
    
    :param input_folder: Path to the input folder containing JPG files
    :param output_folder: Path to the output folder where processed images will be saved
    :param operation: The operation to perform ('noise' or 'blur')
    :param mean: Mean of the Gaussian noise (only used if operation is 'noise')
    :param std_dev: Standard deviation of the Gaussian noise as a continuous value (only used if operation is 'noise')
    :param kernel_size: Size of the Gaussian kernel for blurring (only used if operation is 'blur')
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    image_filenames = os.listdir(args.input_folder)
    image_filenames.sort() # sort filenames
    # 
    train_split_fraction = 0.9 # hardcoded to be default in (https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/dataparsers/nerfstudio_dataparser.py)

    # https://github.com/nerfstudio-project/nerfstudio/blob/b70772795080730ec36b0841641993b34807aada/nerfstudio/data/utils/dataparsers_utils.py#L23
    # filter image_filenames and poses based on train/eval split percentage
    num_images = len(image_filenames)
    num_train_images = np.ceil(num_images * train_split_fraction).astype(np.int32)
    num_eval_images = num_images - num_train_images
    i_all = np.arange(num_images)
    i_train = np.linspace(
        0, num_images - 1, num_train_images, dtype=int
    )  # equally spaced training images starting and ending at 0 and num_images-1
    i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
    assert len(i_eval) == num_eval_images

    for idx, filename in enumerate(image_filenames): #os.listdir(input_folder):
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            if idx in i_train:
                if operation == 'noise':
                    processed_img = add_gaussian_noise(img, mean, std_dev)
                elif operation == 'blur':
                    processed_img = add_gaussian_blur(img, kernel_size)
                else:
                    print(f"Unknown operation: {operation}. Skipping {filename}.")
                    continue
            
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, processed_img)
                print(f"Processed {filename} and saved to {output_path}")
            else:
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, img)
                print(f"[eval image] non-processed {filename} and saved to {output_path}")
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Gaussian noise or Gaussian blur to images in a folder.")
    parser.add_argument("input_folder", type=str, help="Path to the input folder containing JPG files")
    parser.add_argument("output_folder", type=str, help="Path to the output folder where processed images will be saved")
    parser.add_argument("operation", type=str, choices=['noise', 'blur'], help="Operation to perform: 'noise' or 'blur'")
    parser.add_argument("--mean", type=float, default=0, help="Mean of the Gaussian noise (used if operation is 'noise')")
    parser.add_argument("--std_dev", type=float, default=0.1, help="Standard deviation of the Gaussian noise as a continuous value (used if operation is 'noise')")
    parser.add_argument("--kernel_size", type=int, default=5, help="Kernel size for Gaussian blur (used if operation is 'blur')")

    args = parser.parse_args()
    
    # Process the images
    process_images(input_folder=args.input_folder, 
                   output_folder=args.output_folder, 
                   operation=args.operation, 
                   mean=args.mean, 
                   std_dev=args.std_dev, 
                   kernel_size=args.kernel_size)