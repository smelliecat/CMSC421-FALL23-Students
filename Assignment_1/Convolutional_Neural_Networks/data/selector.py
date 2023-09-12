# #%%
# import os
# from PIL import Image

# #%%
# def load_dataset(directory):
#     images = []
#     labels = []
#     for label_name in os.listdir(directory):
#         label_dir = os.path.join(directory, label_name)
#         print(label_dir)
#         for image_name in os.listdir(label_dir):
#             image_path = os.path.join(label_dir, image_name)
#             image = Image.open(image_path)
#             images.append(image)
#             labels.append(label_name)
#     return images, labels


# #%%
# import random

# def shuffle_data(images, labels):
#     combined = list(zip(images, labels))
#     random.shuffle(combined)
#     images[:], labels[:] = zip(*combined)


# #%%
# # /Users/kwesicobbina/Documents/CMSC421-FALL2023/Assignment_1/part_2/data/CINIC-10

# def subsample_data(images, labels, num_samples=1000):
#     return images[:num_samples], labels[:num_samples]

# #%%
# def save_subsample(images, labels, directory):
#     for img, lbl in zip(images, labels):
#         save_path = os.path.join(directory, lbl)
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
#         img.save(os.path.join(save_path, "some_unique_name.jpg"))

# #%%

# # Load the dataset
# images, labels = load_dataset("/Users/kwesicobbina/Documents/CMSC421-FALL2023/Assignment_1/part_2/data/CINIC-10/train")
# #%%
# # Shuffle the data
# shuffle_data(images, labels)

# # Subsample the data
# sub_images, sub_labels = subsample_data(images, labels, num_samples=5000)

# # Save the subsample
# save_subsample(sub_images, sub_labels, "path/to/save/subsample")

# # Cinic-10
# #     |
# #     |-train
# #         |-class1
# #             |-image1
# #             |-image2
# #             |-image1
# #         |-class2
# #             |-image1
# #             |-image2
# #             |-image1
# #         |-class3
# #             |-image1
# #             |-image2
# #             |-image1
# #     |-test
# #         |-class1
# #             |-image1
# #             |-image2
# #             |-image1
# #         |-class2
# #             |-image1
# #             |-image2
# #             |-image1
# #         |-class3
# #             |-image1
# #             |-image2
# #             |-image1


# #%%

# import os
# import shutil
# import random

# def subsample_images(root_folder, subsample_ratio=0.5):
#     """
#     Subsample images in each class folder under 'train' and 'test'.
    
#     Parameters:
#     - root_folder: The root folder containing 'train' and 'test' subfolders.
#     - subsample_ratio: The ratio of images to keep in each class folder.
#     """
#     for subset in ['train', 'test']:
#         subset_folder = os.path.join(root_folder, subset)
        
#         # List all class folders in the subset folder
#         class_folders = [f for f in os.listdir(subset_folder) if os.path.isdir(os.path.join(subset_folder, f))]
        
#         for class_folder in class_folders:
#             class_path = os.path.join(subset_folder, class_folder)
            
#             # List all image files in the class folder
#             image_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            
#             # Shuffle the image files
#             random.shuffle(image_files)
            
#             # Calculate the number of images to keep
#             num_to_keep = int(len(image_files) * subsample_ratio)
            
#             # Remove the extra images
#             for image_file in image_files[num_to_keep:]:
#                 image_path = os.path.join(class_path, image_file)
#                 os.remove(image_path)

# # Example usage
# root_folder = "/Users/kwesicobbina/Documents/CMSC421-FALL2023/Assignment_1/part_2/data/CINIC-10"  # Replace with the path to your Cinic-10 root folder
# subsample_ratio = 0.5  # Keep 50% of images in each class folder
# subsample_images(root_folder, subsample_ratio)

# # %%

# import os
# import shutil
# import random

# def subsample_images(root_folder, dest_folder, subsample_ratio=0.5):
#     """
#     Subsample images from root_folder and store them in dest_folder.
#     """
#     for subset in ['train', 'test']:
#         subset_folder = os.path.join(root_folder, subset)
#         dest_subset_folder = os.path.join(dest_folder, subset)
        
#         # Create destination subset folder if it doesn't exist
#         os.makedirs(dest_subset_folder, exist_ok=True)
        
#         for class_folder in os.listdir(subset_folder):
#             class_path = os.path.join(subset_folder, class_folder)
#             dest_class_folder = os.path.join(dest_subset_folder, class_folder)
            
#             # Create destination class folder if it doesn't exist
#             os.makedirs(dest_class_folder, exist_ok=True)
            
#             if not os.path.isdir(class_path):
#                 continue
            
#             image_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
#             # Print original size
#             print(f"Original size of {subset}/{class_folder}: {len(image_files)}")
            
#             random.shuffle(image_files)
            
#             num_to_keep = int(len(image_files) * subsample_ratio)
#             # Print sampled size
#             print(f"Sampled size of {subset}/{class_folder}: {num_to_keep}")
            
#             # Copy the subsampled files to the destination folder
#             for image_file in image_files[:num_to_keep]:
#                 src_path = os.path.join(class_path, image_file)
#                 dest_path = os.path.join(dest_class_folder, image_file)
#                 shutil.copy2(src_path, dest_path)

# # Example usage
# root_folder = "/Users/kwesicobbina/Documents/CMSC421-FALL2023/Assignment_1/part_2/data/CINIC-10"  # Replace with the path to your Cinic-10 root folder
# dest_folder = "/Users/kwesicobbina/Documents/CMSC421-FALL2023/Assignment_1/part_2/data/sampled_CINIC_10"  # Replace with the path to your destination folder
# subsample_ratio = 0.01  # Keep 1% of images in each class folder

# subsample_images(root_folder, dest_folder, subsample_ratio)

import os
import shutil
import random
import string

def generate_random_string(length=6):
    return ''.join(random.choice(string.digits) for i in range(length))

def subsample_images(root_folder, dest_folder, train_subsample_ratio=0.5, test_subsample_ratio=0.2):
    """
    Subsample images from root_folder and store them in dest_folder.
    """
    for subset in ['train', 'test']:
        subset_folder = os.path.join(root_folder, subset)
        dest_subset_folder = os.path.join(dest_folder, subset)
        
        # Create destination subset folder if it doesn't exist
        os.makedirs(dest_subset_folder, exist_ok=True)
        
        # Choose the appropriate subsample ratio
        subsample_ratio = train_subsample_ratio if subset == 'train' else test_subsample_ratio
        
        for class_folder in os.listdir(subset_folder):
            class_path = os.path.join(subset_folder, class_folder)
            dest_class_folder = os.path.join(dest_subset_folder, class_folder)
            
            # Create destination class folder if it doesn't exist
            os.makedirs(dest_class_folder, exist_ok=True)
            
            if not os.path.isdir(class_path):
                continue
            
            image_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            
            # Print original size
            print(f"Original size of {subset}/{class_folder}: {len(image_files)}")
            
            random.shuffle(image_files)
            
            num_to_keep = int(len(image_files) * subsample_ratio) + 1
            
            # Print sampled size
            print(f"Sampled size of {subset}/{class_folder}: {num_to_keep}")
            
            # Copy the subsampled files to the destination folder and rename them
            for image_file in image_files[:num_to_keep]:
                src_path = os.path.join(class_path, image_file)
                random_string = generate_random_string()
                new_image_name = f"{subset}_{random_string}.jpg"  # Assuming images are in jpg format
                dest_path = os.path.join(dest_class_folder, new_image_name)
                shutil.copy2(src_path, dest_path)

# Example usage
root_folder = "/Users/kwesicobbina/Documents/CMSC421-FALL2023/Assignment_1/part_2/data/CINIC-10"  # Replace with the path to your Cinic-10 root folder
dest_folder = "/Users/kwesicobbina/Documents/CMSC421-FALL2023/Assignment_1/part_2/data/sampled_CINIC_10"  # Replace with the path to your destination folder
train_subsample_ratio = 0.05556  # Keep 1% of images in each class folder
test_subsample_ratio = 0.0111  # Keep 20% of images in the test folder

subsample_images(root_folder, dest_folder, train_subsample_ratio, test_subsample_ratio)
