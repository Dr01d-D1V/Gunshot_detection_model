import kagglehub
import os
import shutil

# Create documents folder if it doesn't exist
data_path = "/home/div/vsCode/Audio_classifier_Model/data"
os.makedirs(data_path, exist_ok=True)

# Move the UrbanSound8K dataset from the Kaggle cache into the project tree
dataset_handle = "chrisfilo/urbansound8k"
dataset_name = dataset_handle.split("/")[-1]
destination_path = os.path.join(data_path, dataset_name)
final_path = destination_path

if os.path.exists(destination_path):
    print(f"Dataset already exists in documents folder: {destination_path}")
    final_path = destination_path
else:
    cache_path = kagglehub.dataset_download(dataset_handle)
# cache_path = "/home/div/.cache/kagglehub/datasets/chrisfilo/urbansound8k/versions/1"
print("Dataset downloaded to:", cache_path)
print(f"Moving dataset to documents folder: {destination_path}")
shutil.move(cache_path, destination_path)
print("Dataset moved successfully!")
final_path = destination_path

print("Final dataset location:", final_path)
print("Contents of dataset in documents folder:")
for root, dirs, files in os.walk(final_path):
    for file in files[:10]:  # Show first 10 files to avoid too much output
        print(os.path.join(root, file))
    if len(files) > 10:
        print(f"... and {len(files) - 10} more files in this directory")