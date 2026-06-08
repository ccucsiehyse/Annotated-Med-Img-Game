import kagglehub

# Download latest version
path = kagglehub.dataset_download("vbookshelf/computed-tomography-ct-images")

print("Path to dataset files:", path)