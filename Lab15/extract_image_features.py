import torch
from PIL import Image
from torch import nn
from torchvision import transforms, models
import pandas as pd
import os


def main():
    # Load the ground truth pickle
    df = pd.read_pickle("ground_truth.pkl")
    images_in_pickle = df["image"].unique()

    # Path to images
    image_folder = "image_captioning_dataset/Images"
    existing_images = set(os.listdir(image_folder))  # all filenames in folder

    # Initialize ResNet50 without downloading weights
    resnet = models.resnet50(weights=None)

    # Load local pretrained weights
    resnet_weights_path = "resnet50-11ad3fa6.pth"
    state_dict = torch.load(resnet_weights_path, map_location="cpu")
    resnet.load_state_dict(state_dict)

    # Remove the last fully connected layer to get feature vectors
    resnet.fc = nn.Identity()
    resnet.eval()

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Extract features
    features_list = []
    for img in images_in_pickle:
        img_clean = str(img).strip()
        if img_clean not in existing_images:
            print(f"{img_clean} NOT FOUND, skipping")
            continue

        img_path = os.path.join(image_folder, img_clean)
        image = Image.open(img_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # add batch dimension

        with torch.no_grad():
            feat = resnet(image).squeeze(0).numpy()  # get 2048-dim feature vector

        features_list.append({"image": img_clean, "features": feat})

    # Save extracted features
    pd.DataFrame(features_list).to_pickle("features_to_rnn.pkl")
    print(f"Saved features_to_rnn.pkl for {len(features_list)} images")


if __name__ == "__main__":
    main()
