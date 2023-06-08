import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import torch
from models import PlantDiseaseDetector

# instantize the model
plant_model = PlantDiseaseDetector(num_classes=38)
plant_model.load_state_dict(torch.load('trained_models/plant_model_v1.pth'))
plant_model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

st.title("Plant Disease Detector")
upload= st.file_uploader("Choose an image..", type=["jpg", "png", "jpeg", "webp"])

if upload is not None:
    img = Image.open(upload)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t,0)

    # get the model prediction
    out = plant_model(batch_t)

    _, predicted = torch.max(out.data, 1)

    st.write("Predicted Class: ", predicted.item())