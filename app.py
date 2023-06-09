import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import torch
import pickle
import json

# getting the json data containing info about classes
# Read the JSON data from the file
file_path = "info.json"
with open(file_path, "r") as file:
    json_data = json.load(file)


# function to search by id
def search_by_id(id_value, data):
    for item in data:
        if item["id"] == id_value:
            return item
    return None


# setting device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# instantize the model
plant_model = torch.load('trained_models/model_full.pth',  map_location=device)
plant_model.to(device)
# Define the transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the class name
with open('classes.pkl', 'rb') as f:
    loaded_classes = pickle.load(f)

st.title("Plant Disease Detector")
upload= st.file_uploader("Choose an image..", type=["jpg", "png", "jpeg", "webp"])

if upload is not None:
    img = Image.open(upload)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t,0)
    batch_t = batch_t.to(device)
    # get the model prediction
    with torch.inference_mode():
        out = plant_model(batch_t)

    _, predicted = torch.max(out.data, 1)
    ## the infromation related to the predicted class is retrieved
    predicted_class = search_by_id(predicted.item(), json_data)
    st.write("Predicted Class: ",predicted_class)