import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import torch
import pickle
import json
from models import PlantDiseaseDetector
# getting the json data containing info about classes
# Read the JSON data from the file
file_path = "info.json"
with open(file_path, "r") as file:
    json_data = json.load(file)




# function to search by id
def search_by_id(id_value, data):
    """
    Function to give information regarding plant disease by id
    id_value: id given by the model
    data: josn data containting information
    """
    for item in data:
        if item["id"] == id_value:
            return item
    return None


# setting device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# instantiate the model
# 38 is used as there are 38 classes and model was trained on 38 classes
plant_model = PlantDiseaseDetector(38)
plant_model.load_state_dict(torch.load('trained_models/plant_model_v3.pth'))
plant_model.to(device)

# Creating image transformers similar to one used in training
transform = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


st.set_page_config(page_title="Plant Disease Vision",page_icon="🌱")

# Title of the app
st.title("PlantDiseaseVision")

# uploading the image we want to predict on
upload = st.file_uploader("Choose an image of plant disease..", type=["jpg", "png", "jpeg", "webp"])

predicted_class = None
if upload is not None:
    img = Image.open(upload)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t,0)
    batch_t = batch_t.to(device)
    # get the model prediction
    plant_model.eval()
    with torch.inference_mode():
        out = plant_model(batch_t)

    _, predicted = torch.max(out.data, 1)
    ## the infromation related to the predicted class is retrieved
    predicted_class = search_by_id(predicted.item(), json_data)
if predicted_class:
    st.header("Predicted : "+predicted_class["name"])
    st.divider()
    st.header("Solutions: ")
    for item in predicted_class["solutions"]:
        st.markdown(f"* {item.capitalize()}")
    # st.write(predicted_class["solutions"])

st.divider()
st.markdown("""
Made with :heart: by Niranjan Shah \n
Repo: https://github.com/niranjanblank/PlantDiseaseVision
""")
