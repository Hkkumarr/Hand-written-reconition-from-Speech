import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64*7*7,128),
            nn.ReLU(),
            nn.Linear(128,26)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        return self.fc(x)

# Load Model
@st.cache_resource
def load_model():
    model = CNN().to(device)

    model.load_state_dict(
        torch.load("model.pth", map_location=device)
    )

    model.eval()
    return model

model = load_model()

# Transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# UI
st.title("✍ Handwritten Character Recognition")

st.write("Upload an image containing a handwritten letter.")

uploaded_file = st.file_uploader(
    "Choose Image",
    type=["png","jpg","jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=250
    )

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        pred = output.argmax(1).item()

    st.success(
        f"Predicted Character : {letters[pred]}"
    )
    