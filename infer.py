# create a file can inference the model
import pytorch_lightning as lt
import torch
from PIL import Image

from model.lit_model import LitModel

model_path = "checkpoints/convnext_v2_tiny/convnext_v2_tiny-drop0.5-epoch=09-val/acc=0.99.ckpt"
image_path = "cats_dogs_light/test/cat.983.jpg"

model = LitModel()
model.load_from_checkpoint(model_path)
model.eval()

transform = lt.transforms.Compose([
    lt.transforms.Resize((224, 224)),
    lt.transforms.ToTensor(),
])

image = Image.open(image_path)
image = transform(image)
image = image.unsqueeze(0)

class_names = {1: "cat", 0: "dog"}

with torch.no_grad():
    pred = model(image)

# print out the result
print(f"Predicted class is: {class_names[pred.argmax(dim=1).item()]}")
