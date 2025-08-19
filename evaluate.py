import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse

# Same labels as AffectNet
labels = ["Anger","Contempt","Fear","Sadness","Happiness","Neutral","Surprise","Disgust"]

class EmotionResNet(nn.Module):
    def __init__(self, num_classes=8):
        super(EmotionResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

def predict(image_path, model_path="emotion_resnet50.pth"):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionResNet(num_classes=8).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load image
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        _, predicted = outputs.max(1)
        print(f"Predicted emotion: {labels[predicted.item()]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    args = parser.parse_args()

    predict(args.image)
