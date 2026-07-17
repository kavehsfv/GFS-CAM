import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Step 1: Model and preprocessing
def load_model():
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model.eval()
    return model

def preprocess_image(img_path):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img = Image.open(img_path).convert('RGB')
    return tf(img).unsqueeze(0)

# Step 2: Hook to grab last‐conv features
class FeatureExtractor:
    def __init__(self):
        self.features = None
    def __call__(self, module, inp, out):
        self.features = out

# Step 3: Compute GFS-CAM
def compute_gfs_cam(model, img_tensor, target_layer, eps=1e-4):
    # hook
    extractor = FeatureExtractor()
    handle = target_layer.register_forward_hook(extractor)
    with torch.no_grad():
        logits = model(img_tensor)
    handle.remove()

    # pick class
    cls = logits.argmax(dim=1).item()
    feat = extractor.features[0]          # [C,H,W]
    C, H, W = feat.shape
    X = feat.view(C, -1).clone()          # [C, H*W]

    # GFS selection
    weights = torch.zeros(C)
    rem = list(range(C))
    while rem:
        vars_ = X[rem].var(dim=1)
        mv, idx = vars_.max(0)
        if mv < eps:
            break
        ci = rem[idx]
        weights[ci] = mv
        # orthogonalize remaining
        v = (X[ci] - X[ci].mean()) / (X[ci].std()+1e-8)
        for j in rem:
            if j == ci: continue
            proj = (X[j].dot(v) / v.dot(v)) * v
            X[j] -= proj
        rem.remove(ci)

    # build CAM
    cam = torch.zeros(H, W)
    for i, w in enumerate(weights):
        cam += w * feat[i]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.cpu().numpy(), cls

# Step 4: Simple visualization
def visualize_cam(img_path, cam):
    # load & prep for plotting
    orig = cv2.imread(img_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    orig = cv2.resize(orig, (224, 224))

    # heatmap
    # Resize CAM to match input image size before applying colormap
    cam_resized = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)
    hm = (cam_resized * 255).astype(np.uint8)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * orig + 0.5 * hm).astype(np.uint8)

    # plot side by side
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(orig)
    plt.title('Input')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(overlay)
    plt.title('GFS-CAM')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Main
if __name__ == '__main__':
    # Input image path
    image_path = 'cat.jpg'
    model       = load_model()
    img_tensor  = preprocess_image(image_path)
    target_layer= model.features[30]  # last conv of VGG16

    cam, cls = compute_gfs_cam(model, img_tensor, target_layer)
    visualize_cam(image_path, cam)