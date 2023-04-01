from PIL import Image
from torchvision import transforms
import torch

# Paths to content and style images
image_path = "/home/dakire/Desktop/image.jpeg"
style_path = "/home/dakire/Desktop/style.jpeg"

# Check if CUDA is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load content and style images
content_image = Image.open(image_path).convert('RGB')
style_image = Image.open(style_path).convert('RGB')

# Define the transforms for the images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Apply the transforms to the images
content_tensor = transform(content_image)
style_tensor = transform(style_image)

# Add batch dimension to tensors
content_tensor = content_tensor.unsqueeze(0)
style_tensor = style_tensor.unsqueeze(0)

# Load the model
vgg = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=True)

# Define the style layer
style_layer = 'relu2_2'

# Extract features from style image
style_features = vgg(style_tensor.to(device))
style_gram = [torch.mm(feature, feature.t()) for feature in style_features[style_layer]]

# Extract features from content image
content_features = vgg(content_tensor.to(device))
content_gram = [torch.mm(feature, feature.t()) for feature in content_features[style_layer]]

# Initialize the output image with random noise
output_tensor = torch.randn(content_tensor.shape, device=device, requires_grad=True)

# Define optimizer
optimizer = torch.optim.Adam([output_tensor], lr=0.01)

# Set number of iterations and lambda values
num_iterations = 2000
content_weight = 1
style_weight = 1e6

# Run style transfer
for i in range(num_iterations):
    # Zero out gradients
    optimizer.zero_grad()

    # Extract features from output image
    output_features = vgg(output_tensor)
    output_gram = [torch.mm(feature, feature.t()) for feature in output_features[style_layer]]

    # Calculate content loss
    content_loss = sum([torch.nn.functional.mse_loss(output_features[layer], content_features[layer]) for layer in content_features])

    # Calculate style loss
    style_loss = sum([torch.nn.functional.mse_loss(output_gram[layer], style_gram[layer]) for layer in range(len(style_gram))])

    # Calculate total loss
    total_loss = content_weight * content_loss + style_weight * style_loss

    # Backpropagate and update
    total_loss.backward()
    optimizer.step()

    # Print progress
    if i % 100 == 0:
        print(f"Iteration {i}: total loss = {total_loss.item()}")

# Remove batch dimension from tensor
output_tensor = output_tensor.squeeze(0)

# Convert tensor to image
output_image = transforms.ToPILImage()(output_tensor.cpu())

# Save output image
output_image.save('output.jpg')
