import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

image_path = "/home/dakire/Desktop/image.png"
mask_image_path = "/home/dakire/Desktop/mask.png"

# Load the image and mask as PIL images
image = Image.open(image_path)
mask_image = Image.open(mask_image_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("torch.cuda.current_device()", torch.cuda.current_device())

# Initialize the pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float32,
)
pipe.to(device)

# Define the prompt for the inpainting
prompt = "Face of a yellow cat, high resolution, sitting on a park bench"

# Apply the inpainting to the image using the pipeline
output = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]

# Save the output image
output.save("./yellow_cat_on_park_bench.png")
