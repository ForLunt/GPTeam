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

# Define the prompt for the inpainting
prompt = "Face of a yellow cat, high resolution, sitting on a park bench"

# Apply the inpainting to the image using the pipeline
output = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]

# Save the output image
output.save("./yellow_cat_on_park_bench.png")


# from diffusers import StableDiffusionInpaintPipeline
# import torch
# from PIL import Image
# import numpy as np
#
# image_Path = "/home/dakire/Desktop/image.png"
# mask_image_Path = "/home/dakire/Desktop/mask.png"
#
# device = "cuda" if torch.cuda.is_available() else "cpu"  # check if GPU is available, otherwise use CPU
# pipe = StableDiffusionInpaintPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2-inpainting",
#     torch_dtype=torch.float32,
#     device=device,  # set the device to use
# )
#
# prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
# # image and mask_image should be PIL images.
# # The mask structure is white for inpainting and black for keeping as is
# image = Image.open(image_Path)
# mask_image = Image.open(mask_image_Path)
#
# # convert PIL images to PyTorch tensors and move them to the device
# image = torch.tensor(torch.FloatTensor(np.array(image)).permute(2, 0, 1)).to(device)
# mask_image = torch.tensor(torch.FloatTensor(np.array(mask_image)).unsqueeze(0)).to(device)
#
# output = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
#
# # move the output tensor back to the CPU and convert it to a PIL image
# output = output.to("cpu").squeeze(0).permute(1, 2, 0).numpy().clip(0, 1) * 255
# output = Image.fromarray(output.astype(np.uint8))
#
# output.save("./yellow_cat_on_park_bench.png")