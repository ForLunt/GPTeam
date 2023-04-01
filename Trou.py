import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


def fill_mask(image_path, mask_image_path):

    image_path = "/home/dakire/Desktop/image.png"
    mask_image_path = "/home/dakire/Desktop/mask.png"

    # Load the image and mask as PIL images
    image = Image.open(image_path)
    mask_image = Image.open(mask_image_path)

    old_dimensions_image = image.size
    old_dimensions_mask = mask_image.size
    image = image.resize((512, 512))
    mask_image = mask_image.resize((512, 512))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
    )
    pipe.to(device)

    # Define the prompt for the inpainting
    prompt = "motor racing circuit"

    # Apply the inpainting to the image using the pipeline
    output = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
    # Save the output image
    output.resize(old_dimensions_image).save("./result.png")

    torch.cuda.empty_cache()

