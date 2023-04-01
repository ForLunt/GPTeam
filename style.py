from PIL import Image, ImageFilter, ImageEnhance
from Exodus.Styling.Filters import OilPainting

# Charger l'image
image = Image.open("62a4d4ce8aece.jpg")
# Appliquer un flou gaussien
image = image.filter(ImageFilter.GaussianBlur(radius=2))

# Accentuer les bords
image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

# Adoucir les zones à texture fine
image = image.filter(ImageFilter.MedianFilter(size=5))

# Réduire le nombre de couleurs
image = image.quantize(colors=32)

# Convertir l'image au mode RGB
image = image.convert("RGB")

# Enregistrer l'image au format JPEG
image.save("image_aquarelle.jpg", "JPEG")

image2 = Image.open("62a4d4ce8aece.jpg")
# Appliquer un effet de peinture à l'huile
image2 = OilPainting(image, size=6, dyn_ratio=2)

# Convertir l'image au mode RGB
image2 = image2.convert("RGB")

# Enregistrer l'image au format JPEG
image2.save("image_peinture_huile.jpg", "JPEG")