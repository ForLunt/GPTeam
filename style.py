import PIL
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub

def load_image(image_path):
    base=tf.io.read_file(image_path)
    img = tf.io.decode_image(
        base,
        channels=3, dtype=tf.float32)[tf.newaxis, ...]
    image_size=(base.shape[1],base.shape[0])
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img

def export_image(tf_img):
    tf_img = tf_img * 255
    tf_img = np.array(tf_img, dtype=np.uint8)
    if np.ndim(tf_img) > 3:
        assert tf_img.shape[0] == 1
        img = tf_img[0]
    return PIL.Image.fromarray(img)

def generate_img(original_img,style_img,nom):

    # Enable GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        except RuntimeError as e:
            print(e)

    # Brute force pour trouver des valeurs de ksize optimale 
    style_img = tf.nn.avg_pool(style_img, ksize=[3, 3], strides=[1, 1], padding='VALID')

    stylize_model = tf_hub.load("magenta")

    results = stylize_model(tf.constant(original_img), tf.constant(style_img))
    stylized_photo = results[0]
    #stylized_photo = resize(original_img.size)
    export_image(stylized_photo).save("output/"+nom+".png")

