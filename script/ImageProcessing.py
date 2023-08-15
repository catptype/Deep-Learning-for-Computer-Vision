import tensorflow as tf

def image_resize_square(image_path, size, aspect_ratio=True, padding=False):

    # read image from file
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, expand_animations = False) # set expand_animations = False if no GIF image
    
    # pre processing
    image = tf.image.resize(image,
                            (size,size),
                            #method='bicubic', # Default method='bilinear', 
                            preserve_aspect_ratio=aspect_ratio,
                            antialias=True)
    if padding:
        image = tf.image.resize_with_pad(image,size,size)
    
    image = image / 255.0

    return image