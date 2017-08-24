import scipy.misc
import numpy as np

def save_images(images, size, path):
    img = (images+1.0) / 2.0
    h,w = img.shape[1], img.shape[2]

    merge_img = np.zeros((h*size[0],w*size[1],3))
    for idx,image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j*h:j*h+h,i*w:i*w+w,:] = image
    return scipy.misc.imsave(path,merge_img)
