# important libraries
import numpy as np
from PIL import Image
import cv2
import math

pixel_list={}
arr=[]

# to get array of pixel values of all pixels
def flatten_image(image_array):
    flat_array = []
    height = image_array.shape[0]
    width = image_array.shape[1]
    for i in range(0, height):
        for j in range(0, width):
            red = image_array[i][j][0]
            green = image_array[i][j][1]
            blue = image_array[i][j][2]
            flat_array.append([i, j, red, green, blue])
    return np.array(flat_array)

# to obtain red channel
def red_channel(image_array):
    flat_im = flatten_image(image_array)
    return flat_im[:, 2]

# to obtain green channel
def green_channel(image_array):
    flat_im = flatten_image(image_array)
    return flat_im[:, 3]

# to obtain blue channel
def blue_channel(image_array):
    flat_im = flatten_image(image_array)
    return flat_im[:, 4]

# to get range of red channel
def red_channel_range(image_array):
    r_channel = red_channel(image_array)
    return np.max(r_channel)-np.min(r_channel)

# to get range of green channel
def green_channel_range(image_array):
    g_channel = green_channel(image_array)
    return np.max(g_channel)-np.min(g_channel)

# to get range of blue channel
def blue_channel_range(image_array):
    b_channel = blue_channel(image_array)
    return np.max(b_channel)-np.min(b_channel)

# to assign median colour of a bucket to each pixel in the bucket
def median_cut(image, flat_im):
    red_average = np.mean(flat_im[:, 2])
    green_average = np.mean(flat_im[:, 3])
    blue_average = np.mean(flat_im[:, 4])

    for pixel_point in flat_im:
        pixel_list[(pixel_point[0],pixel_point[1])]=(red_average,green_average,blue_average)


# Function to get the median index of the array
def getMedianIndex(input_array_flat):
    length = len(input_array_flat)
    return int((length+1)/2)


# starting of algorithum by spliting the array into 2 buckets
def split_into_boxes(image_array, flat_image, num_colors):
    if len(flat_image) == 0:
        return

    if num_colors == 0:
        median_cut(image_array, flat_image)
        return

    red_range = red_channel_range(image_array)
    green_range = green_channel_range(image_array)
    blue_range = blue_channel_range(image_array)

    channel_with_highest_range = 2
    if red_range >= green_range and red_range >= blue_range:
        channel_with_highest_range = 2
    elif green_range >= red_range and green_range >= blue_range:
        channel_with_highest_range = 3
    elif blue_range >= green_range and blue_range >= red_range:
        channel_with_highest_range = 4

    flat_image = flat_image[flat_image[:,
                                       channel_with_highest_range].argsort()]
    median_index = getMedianIndex(flat_image)

    split_into_boxes(image_array, flat_image[0:median_index], num_colors-1)
    split_into_boxes(image_array, flat_image[median_index:], num_colors-1)



# to load image
im = cv2.imread("Lena.png")

image_array = np.copy(im)
image_array=cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)


im2 = np.copy(image_array)
slim_image_array = flatten_image(image_array)

split_into_boxes(im2, slim_image_array, 8)

row=im2.shape[0]
col=im2.shape[1]
arr=np.asarray(arr)
arr=arr.astype('uint8')

# iterating image to quantatize the pixels
for i in range(row):
    for j in range(col):
        old_value=im2[i][j]
        new_value=np.asarray(pixel_list[(i,j)])
        im2[i][j][:]=new_value.astype('uint8')




im2=cv2.cvtColor(im2, cv2.COLOR_RGB2BGR)
cv2.imshow('image',im2)
cv2.waitKey(0)
cv2.destroyAllWindows()
