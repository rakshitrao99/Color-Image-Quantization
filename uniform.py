from PIL import Image
# important python library to handle images

import numpy as np
# important python libray to handle array in python and easy computation

# function to get the array for the image


def getArray(img):
    return np.asarray(img)


# Enter the location of the image
im = Image.open("/home/rakshit/Documents/col783/me.jpeg")

imagedata = getArray(im)


# This function is used to divide the red channel of the image into 8 regions
# We will know have instead of 256 different of shades of red only 8 different
#  shades of red color

def get_region_for_red(color_value):
    # colors divided into 8 regions for red color space
    eight_regions = np.array([[0, 31], [32, 63], [64, 95], [96, 127], [
        128, 159], [160, 191], [192, 223], [224, 255]])
    size_eight_regions = eight_regions.shape[0]
    for i in range(0, size_eight_regions):
        x = eight_regions[i][0]
        y = eight_regions[i][1]
        if color_value >= x and color_value <= y:
            return i


# This function is used to divide the green channel of the image into 8 regions
# We will know have instead of 256 different of shades of green
# only 8 different shades of green color

def get_region_for_green(color_value):
    # colors divided into 8 regions for green color space
    eight_regions = np.array([[0, 31], [32, 63], [64, 95], [96, 127],
                              [128, 159], [160, 191], [192, 223], [224, 255]])
    size_eight_regions = eight_regions.shape[0]
    for i in range(0, size_eight_regions):
        x = eight_regions[i][0]
        y = eight_regions[i][1]
        if (color_value >= x and color_value <= y):
            return i

# This function is used to divide the blue channel of the image into 4 regions
# We will know have instead of 256 different of shades of blue
# only 4 different shades of blue color


def get_region_for_blue(color_value):
    # colors divided into 4 regions for blue color space
    eight_regions = np.array([[0, 63], [64, 127], [128, 191], [192, 255]])
    size_eight_regions = eight_regions.shape[0]
    for i in range(0, size_eight_regions):
        x = eight_regions[i][0]
        y = eight_regions[i][1]
        if (color_value >= x and color_value <= y):
            return i


# mean/average color of each bucket
def define_red(index_value):
    representative_red_color_per_region = np.array(
        [16, 48, 80, 112, 144, 176, 208, 240])
    return representative_red_color_per_region[index_value]

# mean/average color of each bucket


def define_green(index_value):
    representative_green_color_per_region = np.array(
        [16, 48, 80, 112, 144, 176, 208, 240])
    return representative_green_color_per_region[index_value]

# mean/average color of each bucket


def define_blue(index_value):
    representative_blue_color_per_region = np.array([32, 96, 160, 224])
    return representative_blue_color_per_region[index_value]


# copy the original image into a new array
new_image = np.copy(imagedata)

for i in range(imagedata.shape[0]):
    for j in range(imagedata.shape[1]):

        red = new_image[i][j][0]
        green = new_image[i][j][1]
        blue = new_image[i][j][2]

        r_index = get_region_for_red(red)
        g_index = get_region_for_green(green)
        b_index = get_region_for_blue(blue)

        new_image[i][j][0] = define_red(r_index)
        new_image[i][j][1] = define_green(g_index)
        new_image[i][j][2] = define_blue(b_index)

data = Image.fromarray(new_image)
# data.save("output_uniform_quantization.jpg")
data.show()