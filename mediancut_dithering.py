# important libraries
import numpy as np
from PIL import Image
import cv2
import math

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


# Function to get the median index of the array
def getMedianIndex(input_array_flat):
    length = len(input_array_flat)
    return int((length+1)/2)


# to store median colour of a bucket in array for making colour map
def median_cut_with_dithering(image, flat_im):
    red_average = np.mean(flat_im[:, 2])
    green_average = np.mean(flat_im[:, 3])
    blue_average = np.mean(flat_im[:, 4])

    red_average = np.uint8(red_average)
    green_average = np.uint8(green_average)
    blue_average = np.uint8(blue_average)

    arr.append((red_average,green_average,blue_average))
        

# starting of algorithum by spliting the array into 2 buckets
def split_into_boxes(image_array, flat_image, num_colors):
    if len(flat_image) == 0:
        return

    if num_colors == 0:
        median_cut_with_dithering(image_array, flat_image)
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

# helpper function to take care of pixel colours going out of [0,255]
def comp(x):
    if x<0:
        return 0.0
    if x>255:
        return 255.0
    return x

# to compute neirest neighbour colour map for a pixel colour
def comp1(a,b,c,arr1):
    x,y=arr1.shape
    maxi=float(10000000)
    r=-1
    b=-1
    g=-1
    for i in range(x):
        point1=np.array(arr1[i][:])
        point2=np.array((a,b,c))
        cons=math.dist(point1,point2)
        if cons < maxi :
            r,g,b=arr1[i][:]
            maxi=cons
        else :
            z=2
    return(r,g,b)

# to load image
im = cv2.imread("Lena.png")

image_array = np.copy(im)
image_array=cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)


im2 = np.copy(image_array)
slim_image_array = flatten_image(image_array)

split_into_boxes(im2, slim_image_array, 2)

row=im2.shape[0]
col=im2.shape[1]
arr=np.asarray(arr)
arr=arr.astype('uint8')

# iterating image to quantatize the pixels
for i in range(row):
    for j in range(col):
        old_value=im2[i][j]
        new_value=np.asarray(comp1(im2[i][j][0],im2[i][j][1],im2[i][j][2],arr))
        old_value=old_value.astype(float)
        new_value=new_value.astype(float)

        # computing error
        e=np.subtract(old_value,new_value)

        # assigning colours from colour map
        im2[i][j][:]=new_value.astype('uint8')

        # appling floyed steinberg dithering into 4 directions
        if i<row-1:
            r=comp(im2[i+1][j][0].astype(float)+e[0]*3/8.0)
            im2[i+1][j][0]=np.uint8(r)
            b=comp(im2[i+1][j][1].astype(float)+e[1]*3/8.0)
            im2[1+1][j][1]=np.uint8(b)
            g=comp(im2[i+1][j][2].astype(float)+e[2]*3/8.0)
            im2[1+1][j][2]=np.uint8(g)

        if j<col-1:
            r=comp(im2[i][j+1][0].astype(float)+e[0]*3/8.0)
            im2[i][j+1][0]=np.uint8(r)
            b=comp(im2[i][j+1][1].astype(float)+e[1]*3/8.0)
            im2[1][j+1][1]=np.uint8(b)
            g=comp(im2[i][j+1][2].astype(float)+e[2]*3/8.0)
            im2[1][j+1][2]=np.uint8(g)
            
        if (i<row-1)and(j<col-1):
            r=comp(im2[i+1][j+1][0].astype(float)+e[0]*1/16.0)
            im2[i+1][j+1][0]=np.uint8(r)
            b=comp(im2[i+1][j+1][1].astype(float)+e[1]*1/16.0)
            im2[1+1][j+1][1]=np.uint8(b)
            g=comp(im2[i+1][j+1][2].astype(float)+e[2]*1/16.0)
            im2[1+1][j+1][2]=np.uint8(g)
        
        if (j>0)and(i<row-1):
            r=comp(im2[i+1][j-1][0].astype(float)+e[0]*3/16.0)
            im2[i+1][j-1][0]=np.uint8(r)
            b=comp(im2[i+1][j-1][1].astype(float)+e[1]*3/16.0)
            im2[1+1][j-1][1]=np.uint8(b)
            g=comp(im2[i+1][j-1][2].astype(float)+e[2]*3/16.0)
            im2[1+1][j-1][2]=np.uint8(g)



im2=cv2.cvtColor(im2, cv2.COLOR_RGB2BGR)
cv2.imshow('image',im2)
cv2.waitKey(0)
cv2.destroyAllWindows()
