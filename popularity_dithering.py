# important libraries
import numpy as np
import cv2
import math

# helpper function to take care of pixel colours going out of [0,255]
def comp(x):
    if x<0:
        return 0.0
    if x>255:
        return 255.0
    return x


# to compute neirest neighbour colour map for a pixel colour
def comp1(a,b,c,arr):
    x,y=arr.shape
    maxi=float(10000000)
    r=-1
    b=-1
    g=-1
    for i in range(x):
        point1=np.array(arr[i][0])
        point2=np.array((a,b,c))
        cons=math.dist(point1,point2)
        if cons < maxi :
            b,g,r=arr[i][0]
            maxi=cons
        else :
            z=2
    return(b,g,r)


# loading image 
img1 =cv2.imread('Lena.png')
img=img1.copy()
row,col,cha=img.shape
a=0
b=0
c=0
dicti={}

# iterating image to enter colours into dictionary
for i in range(row) :
    for j in range(col) :
        a=img[i,j,0]
        b=img[i,j,1]
        c=img[i,j,2]
        if (a,b,c) in dicti:
            dicti[(a,b,c)]= dicti[(a,b,c)]+1
        else:
            dicti[(a,b,c)]=1

# to get sorted array of distinct colours accrding to frequency
sorted_tuples = sorted(dicti.items(), key=lambda item: item[1],reverse=True)
arr=np.array(sorted_tuples)

# to get top k+1 colours
k=255
arr=arr[0:k]

# iterating image to quantatize the pixels
for i in range(row):
    for j in range(col):
        old_value=img[i][j]
        new_value=np.asarray(comp1(img[i][j][0],img[i][j][1],img[i][j][2],arr))
        old_value=old_value.astype(float)
        new_value=new_value.astype(float)

        # computing error
        e=np.subtract(old_value,new_value)

        # assigning colours from colour map
        img[i][j][:]=new_value.astype('uint8')

        # appling floyed steinberg dithering into 4 directions
        if i<row-1:
            b=comp(img[i+1][j][0].astype(float)+e[0]*5/16.0)
            img[i+1][j][0]=np.uint8(b)
            g=comp(img[i+1][j][1].astype(float)+e[1]*5/16.0)
            img[1+1][j][1]=np.uint8(g)
            r=comp(img[i+1][j][2].astype(float)+e[2]*5/16.0)
            img[1+1][j][2]=np.uint8(r)

        if j<col-1:
            b=comp(img[i][j+1][0].astype(float)+e[0]*7/16.0)
            img[i][j+1][0]=np.uint8(b)
            g=comp(img[i][j+1][1].astype(float)+e[1]*7/16.0)
            img[1][j+1][1]=np.uint8(g)
            r=comp(img[i][j+1][2].astype(float)+e[2]*7/16.0)
            img[1][j+1][2]=np.uint8(r)

            
        if (i<row-1)and(j<col-1):
            b=comp(img[i+1][j+1][0].astype(float)+e[0]*1/16.0)
            img[i+1][j+1][0]=np.uint8(b)
            g=comp(img[i+1][j+1][1].astype(float)+e[1]*1/16.0)
            img[1+1][j+1][1]=np.uint8(g)
            r=comp(img[i+1][j+1][2].astype(float)+e[2]*1/16.0)
            img[1+1][j+1][2]=np.uint8(r)

        if (j>0)and(i<row-1):
            b=comp(img[i+1][j-1][0].astype(float)+e[0]*3/16.0)
            img[i+1][j-1][0]=np.uint8(b)
            g=comp(img[i+1][j-1][1].astype(float)+e[1]*3/16.0)
            img[1+1][j-1][1]=np.uint8(g)
            r=comp(img[i+1][j-1][2].astype(float)+e[2]*3/16.0)
            img[1+1][j-1][2]=np.uint8(r)


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


