import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import re
#%matplotlib qt


#Define a function that can calculate the calibration parameter and retunr them
def camera_cal(images, nx, ny):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.


    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)
        print(corners)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()
    image_shape = gray.shape[::-1]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

    return ret, mtx, dist, rvecs, tvecs

#Define a function that disorts an image given the calibration parameter
def undistort_image(image_raw, mtx, dist):
    dst = cv2.undistort(image_raw, mtx, dist, None, mtx)
    return dst

#######################
#Project 2 - Rubric
#######################


#--------------------------------------------------------
#Rubric 1: Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

nx = 9 #TODO: enter the number of inside corners in x
ny = 6 #TODO: enter the number of inside corners in y

#store all calibration images in images
images = glob.glob("./camera_cal/calibration*.jpg")

#Call Camera Cal Function for matrix and coefficients
ret, mtx, dist, rvecs, tvecs = camera_cal(images, nx, ny)

#--------------------------------------------------------
# Rubric 2: Apply a distortion correction to raw images.
# Pipline from Project 1

filepath = "test_images/"
outputpath = "output_images/"
pattern = re.compile("^.+processed.jpg$")
fileExtension = '.jpg'
files = os.listdir(filepath)
for file in files:
    if file.endswith(fileExtension): #somehow i had a none jpg file in the folder after severall iterations
        # import the image if it is not a saved output
        if not pattern.match(file):
            image_filepath = filepath + file
            image_raw = mpimg.imread(image_filepath)

            # process image
            un_dist = undistort_image(image_raw, mtx, dist)

            #dst = cv2.undistort(image_raw, mtx, dist, None, mtx)
            #image_wlines = prepare_img(image)               # attempt 2
            #print('Image ', image_filepath, ' has dimensions: ', image.shape)

            # next image
            plt.figure()

            # plot the image
            plt.imshow(un_dist)

            # writeout the image with "-processed" in the name so it will not be reprocessed.
            #plt.savefig(image_filepath.replace(".jpg","-processed.jpg"))
            #plt.savefig(outputpath.replace(".jpg","-processed.jpg"))
            #plt.savefig()
            plt.savefig("output_images/" + 'final_' + file)
            #cv2.imwrite(os.path.join(outputpath + file.name, un_dist))
            #cv2.imwrite(os.path.join('output_images/final{}.jpg',un_dist))
            #cv2.imwrite(os.path.join("output_images/" + str(file) + '.jpg', un_dist))
            #cv2.imwrite(os.path.join("output_images/" + file + 'final.jpg', un_dist))
