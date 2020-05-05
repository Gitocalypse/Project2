import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import re
#%matplotlib qt


"""
#---------------------------------------------------------------
def camera_cal()

# prepare object points
nx = 8#TODO: enter the number of inside corners in x
ny = 6#TODO: enter the number of inside corners in y

# Make a list of calibration images
fname = 'calibration_test.png'
img = cv2.imread(fname)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# If found, draw corners
if ret == True:
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    plt.imshow(img)
#---------------------------------------------------------------
"""



#def camera_cal(images, nx, ny):

nx = 9 #TODO: enter the number of inside corners in x
ny = 6 #TODO: enter the number of inside corners in y

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
#images = glob.glob('../camera_cal/calibration*.jpg')
images = glob.glob("./camera_cal/calibration*.jpg")

print(images)

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    plt.imshow(gray)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        cv2.imshow('img',img)
        #cv2.waitKey(500)



image_shape = gray.shape[::-1]

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

print(ret)


#return ret, mtx, dist, rvecs, tvecs

#dst = cv2.undistort(img, mtx, dist, None, mtx)

#cv2.destroyAllWindows()

#def undistort_image():


#___________________________________________________________________
#Pipeline

#Rubric 1: Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

nx = 9 #TODO: enter the number of inside corners in x
ny = 6 #TODO: enter the number of inside corners in y

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#objp = np.zeros((6*9,3), np.float32)
#objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

#store all calibration images in images
images = glob.glob('../camera_cal/calibration*.jpg')

#Call Camera Cal Function for matrix and coefficients
ret, mtx, dist, rvecs, tvecs = camera_cal(images, nx, ny)

#--------------------------------------------------------


#            dst = cv2.undistort(img, mtx, dist, None, mtx)
