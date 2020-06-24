import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import re
#import shutil
#from matplotlib.pyplot import figure
#%matplotlib qt

#------------------------------------------------------
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
            #cv2.imshow('img',img)
            #cv2.waitKey(500)

    cv2.destroyAllWindows()
    image_shape = gray.shape[::-1]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

    return ret, mtx, dist, rvecs, tvecs
#-------------------------------------------------------
#Define a function that disorts an image given the calibration parameter
def undistort_image(image_raw, mtx, dist):
    dst = cv2.undistort(image_raw, mtx, dist, None, mtx)
    return dst
#---------------------------------------------------------

# Define Function to convert the image threshiold
def chanel_function(img, s_thresh, l_thresh, sx_thresh, sy_thresh, red_tresh, sobel_kernel):
    #img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    #Convert to Red channel
    img_red = img[:,:,0]
    #convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #Red Chanel Threshold
    red_binary = np.zeros_like(img_red)
    red_binary[(img_red >= red_tresh[0]) & (img_red <= red_tresh[1])]  = 1

    # Sobel x
    #sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    #sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack((red_binary, sxbinary, s_binary)) * 255

    #cv2.destroyAllWindows()
    return color_binary

#-------------------------------------------------
#region of interest from project 1
def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#Define Function for perspective transforms
def perspective_transform(img):


    #cv2.imshow('region of interewt',img)
    #cv2.waitKey(2000)

    imshape = img.shape
    #vertices = np.array([[(60,imshape[0]),(imshape[1]//2-100, 420), (imshape[1]//2+100, 420), (imshape[1]-60,imshape[0])]], dtype=np.int32)
    vertices = np.array([[(60,imshape[0]),(imshape[1]//2-90, 450), (imshape[1]//2+90, 450), (imshape[1]-60,imshape[0])]], dtype=np.int32)

    masked = region_of_interest(img, vertices)
    #cv2.imshow('region of interewt',masked)
    #cv2.waitKey(2000)


    img_size = (img.shape[1], img.shape[0])
    #src = np.float32([[700,450], [1100,img_size[0]], [200,img_size[0]], [500,450]])
    #dst = np.float32([[900,0], [900,img_size[0]], [350,img_size[0]], [350,0]])

    #src = np.float32([[0, 673], [1207, 673], [0, 450], [1280, 450]])
    #dst = np.float32([[569, 223], [711, 223], [0, 0], [1280, 0]])

    #src = np.float32([[120, 720], [550, 470], [700, 470], [1160, 720]])
    #dst = np.float32([[200,720], [200,0], [1080,0], [1080,720]])

    src = np.float32([[585, 455], [705, 455], [1130, 720], [190, 720]])
    offset = 200
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])


    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size)

    return warped, Minv, img_size

#-------------------------------------------------
#Define Function for Lane Line Detection Sliding Sliding Windows

def lane_finding(img): #, nwindows, margin, minipix):

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50


    bottom_half = img[img.shape[0]//2:,:]
    histogram = np.sum(bottom_half, axis=0)
    out_img = np.dstack((img, img, img))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(img.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2)

        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(img):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = lane_finding(img)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fit, right_fit, ploty, left_fitx, right_fitx


def measure_curvature_real(ploty, left_fit_cr, right_fit_cr, image_mid):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    #ploty, left_fit_cr, right_fit_cr = generate_data(ym_per_pix, xm_per_pix)

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    left_pos = (left_fit_cr[0]*y_eval**2 + left_fit_cr[1]*y_eval + left_fit_cr[2])
    right_pos = (right_fit_cr[0]*y_eval**2 + right_fit_cr[1]*y_eval + right_fit_cr[2])

    lanes_mid = (left_pos+right_pos)/2.0
    distance_from_mid = image_mid  - lanes_mid
    offset_mid = xm_per_pix*distance_from_mid

    return left_curverad, right_curverad, offset_mid


#######################################################


#################################################

##############################
#NanoDegree Project 2 - Rubric
#############################
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
#pattern = re.compile("^.+processed.jpg$")
fileExtension = '.jpg'
files = os.listdir(filepath)
for file in files:
    if file.endswith(fileExtension): #somehow i had a none jpg file in the folder after severall iterations
        # import the image if it is not a saved output
    #    if not pattern.match(file):
        image_filepath = filepath + file
        image_output = outputpath + file
        image_raw = mpimg.imread(image_filepath)

        # process image
        un_dist = undistort_image(image_raw, mtx, dist)
        # next images
        #plt.figure()
        # plot the image
        #plt.imshow(un_dist)
        un_dist_BGR = cv2.cvtColor(un_dist, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_output.replace(".jpg","-undistort.jpg"), un_dist_BGR)

#--------------------------------------------------------
# Rubric 3: Use color transforms, gradients, etc., to create a thresholded binary image.
#Defining parameters
sobel_kernel = 7
s_thresh = (180, 255)
l_thresh = (180, 255)
sx_thresh = (20, 100)
sy_thresh = (20,100)
red_tresh = (230, 255)


filepath_3 = "output_images/"
outputpath_3 = "output_images/"
pattern_3 = re.compile("^.+undistort.jpg$")
fileExtension = '.jpg'
files = os.listdir(filepath_3)
for file in files:
    if file.endswith(fileExtension):
        # import the image if it is not a saved output
        if pattern_3.match(file):
            image_filepath_3 = filepath_3 + file
            image_output_3 = outputpath_3 + file
            image_3 = mpimg.imread(image_filepath_3)

            # process image
            process_Channel = chanel_function(image_3, s_thresh, l_thresh, sx_thresh, sy_thresh, red_tresh, sobel_kernel)

            #cv2.imshow('processed',process_Channel)
            #cv2.waitKey(5000)
            #cv2.destroyAllWindows()

            gray_rub3 = cv2.cvtColor(process_Channel, cv2.COLOR_RGB2GRAY)
            #cv2.imshow('output',gray)
            #cv2.waitKey(5000)

            cv2.imwrite(image_output_3.replace("-undistort.jpg","-Rubric3.jpg"), gray_rub3)


#----------------------------------------------------------------
#Rubric 4: Apply a perspective transform to rectify binary image ("birds-eye view").
filepath_4 = "output_images/"
outputpath_4 = "output_images/"
pattern_4 = re.compile("^.+Rubric3.jpg$")
fileExtension = '.jpg'
files = os.listdir(filepath_4)
for file in files:
    if file.endswith(fileExtension):
        # import the image if it is not a saved output
        if  pattern_4.match(file):
            image_filepath_4 = filepath_4 + file
            image_output_4 = outputpath_4 + file
            image_4 = mpimg.imread(image_filepath_4)


            # process image
            process_transform, Minv, img_size_transform = perspective_transform(image_4)
            #cv2.imshow('processed',process_transform)
            #cv2.waitKey(500)
            #cv2.destroyAllWindows()

            # next image
            #plt.figure()

            # plot the image
            #plt.imshow(un_dist)

            #un_dist_BGR = cv2.cvtColor(un_dist, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_output_4.replace("-Rubric3.jpg","-Rubric4.jpg"), process_transform)

#----------------------------------------------------------------
#Rubric 5: Detect lane pixels and fit to find the lane boundary.
filepath_5 = "output_images/"
outputpath_5 = "output_images/"
pattern_5 = re.compile("^.+Rubric4.jpg$")
fileExtension = '.jpg'
files = os.listdir(filepath_5)
curve_list = []
for file in files:
    if file.endswith(fileExtension):
        # import the image if it is not a saved output
        if  pattern_5.match(file):
            image_filepath_5 = filepath_5 + file
            image_output_5 = outputpath_5 + file
            image_5 = mpimg.imread(image_filepath_5)


            #process images
            image_5_processed, left_fit, right_fit, ploty, left_fitx, right_fitx = fit_polynomial(image_5)
            #plt.imshow(image_5_processed)
            #cv2.imshow('processed',image_5_processed)
            #cv2.waitKey(500)
            #cv2.destroyAllWindows()

            #Rubric 6: Determine the curvature of the lane and vehicle position with respect to center.
            image_mid = image_5.shape[1]/2.0
            left_curverad, right_curverad, veh_pos_offset = measure_curvature_real(ploty, left_fit, right_fit, image_mid)
            curve_list.append([file, left_curverad, right_curverad, veh_pos_offset])


            #print(left_curverad, 'm', right_curverad, 'm', veh_pos_offset, 'm')

            #Rubric 7: Warp the detected lane boundaries back onto the original image.
            #Source for Help: https://github.com/wonjunee/Advanced-Lane-Finding/blob/master/Advanced-Lane-Finding-Submission.ipynb

            #Create new empty Image to put the found Lines on
            empty_warp = np.zeros_like(image_5).astype(np.uint8)
            color_empty = np.dstack((empty_warp, empty_warp, empty_warp))

            #transform the curvature into cv2 format
            #print(left_pos, 'left pos', y_eval, 'yeval')

            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            lanes_pts = np.hstack((pts_left, pts_right))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(color_empty, np.int_([lanes_pts]), (0, 255, 0))

            # Warp the blank back to original image space using inverse perspective matrix (Minv)
            newwarp = cv2.warpPerspective(color_empty, Minv, img_size_transform)
            cv2.imwrite(image_output_5.replace("-Rubric4.jpg","-Rubric7.jpg"), newwarp)

            #cv2.imshow('processed',newwarp)
            #cv2.waitKey(500)
            #cv2.destroyAllWindows()

print(curve_list)
print(len(curve_list))

#Rubric 8: Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
filepath = "test_images/"
outputpath = "output_images/"
#pattern = re.compile("^.+processed.jpg$")
fileExtension = '.jpg'
fileExtension_rubric7 = '-Rubric7.jpg'
files = os.listdir(filepath)
files_output = os.listdir(outputpath)
for file in files:
    if file.endswith(fileExtension):
        image_filepath = filepath + file
        image_raw = mpimg.imread(image_filepath)
        file_orginal = file.replace('.jpg', '')
        #file_orginal = file - fileExtension

        #print(file_orginal)

        #cv2.imshow('processed',image_raw)
        #cv2.waitKey(500)
        #cv2.destroyAllWindows()

        for file in files_output:
            if file.endswith(fileExtension_rubric7):
                if file.startswith(file_orginal):
                    lanes_filepath = outputpath + file
                    lanes_pic = mpimg.imread(lanes_filepath)


                    #cv2.imshow('processed',lanes_pic)
                    #cv2.waitKey(500)
                    #cv2.destroyAllWindows()
                    image_raw_RGBsss = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
                    result_matched = cv2.addWeighted(image_raw_RGBsss, 1, lanes_pic, 0.3, 0)

                    cv2.imshow('processed',result_matched)
                    cv2.waitKey(500)
                    cv2.destroyAllWindows()


                    cv2.imwrite(lanes_filepath.replace("-Rubric7.jpg","-Rubric7-alternative.jpg"), result_matched)

                    for i in range(len(curve_list)):
                        if curve_list[i][0].startswith(file_orginal): #'file_orginal'+ 'Rubric4.jpg':
                            print(curve_list[i][0])
                            curvature_average = (curve_list[i][1] + curve_list[i][2])/2
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            text1 = "Radius of Curvature: {} m".format(int(curvature_average))
                            #textxx = 'testss'
                            cv2.putText(result_matched,text1,(400,100), font, 1,(255,255,255),2)

                            text2 = "Vehicle offset: {} m".format(float(curve_list[i][3]))
                            cv2.putText(result_matched,text2,(400,150), font, 1,(255,255,255),2)

                            cv2.imshow('processed',result_matched)
                            cv2.waitKey(3000)
                            cv2.destroyAllWindows()

                            cv2.imwrite(lanes_filepath.replace("-Rubric7.jpg","-Rubric8.jpg"), result_matched)
