#After Feedback for first submit of Project 2, I created a new file that deals with the Video
#I rewrote the complete pipline to optimize for video

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
### Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

################################################################################
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
        #print(corners)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)

    image_shape = gray.shape[::-1]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

    return ret, mtx, dist, rvecs, tvecs
################################################################################
#Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
nx = 9 #TODO: enter the number of inside corners in x
ny = 6 #TODO: enter the number of inside corners in y
#store all calibration images in images
images = glob.glob("./camera_cal/calibration*.jpg")
#Call Camera Cal Function for matrix and coefficients
ret, mtx, dist, rvecs, tvecs = camera_cal(images, nx, ny)
#print(mtx, 'mtx')
################################################################################
#Used functions - Used in the image process pipline
#-------------------------------------------------------------------------------
#Define Gradient direction threshold Function
def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(abs_sobel_y,abs_sobel_x)
    direction = np.absolute(direction)
    # 5) Create a binary mask where direction thresholds are met
    mask = np.zeros_like(direction)
    mask[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return mask
#-------------------------------------------------------------------------------
def abs_sobel_thresh(gray, orient='x', thresh_min=0, thresh_max=255):
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    max_value = np.max(abs_sobel)
    binary_output = np.uint8(255*abs_sobel/max_value)
    threshold_mask = np.zeros_like(binary_output)
    threshold_mask[(binary_output >= thresh_min) & (binary_output <= thresh_max)] = 1
    return threshold_mask
#-------------------------------------------------------------------------------
# Define Function to convert the image using different threshold
def chanel_function(img):
    #Define parameters - rework after first commit to deal better with shadows etc.
    sobel_kernel = 7
    s_thresh = (100, 255)
    l_thresh = (120, 255)
    sx_thresh = (10, 200)
    colorthreshold = 150

    #convert to gray
    grayy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply gradient threshold on the horizontal gradient
    sxbinary = abs_sobel_thresh(grayy, 'x', 10, 200)

    #Gradient Direction Threshold
    dirbinary = dir_threshold(grayy, thresh=(np.pi/6, np.pi/2))

    # combine the gradient and direction thresholds.
    combined_sobel_dir = ((sxbinary == 1) & (dirbinary == 1))

    #Convert to Red and Green channel
    img_red = img[:,:,0]
    img_green = img[:,:,1]
    red_green = (img_red > colorthreshold) & (img_green > colorthreshold)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    l_condition = (l_channel > l_thresh[0]) & (l_channel <= l_thresh[1])
    s_condition = (s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])

    color_binary = np.zeros_like(img_red)
    color_binary[(red_green & l_condition) & (s_condition | combined_sobel_dir)] = 1

    #For visualization
    color_binary = color_binary * 255

    return color_binary
#-------------------------------------------------------------------------------
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
    cv2.fillPoly(mask, [vertices], ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
#-------------------------------------------------------------------------------
#Define Function for perspective transforms
def perspective_transform(img):

    imshape = img.shape
    #After feedback change of verties
    #vertices = np.array([[(60,imshape[0]),(imshape[1]//2-90, 450), (imshape[1]//2+90, 450), (imshape[1]-60,imshape[0])]], dtype=np.int32)
    #changed verties for region of interest
    height, width = img.shape
    vertices = np.array([[0,height-1], [width/2, int(0.5*height)], [width-1, height-1]], dtype=np.int32)

    masked = region_of_interest(img, vertices)

    img_size = (img.shape[1], img.shape[0])
    #changed source points
    #src = np.float32([[585, 455], [705, 455], [1130, 720], [190, 720]])
    src = np.float32([[220, 720], [1110, 720], [720, 470], [570, 470]])
    #changed destination points
    #offset = 200
    #dst = np.float32([[offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])
    dst = np.float32([[320, 720], [920, 720], [920, 1], [320, 1]])


    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(masked, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, Minv, img_size
#-------------------------------------------------------------------------------
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
        #cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        #(win_xleft_high,win_y_high),(0,255,0), 2)
        #cv2.rectangle(out_img,(win_xright_low,win_y_low),
        #(win_xright_high,win_y_high),(0,255,0), 2)

        ###Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        ### If you found > minpix pixels, recenter next window ###
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
#------------------------------------------------------------------------------
def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 48

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    #Search arround the lane from the frame before
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return leftx, lefty, rightx, righty, out_img
#-------------------------------------------------------------------------------
def measure_curvature_real(ploty, left_fit_cr, right_fit_cr, image_mid):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
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
#-------------------------------------------------------------------------------
def fit_polynomial(img):
    global frame_count
    global left_fit
    global right_fit
    global old_left_fit
    global old_right_fit

    old_left_fit = left_fit #store values of previous frame to potentialaly re-use
    old_right_fit = right_fit

    #Search for the pixel that make up the lane boundary
    if frame_count  == 0: #for the first frame always use sliding window
        leftx, lefty, rightx, righty, out_img = lane_finding(img)
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    elif frame_count != 0: #for all following frames try search_around_poly Function
        leftx, lefty, rightx, righty, out_img = search_around_poly(img, left_fit, right_fit)
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx,  2)
        if left_fit == [] or right_fit == []: #if the method of search arround the polynomial doenst work use sliding window
            leftx, lefty, rightx, righty, out_img = lane_finding(img)
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

    # Check quality of the found curves
    # Check for similiarty
    similarty_check = cv2.matchShapes(left_fit, right_fit, 1,0.0)

    #check for distance between the linces
    line_distance = np.mean(right_fitx - left_fitx )

    #using old line, if quality isnt good enough
    if frame_count != 0:
        #if line_distance > 620:
        if similarty_check > 0.2 or line_distance < 575 or line_distance > 625:
            left_fit = old_left_fit
            right_fit = old_right_fit
            #print('pingg alte linie')

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    frame_count += 1

    return out_img, left_fit, right_fit, ploty, left_fitx, right_fitx
#------------------------------------------------------------------------------
#Define function that averages the found lines over the last couple of frames
def  averaging_lines(previous_lines, new_line):
    """
        This function computes an averaged lane line by averaging over previous good frames.
        https://github.com/subodh-malgonde/advanced-lane-finding/blob/master/Advanced_Lane_Lines.ipynb
    """
    # Number of frames to average over
    num_frames = 13

    if new_line is None:
        # No line was detected

        if len(previous_lines) == 0:
            # If there are no previous lines, return None
            return previous_lines, None
        else:
            # Else return the last line
            return previous_lines, previous_lines[-1]
    else:
        if len(previous_lines) < num_frames:
            # we need at least num_frames frames to average over
            previous_lines.append(new_line)
            return previous_lines, new_line
        else:
            # average over the last num_frames frames
            previous_lines[0:num_frames-1] = previous_lines[1:]
            previous_lines[num_frames-1] = new_line
            new_line = np.zeros_like(new_line)
            for i in range(num_frames):
                new_line += previous_lines[i]
            new_line /= num_frames
            #print('pingg averaging')
            return previous_lines, new_line

################################################################################
# Function for processing the video part of Project 2
def process_image(image): #image for bgr to rgb
    global left_average_range
    global right_average_range
    global left_average_range_x
    global right_average_range_x

    img = image
    #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #Apply a distortion correction to raw images.
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    #Use color transforms, gradients, etc., to create a thresholded binary image
    process_Channel = chanel_function(dst)

    # Apply a perspective transform to rectify binary image ("birds-eye view").
    #gray = cv2.cvtColor(process_Channel, cv2.COLOR_RGB2GRAY)
    process_transform, Minv, img_size_transform = perspective_transform(process_Channel)

    #Detect lane pixels and fit to find the lane boundary
    image_5_processed, left_fit, right_fit, ploty, left_fitx, right_fitx = fit_polynomial(process_transform)

    #smoothing the line by avering ofer the last couple of frames
    right_average_range_x, right_fitx = averaging_lines(right_average_range_x, right_fitx)
    left_average_range_x, left_fitx = averaging_lines(left_average_range_x, left_fitx)
    right_average_range, right_fit = averaging_lines(right_average_range, right_fit) #also average to calculate the lane curvature
    left_average_range, left_fit = averaging_lines(left_average_range, left_fit) #also average to calculate the lane curvature

    #Determine the curvature of the lane and vehicle position with respect to center.
    image_mid = process_transform.shape[1]/2.0
    left_curverad, right_curverad, veh_pos_offset = measure_curvature_real(ploty, left_fit, right_fit, image_mid)

    #Warp the detected lane boundaries back onto the original image.
    #Create new empty Image to put the found Lines on
    empty_warp = np.zeros_like(process_transform).astype(np.uint8)
    color_empty = np.dstack((empty_warp, empty_warp, empty_warp))

    #transform the curvature into cv2 format
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    lanes_pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_empty, np.int_([lanes_pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_empty, Minv, img_size_transform)

    #Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    result_matched = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    curvature_average = (left_curverad + right_curverad)/2 #falls in das video die kurve mit soll, dass muss die funktion diese übergeben
    font = cv2.FONT_HERSHEY_SIMPLEX
    text1 = "Radius of Curvature: {} m".format(int(curvature_average))
    cv2.putText(result_matched,text1,(400,100), font, 1,(255,255,255),2)

    text2 = "Vehicle offset: {} m".format(float(veh_pos_offset))
    cv2.putText(result_matched,text2,(400,150), font, 1,(255,255,255),2)

    return result_matched

################################################################################
"""
Start the pipline with the video.
"""
frame_count = 0 #to count fhe frames, to use sliding window method only in the beginning
left_fit = None
right_fit = None
old_left_fit = None
old_right_fit = None
left_average_range = []
right_average_range = []
left_average_range_x = []
right_average_range_x = []

#Vide processing
white_output = 'white.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)

"""
Feedback from Reviewer first submit
SUGGESTION:
    - Applying color threshold to the B(range:145-200 in LAB for shading & brightness changes and R in RGB in final pipeline can also help in detecting the yellow lanes.
    - And thresholding L (range: 215-255) of Luv for whites.
    - If perspective transform captures a very wide region of interest then it can bring noise. For single lane taking little wider region of interest will led in lesser noise in the warped binary images. Please don't shift the lane horizontally after using the perspective transform because this way your camera position get disturb from the center image and you can get an issues in calculating vehicle offset from center. Also you can check this link for more information:
        https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
        http://www.ijser.org/researchpaper%5CA-Simple-Birds-Eye-View-Transformation-Technique.pdf
    - Check both the polynomials are correctly distanced with respect to the width of a lane.
    - Check the curvature of both polynomials whether it is similiar or not.
    - Also check the binary thresholding for improvement because mostly the detection is failing in shadows and brighter lane.
    - Also if there is an issue in a single frame then you can reject that wrong detection and reuse the confident detection from the previous detection. For smoothening the lane lines and for reducing wrong detection you can try averaging lane detection using a series of multiple frames.
    - In addition to other filtering mechanisms. You can also use cv2.matchShapes as a means to make sure the final warp polygon is of good quality. This can be done by comparing two shapes returning 0 index for identical shapes. You can use this to make sure that the polygon of your next frame is closer to what is expected and if not then can use the old polygon instead. This way you are faking it until a new frames appear and hence will get good results.

"""
