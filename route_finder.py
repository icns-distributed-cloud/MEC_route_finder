import threading
import time
import cv2
import numpy as np
import serial
import sys, getopt
import paho.mqtt
from PIL import Image
import paho.mqtt.client as mqtt
from paho.mqtt import publish
from video import create_capture
import Adafruit_CharLCD as LCD
import Adafruit_GPIO as GPIO

lcd_rs = 25
lcd_en = 24
lcd_d4 = 23
lcd_d5 = 17
lcd_d6 = 18
lcd_d7 = 22
lcd_backlight = 2
# Define LCD column and row size for 16x2 LCD.
lcd_columns = 16
lcd_rows = 2
lcd = LCD.Adafruit_CharLCD(lcd_rs, lcd_en, lcd_d4, lcd_d5, lcd_d6, lcd_d7, lcd_columns, lcd_rows, lcd_backlight)


ser = serial.Serial(
    "/dev/ttyAMA0",
    baudrate=115200,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    writeTimeout=1,
    timeout=10,
    rtscts=False,
    dsrdtr=False,
    xonxoff=False)

serLock = threading.Lock()

# --------------------------------------------------------MQTT----------------------------------------------------------#
def on_connect_cart(client, obj, flags, rc):
    if rc == 0:
        print("Cart connected with result code " + str(rc))
        client.subscribe("cart/room/starting_room_number")
        client.subscribe("cart/room/destination_room_number")
        client.subscribe("cart/parking")

    else:
        print("Bad connection returned code = ", rc)


def on_message_cart(client, obj, msg):
    # print("Cart new message: " + msg.topic + " " + str(msg.payload))
    roomNumber = str(msg.payload)
    if msg.topic == "cart/room/starting_room_number":
        if roomNumber == "331":
            # pass a value to micom for scenario 1
            serLock.acquire()
            try:
                ser.write('!')
            finally:
                serLock.release()
                # print("2 unlock")
            print(roomNumber)
        else:
            print("Unknown roomNumber")
            print(roomNumber)

    elif msg.topic == "cart/room/destination_room_number":
        if roomNumber == "323-1":
            # pass a value to micom for scenario 2
            serLock.acquire()
            try:
                ser.write('@')
            finally:
                serLock.release()
                # print("2 unlock")
            print(roomNumber)
        elif roomNumber == "250":
            # pass a value to micom for scenario 3
            serLock.acquire()
            try:
                ser.write('#')
            finally:
                serLock.release()
                # print("2 unlock")
            print(roomNumber)
        else:
            print("Unknown roomNumber")
            print(roomNumber)

    elif msg.topic == "cart/parking":
        if roomNumber == "0":
            # pass a vale to micom for parking scenario, which is not yet decided
            print(roomNumber)

        else:
            print("Unknown roomNumber")
            print(roomNumber)
    else:
        print("Unknown topic")


def on_publish_cart(client, obj, mid):
    print("mid: " + str(mid))


def on_subscribe_cart(client, obj, mid, granted_qos):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))


def on_log_cart(client, obj, level, string):
    print(string)

# The below lines will be used to publish the topics
# publish.single("elevator/starting_floor_number", "3", hostname="163.180.117.195", port=1883)
# publish.single("elevator/destination_floor_number", "2", hostname="163.180.117.195", port=1883)
# ---------------------------------------------------------------------------------------------------------------------#


# ------------------------------------------Hallway detction------------------------------------------------------------#
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False 
        # smoothing span
        self.n_avg = 10
        # x values of the last n_avg fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None 

        self.i = 0
    
  
    def add(self, recent_xfitted, ploty):
        # add in line
        self.detected = True
        self.recent_xfitted.append(recent_xfitted)

        # smooth out last n fits
        # keep only n_avg values
        if len(self.recent_xfitted) > self.n_avg:
            self.recent_xfitted = self.recent_xfitted[(len(self.recent_xfitted) - self.n_avg):]
        self.bestx = np.mean(self.recent_xfitted, axis=0)

        # find polynomial co-efficients of averaged x values
        self.best_fit = np.polyfit(ploty, self.bestx, 2)


def perspective_transform(img, src, dst):
    h = img.shape[0]
    w = img.shape[1]
    
    M = cv2.getPerspectiveTransform(src, dst)
    #Minv = cv2.getPerspectiveTransform(dst, src)
    
    warped = cv2.warpPerspective(img, M, (w,h))
    
    return warped, M

def find_lanes_sliding_window(binary_warped, draw_rect=False):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9 
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        #if draw_rect:

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    if leftx.size==0 or lefty.size==0 or rightx.size==0 or righty.size==0:
        return np.array([]), np.array([]), None, None
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
        
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    
                                     
    return left_fitx, right_fitx, left_lane_inds, right_lane_inds

def find_lanes_prev_fit(binary_warped, left_fit, right_fit):

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    if leftx.size==0 or lefty.size==0 or rightx.size==0 or righty.size==0:
        return np.array([]), np.array([]), None, None
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, left_lane_inds, right_lane_inds

def get_radius_curvature_center_offset(img_warped, left_lane_inds, right_lane_inds):
    
    if left_lane_inds is None or right_lane_inds is None:
        return -1, -1, -1
    
   
    nonzero = img_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty, leftx, 2)
    right_fit_cr = np.polyfit(righty, rightx, 2)

    # find the radius of curvature at the bottom of the image
    y_eval = (img_warped.shape[0] - 1)  # value = 480 - 1 = 479
    
    # Calculate the radius of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # compute offset from center
    left_bottomxm = left_fit_cr[0]*y_eval**2 + left_fit_cr[1]*y_eval + left_fit_cr[2]
    right_bottomxm = right_fit_cr[0]*y_eval**2 + right_fit_cr[1]*y_eval + right_fit_cr[2]
    
    # assume camera is mounted center of car	
    lane_center = (right_bottomxm + left_bottomxm)/2
    #if (lane_center > img_warped.shape[1]): lane_center = img_warped.shape[1]
    vehicle_pos = img_warped.shape[1]/2
    offset = lane_center - vehicle_pos
 
    return left_curverad, right_curverad, offset


def sersend(mode1, mode2, value):
    val = repr(value)

    ser.write('h'.encode())
    lcd.message('h')
    ser.write('h'.encode())
    lcd.message('h')
    time.sleep(0.004)

    ser.write(mode1.encode())
    lcd.message(mode1)
    ser.write(mode2.encode())
    lcd.message(mode2)
    time.sleep(0.004)

    ser.write(val.encode())
    lcd.message(val)
    time.sleep(0.004)
    if (ser.inWaiting() > 0):
        ser.write('t'.encode())
        lcd.message('t')
        ser.write('t'.encode())
        lcd.message('t')


def get_weight(img, rad_curvature, offset):
    
    max_weight = 9
    one_unit = round(img.shape[1]/max_weight,0) 
    #get absolute value of offset value in case it's negative value
    ratio = round(abs(offset)/one_unit, 0)

    
    #print("offset", offset)

    #direction = ''
    weight_center = 5
    if offset > 0:
        direction = 'right'
        weight_value = weight_center + ratio
        if weight_value > max_weight: weight_value = 9
    elif offset == 0:
        direction = 'center'
        weight_value = weight_center
    else:
        direction = 'left'
        weight_value = weight_center - ratio
        if weight_value <= 0: weight_value = 1

    value = int(weight_value)

    if value != 5:
        print(value)
        mode1 = 'p'
        mode2 = 's'
        sersend(mode1, mode2, value)
    else:
        print(value)
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #txt = 'Weight Value is ' + '{:04.2f}'.format(np.absolute(weight_value)) + ' on the ' + direction + ' direction '
    #cv2.putText(img, txt, (50,100), font, 1, (255,255,255), 2, cv2.LINE_AA)
    
    return img

def videoLineMeasurement_func(lock):
    # video = cv2.VideoCapture(0)
    while(True):
        lock.acquire()
        try:
            ret, img = video.read()
            # print("3 lock")
        finally:
            lock.release()
            # print("3 unlock")

        img2 = cv2.resize(img, (640,480))

        #step 1
        #Pre-processing: generate binary image
        hsv = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 70]))
        morp= cv2.morphologyEx(mask1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        blur = cv2.GaussianBlur(morp, (5, 5), 0)
        edges = cv2.Canny(blur,50,300,apertureSize = 3)



        #step 2: generate warped image - convert to bird-eyes view, output: warped image, matrix for perspective transform

        h = img2.shape[0]
        w = img2.shape[1]
        src = np.float32([[0, 0], [w,0], [0,h], [w,h]])
        __offset = 0
        dst = np.float32([[0, 0], [h-__offset, 0 ], [0, w-__offset], [h-__offset, w-__offset]])
        img_warped, M = perspective_transform(edges, src, dst)


        #step 3: detect Line boundary
        l_line = Line()
        r_line = Line()
        if (not l_line.detected) or (not r_line.detected):

            left_fitx, right_fitx, left_lane_inds, right_lane_inds = find_lanes_sliding_window(img_warped)

        else:

            left_fitx, right_fitx, left_lane_inds, right_lane_inds = find_lanes_prev_fit(img_warped, l_line.best_fit, r_line.best_fit)

        ploty = np.linspace(0, img_warped.shape[0]-1, img_warped.shape[0])

        # the distance between the lanes
        dist_bw_lanes = np.mean(right_fitx - left_fitx)
        if (np.absolute(dist_bw_lanes) < 10) and (left_fitx.size > 0) and (right_fitx.size > 0):
            l_line.add(left_fitx, ploty)
            r_line.add(right_fitx, ploty)
        else: # use previous best fit

            l_line.detected = False
            r_line.detected = False

        # get the center offset of the lanes
        l_radius, r_radius, center_offset = get_radius_curvature_center_offset(img_warped, left_lane_inds, right_lane_inds)

        # get weight based on the center offset
        rad_curvature_value = (l_radius + r_radius)/2
        img_result = get_weight(img2, rad_curvature_value, center_offset)

        #end = timer()
        #print("Time: ", end-start)

        cv2.namedWindow('Window', cv2.WINDOW_AUTOSIZE)
        cv2.setWindowProperty('Window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Window", img_result)



        if cv2.waitKey(1)&0xFF == ord('q'):
            break


# ----------------------------------------------------------------------------------------------------------------------#

video = cv2.VideoCapture(0)
if __name__ == '__main__':
    cart = mqtt.Client("cart")
    cart.on_connect = on_connect_cart
    cart.on_message = on_message_cart

    # Connect to MQTT broker
    try:
        cart.connect("163.180.117.195", 1883, 60)
    except:
        print("ERROR: Could not connect to MQTT")

    print("MQTT client start")
    cart.loop_start()

    lock = threading.Lock()
    # Creating thread for hallway detection
    t1 = threading.Thread(target=videoLineMeasurement_func, args=[lock])

    # Starting thread 1
    t1.start()

    # Wait until thread 1 is completely executed
    t1.join()

    cart.loop_stop()
    # Threads completely executed 
    print("All threads is done!")
