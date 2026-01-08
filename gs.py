#!/usr/bin/env python3

import cv2
import numpy as np
from threading import Thread, Lock
import rospy
from std_msgs.msg import String,Int16MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# from markertracker import MarkerTracker
from tqdm import tqdm
import os, shutil
import random

import time # To ring the bell every 0.5 seconds when there is an exception

CAMERA_CAPTURE_FREQUENCY = 25 # tested, do not increase!


# OPTICAL FLOW CONFIG
COMPUTE_OF = False # compute optical flow
WIDTH = 720
HEIGHT = 960
RESIZED_FOR_OF = True # set true for optical flow

# For different tasks, record for different durations
# Task 2: 7 seconds
# Task 3: 12 seconds
# Task 4: 28 seconds
# Task 7: 32 seconds
# Task 8: 22 seconds
# Task 9: correct side first try: 15 seconds, wrong side first try: 30 seconds
# Task 10: 20 seconds
DURATION = 10


# # HIGH QUALITY IMAGE CONFIG
# COMPUTE_OF = False
# WIDTH = 2464
# HEIGHT = 3280
# RESIZED_FOR_OF = False # set true for optical flow


cvbridge = CvBridge()

def search_for_devices(total_ids = 6):
    '''
    Found gelsights for the sensor.
    '''

    found_ids = []

    print("Looking for gelsight devices")
    for _id in range(0, total_ids, 2):
        try:
            cap = cv2.VideoCapture(_id)
            ret, _ = cap.read()
            if ret:
                found_ids.append(_id)
                cap.release()
            else:
                break
        except:
            pass
    
    found_ids.remove(0)
    print('Found {} gelsight sensors.'.format(len(found_ids)))

    return found_ids

def resize_crop_mini(img, imgw, imgh):
    # remove 1/7th of border from each size
    border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(np.floor(img.shape[1] * (1 / 7)))
    # keep the ratio the same as the original image size
    img = img[border_size_x+2:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
    # final resize for 3d
    img = cv2.resize(img, (imgw, imgh))
    return img


def extract_span(index_number, threshold, min_len, max_len, top_frame_num):

    def find_longest_spans(arr):
        # Find the maximum length by traversing the array
        max_count = 0
        second_max_count = 0
        span, indices, max_indices, second_max_indices = [], [], [], []
        count = 0
        for i in range(1, len(arr)):
            # Check if the current element is equal to previous element +1
            frame_id = int(arr[i].split("/")[-1].split(".")[0])
            prev_frame_id = int(arr[i-1].split("/")[-1].split(".")[0])
            if frame_id == prev_frame_id + 1:
                if count == 0:
                    span.append(arr[i-1])
                    count += 1
                    # indices.append(i-1)
                count += 1
                span.append(arr[i])
                # indices.append(i)
            # Reset the count
            else:
                # Update the maximum
                if count > max_count:
                    max_count = count
                    max_span = span
                    # max_indices = indices
                elif count > second_max_count:
                    second_max_count = count
                    second_max_span = span
                    # second_max_indices = indices
                span = []
                count = 0
                # indices = []
        try:
            return max_span, max_indices, second_max_span, second_max_indices
        except UnboundLocalError:
            pass
        try:
            return max_span, max_indices, None, None
        except UnboundLocalError:
            # If there is no continuous span, get a random frame
            idx = random.randrange(0, len(arr))
            max_span = [arr[idx]]
            max_indices = [idx]
            return max_span, max_indices, None, None

    sample_path = 'demo/vid/{}.mov'.format(index_number)  ###TODO
    sample_frames = natsort.natsorted(os.path.join(sample_path, i) for i in os.listdir(sample_path))
    # 1) Get a certain number of frames above a certain change threshold
    prev_frame_img = cv2.imread(sample_frames[0])
    prev_frame_gray = cv2.cvtColor(prev_frame_img, cv2.COLOR_BGR2GRAY)
    all_diffs = []
    for frame in sample_frames[1:]:
        # frame_id = int(frame.split("/")[-1].split(".")[0])
        frame_img = cv2.imread(frame)
        frame_gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
        try:
            frame_diff = cv2.absdiff(prev_frame_gray, frame_gray)
        except cv2.error:
            break
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        total_diff = np.sum(thresh)
        all_diffs.append((frame, total_diff))
        prev_frame_gray = frame_gray
    all_diffs = sorted(all_diffs, key=lambda t: t[1], reverse=True)[:top_frame_num]
    all_diffs = sorted(all_diffs, key=lambda t: t[0])
    all_diffs = [i[0] for i in all_diffs]
    # Check for minimum length
    if len(all_diffs) < min_len:
        shutil.rmtree(sample_path)
    # 2) Get continuous spans
    max_span, max_indices, second_max_span, second_max_indices = find_longest_spans(all_diffs)
    if second_max_indices is not None:
        final_span = natsort.natsorted(max_span + second_max_span)
        final_indices = natsort.natsorted(max_indices + second_max_indices)
    else:
        final_span = max_span
        final_indices = max_indices
    # Check for minimum length
    if len(final_span) < min_len:
        shutil.rmtree(sample_path)
    if len(final_span) > max_len:
        final_span = final_span[:max_len]
    # 3) Remove frames that are not in the final span
    for frame in sample_frames:
        if frame not in final_span:
            os.remove(frame)


# def test(self):
#
#     !scp -r $sample_path samson@naga.d2.comp.nus.edu.sg:/home/samson/octopi-v2/data/demo_videos/$demo_name

class WebcamVideoStream :
    '''
    Thread based visualization
    '''
    def __init__(self, src, width = 320, height = 240) :
        self.stream = cv2.VideoCapture(src)
        # self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
        # self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        assert self.grabbed != False, "Camera with src={} is not found".format(src)

        self.started = False
        self.read_lock = Lock()

        self.connection_lost = False
        self.src = src

    def start(self) :
        if self.started :
            print ("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        consecutive_frames_lost = 0
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame

            if self.frame is None:
                consecutive_frames_lost += 1
            else:
                consecutive_frames_lost = 0
            if consecutive_frames_lost >= 3:
                self.connection_lost = True
                print("lost connection with camera, src={}".format(self.src))
                print("Shutting down")
                self.read_lock.release()
                raise Exception("Connection lost")
                exit()
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()

class GS:
    def __init__(self, src, width=240, height=320, resized_for_OF=True):
        self.src = src
        self.pub = rospy.Publisher("/gsmini_rawimg_{}".format(src), Image, queue_size=1)
        self.sub = rospy.Subscriber("/gsmini_command", String, self.process_data)
        self.vs = WebcamVideoStream(src=src).start()
        self.width = width
        self.height = height
        self.trial = 0
        self.object_class = "default"
        self.frames = 0
        self.recording_active = False
        self.base_image = None
        self.action = "press"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out_record = cv2.VideoWriter("./default.mov", fourcc, CAMERA_CAPTURE_FREQUENCY, (self.height, self.width), isColor=True)
        self.img = None
        self.OF = None
        self.resized_for_OF = resized_for_OF
        self.tacniq_listener = rospy.Subscriber("tacniq/right", Int16MultiArray, queue_size=1,
                                                callback=self.tacniqCallback)
        self.image_number = 0
        self.in_demo = False

        # flash out black pixels at the beginning
        self.flash_out_size=20
        self.initialize()

    def tacniqCallback(self, msg):
        self.tacniq_data = msg.data
    def process_data(self, msg):
        print("Receiving data {}".format(msg.data))
        command = msg.data
        if command[0] == 'r' and command != 'reset':
            self.object_class = command[2:]
            self.trial = 0
            self.frames = 0
            self.action = "press"

        if command[0] == 'p':
            self.action = 'press'
            print("Changing to pressing mode")
            self.trial = 0
            self.frames = 0
        if command[0] == 's':
            self.action = "slide"
            print("Changing to sliding mode")
            self.trial = 0
            self.frames = 0
        if command[0] == "t":
            self.action = "twist"
            print("Changing to twisting mode")
            self.trial = 0
            self.frames = 0
        if command[0] == 'k':
            self.trial = int(command[2:])
            print("reset to trial " + str(self.trial))

        if command[0] == 'c':
            base_image = self.vs.read()
            if self.resized_for_OF:
                self.base_image = resize_crop_mini(base_image, self.height, self.width)
            else:
                self.base_image = cv2.resize(base_image, (self.width, self.height))
            file_path_l = './data_{}/{}_{}.mov'.format(self.action, self.object_class, self.trial+1)
            print(file_path_l)
            if self.in_demo:
                file_path_l = './demo/{}/item.mov'.format(self.object_class)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out_record = cv2.VideoWriter(file_path_l, fourcc, CAMERA_CAPTURE_FREQUENCY, (self.height, self.width), isColor=True)
            self.pbar = tqdm(total=CAMERA_CAPTURE_FREQUENCY * DURATION)
            self.recording_active = True
            self.trial += 1
        
        ''' # This part is too buggy
        if command[0] == 'b': # b for berhenti, stop in Malay
            self.recording_active = False
            self.frames = 0
            self.out_record = None
            print("Stopping recording for trial " + str(self.trial))
        '''

        if command == "take pic":
            bridge = CvBridge()
            rgb_msg = rospy.wait_for_message('/camera/color/image_raw', Image, timeout=5)
            rgb_img = np.array(bridge.imgmsg_to_cv2(rgb_msg))
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

            cv2.imwrite("{}_{}.png".format(self.object_class, self.image_number), rgb_img)

            rgbd_msg = rospy.wait_for_message('/camera/depth/image_raw', Image, timeout=5)
            rgbd_img = np.array(bridge.imgmsg_to_cv2(rgbd_msg))
            cv2.imwrite("{}_{}_depth.png".format(self.object_class, self.image_number), rgbd_img)

            self.image_number += 1

            if self.in_demo:
                cv2.imwrite("rgb.png", rgb_img)
                sftp = self.ssh.open_sftp()
                target = "/home/samson/octopi-v2/data/demo_videos/demo"
                item = "rgb.png"
                sftp.put("rgb.png", '%s/%s' % (target, item))
                print("uploaded pic.")


    def initialize(self):
        for i in range(self.flash_out_size):
            self.vs.read()

    def capture(self):
        img = self.vs.read()
        if self.resized_for_OF:
            self.img = resize_crop_mini(img,self.height,self.width)
        else:
            self.img = cv2.resize(img, (self.width, self.height))
        if self.recording_active:
            self.out_record.write(self.img)
            self.frames += 1
            self.pbar.update(1)
            ## write tacniq file here
            with open('tacniq_data/{}_{}.txt'.format(self.object_class, self.trial), 'w' if self.frames == 1 else 'a') as f:
                f.write(str(rospy.Time.now().to_nsec()) + ";" + str(self.tacniq_data) + '\n')
            if self.frames > CAMERA_CAPTURE_FREQUENCY * DURATION:
                self.recording_active = False
                self.frames = 0
                self.out_record = None
                print("Stopping recording for trial " + str(self.trial))

                if self.in_demo:
                    #### perform SCP and other things here.
                    sftp = self.ssh.open_sftp()
                    source = "./demo"
                    target = "/home/samson/octopi-v2/data/demo_videos/demo"
                    item = "{}/item.mov".format(self.object_class)
                    sftp.put(os.path.join(source, item), '%s/%s' % (target, item))

                # difference = np.abs(np.mean((self.img.astype(float))/255 - (self.base_image.astype(float))/255))
                # if difference > 0.1:
                #     print(difference)
                #     print("Warning: Gelsight images are at least 10% different")
                # elif difference > 0:
                #     print(difference)

        


    def publish(self):
        img_msg = cvbridge.cv2_to_imgmsg(self.img, encoding="passthrough")
        img_msg.header.stamp = rospy.Time.now()
        img_msg.header.frame_id = 'map'
        self.pub.publish(img_msg)

'''
class OpticalFlowDetector:
    def __init__(self, id=0):
        self.id = id
        self.initialized = False

        self.pre_grey_img = None
        self.curr_grey_img = None
        self.p0 = None
        self.nct = None
        self.Ox = None
        self.Oy = None
        # self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def init(self, img):
        
        # Initialize the markers in the beginning, quite slow process. Takes up to 5 seconds to compute
        
        if not self.initialized:

            if self.pre_grey_img is None:
                self.pre_grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img = np.float32(img) / 255.0

            mtracker = MarkerTracker(img)
            marker_centers = mtracker.initial_marker_center
            Ox = marker_centers[:, 1]
            Oy = marker_centers[:, 0]
            nct = len(marker_centers)
            print("Initial markers are detected")

            

            # Existing p0 array
            p0 = np.array([[Ox[0], Oy[0]]], np.float32).reshape(-1, 1, 2)
            for i in range(nct - 1):
                # New point to be added
                new_point = np.array([[Ox[i+1], Oy[i+1]]], np.float32).reshape(-1, 1, 2)
                # Append new point to p0
                p0 = np.append(p0, new_point, axis=0)

            self.p0 = p0
            self.nct = nct
            self.Ox = Ox
            self.Oy = Oy
            self.initialized = True



    def update(self, img):

        if not self.initialized:
            self.init(img)
            return

        self.curr_grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(self.pre_grey_img, self.curr_grey_img, self.p0, None, **self.lk_params)
        
        # Select good points
        good_new = p1[st == 1]
        good_old = self.p0[st == 1]

        if len(good_new) < self.nct:
            # Detect new features in the current frame
            print("all pts did not converge")
        else:
            # Update points for next iteration
            self.p0 = good_new.reshape(-1, 1, 2)
        
        self.pre_grey_img = self.curr_grey_img.copy()


        # Draw the tracks
        flow_lines = {
            'start': [],
            'end': [],
            'size': len(good_new)
        }
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            ix = int(self.Ox[i])
            iy = int(self.Oy[i])

            flow_lines['start'].append((ix, iy))
            flow_lines['end'].append((int(a),int(b)))

        return flow_lines
'''

if __name__ == '__main__':

    rospy.init_node('GS', anonymous=True)

    gs_ids = search_for_devices()

    r = rospy.Rate(CAMERA_CAPTURE_FREQUENCY)
    NUM_SENSORS = len(gs_ids)
    color = np.random.randint(0, 255, (100, 3))

    if NUM_SENSORS == 0:
        print('No gelsight sensor is found! Exiting ...')
        exit()
    # run sensors
    gss = []
    for src in gs_ids:
        gss.append(GS(src, width=WIDTH, height=HEIGHT))

    
    # intialize the optical flow
    if COMPUTE_OF:
        print('Initializing Optical Flow ... ')
        for gs in gss:
            of = OpticalFlowDetector(id=gs.src)
            gs.capture()
            of.init(gs.img)
            gs.OF = of
        print('Done.')

    # run infinity loop
    recording_active = False
    while not rospy.is_shutdown():
        
        for gs in gss:
            try:
                gs.capture()
                gs.publish()
            except:
                for i in range(24):
                    print('\a')
                    time.sleep(0.5)
                raise
            if COMPUTE_OF:
                # calculate optical flow - MUST be fast!
                flow_lines = gs.OF.update(gs.img)
                
                for i in range(flow_lines['size']):
                    offrame = cv2.arrowedLine(gs.img, flow_lines['start'][i], flow_lines['end'][i], (255,255,255), thickness=1, line_type=cv2.LINE_8, tipLength=.15)
                    # offrame = cv2.circle(offrame, flow_lines['end'][i], 5, color[i].tolist(), -1)
                cv2.imshow('gsmini{}_OF'.format(gs.src), offrame)
                #cv2.imshow('gsmini{}_OF'.format(gs.src), gs.OF.curr_grey_img)
            else:
                cv2.imshow('gsmini{}'.format(gs.src), gs.img)
            
        if cv2.waitKey(1) == 27 :
            break
 
        r.sleep()


    # close all windows
    for gs in gss:
        gs.vs.stop()

    cv2.destroyAllWindows()
