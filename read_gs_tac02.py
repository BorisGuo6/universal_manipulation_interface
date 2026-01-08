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
import tf
import os, natsort, shutil
import random, paramiko

import time # To ring the bell every 0.5 seconds when there is an exception

from umi.common.cv_util import get_image_transform 

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
DURATION = 15


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
    for _id in range(4, total_ids, 2):
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

        assert self.tacniq_data != None

    def publish(self):
        img_msg = cvbridge.cv2_to_imgmsg(self.img, encoding="passthrough")
        img_msg.header.stamp = rospy.Time.now()
        img_msg.header.frame_id = 'map'
        self.pub.publish(img_msg)


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

    # run infinity loop
    recording_active = False
    while not rospy.is_shutdown():
        
        for gs in gss:
            try:
                gs.capture()
                gs.publish()
            except BaseException as e:
                for i in range(4):
                    print('\a')
                    time.sleep(0.5)
                print(e)
                raise
            print(gs.img.shape, gs.tacniq_data, time.time_ns())
            gelsight_ih, gelsight_iw = gs.img.shape
            
            resize_tf_gelsight = get_image_transform(
                in_res=(gelsight_iw, gelsight_ih),
                out_res=(224, 224),
            )
            gs.img = resize_tf_gelsight(gs.img)
            
            cv2.imshow('gsmini{}'.format(gs.src), gs.img)
            
        time.sleep(1) # for testing only
            
        if cv2.waitKey(1) == 27 :
            break
 
        r.sleep()


    # close all windows
    for gs in gss:
        gs.vs.stop()

    cv2.destroyAllWindows()
