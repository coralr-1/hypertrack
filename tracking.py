#!/usr/bin/env python3
# encoding: utf-8
# [Ours Ultimate Version] LAB Main + Nearest Neighbor + Low-freq YOLO + CSRT Fusion + UKF Smoothing
import os
import cv2
import math
import rclpy
import threading
import numpy as np
import time
import csv
from datetime import datetime
import sdk.common as common
import sdk.pid as pid
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool, Trigger
from interfaces.srv import SetPoint, SetFloat64
from large_models_msgs.srv import SetString
from ros_robot_controller_msgs.msg import SetPWMServoState, PWMServoState, RGBState, RGBStates
from rclpy.callback_groups import ReentrantCallbackGroup
from ultralytics import YOLO

# ==========================================
#  New: Import Unscented Kalman Filter (UKF) related libraries
# ==========================================
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

class ObjectTrackingNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        self.is_running = False
        self.lock = threading.RLock()
        self.bridge = CvBridge()

        self.get_logger().info('\033[1;36m[Ours Ultimate Version] LAB+YOLO+CSRT Scheduling Mechanism + UKF Filter is ready!\033[0m')

        self.model = YOLO('/home/ubuntu/yolov8s.onnx', task='detect')
        self.target_locked = False
        self.click_point = None
        self.enable_ptz = False
        self.enable_chassis = False
        self.frame_count = 0
        self.csrt_tracker = None

        # LAB color tracking parameters
        self.pro_size = (320, 240)
        self.last_color_circle = None
        self.lost_target_count = 0

        #  Initialize UKF
        self.init_ukf()

        #  Slow speed servo
        self.servo_x_pid = pid.PID(P=0.15, I=0.03, D=0.005)
        self.servo_y_pid = pid.PID(P=0.15, I=0.03, D=0.005)

        self.servo_x = 1500
        self.servo_y = 1500
        self.servo_min_x = 800
        self.servo_max_x = 2200
        self.servo_min_y = 1200
        self.servo_max_y = 1900

        self.pan_tilt_x_threshold = 15
        self.pan_tilt_y_threshold = 15

        # Chassis PID
        self.pid_yaw = pid.PID(0.008, 0.003, 0.0001)
        self.pid_dist = pid.PID(0.004, 0.003, 0.00001)
        self.x_stop = 320
        self.y_stop = 400

        # PTZ angle compensation
        self.SERVO_CENTER = 1500
        self.SERVO_ANGLE_FACTOR = 0.012
        self.SERVO_OFFSET_THRESHOLD = 50

        self.last_linear_x = 0.0
        self.last_angular_z = 0.0

        # LAB color parameters
        self.target_lab = None
        self.threshold = 0.5

        # Anti-interference - Target position memory
        self.last_target_pos = None
        self.MAX_JUMP_DISTANCE = 150

        # CSRT parameters
        self.csrt_active = False
        self.csrt_lost_count = 0
        self.CSRT_TRIGGER_THRESHOLD = 5
        self.CSRT_MAX_DURATION = 20
        self.csrt_frames_active = 0

        # Lightweight - YOLO low frequency
        self.YOLO_INTERVAL = 40

        self.servo_pub = self.create_publisher(SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 10)
        self.rgb_pub = self.create_publisher(RGBStates, 'ros_robot_controller/set_rgb', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)

        self.enter_srv = self.create_service(Trigger, '~/enter', self.enter_srv_callback)
        self.exit_srv = self.create_service(Trigger, '~/exit', self.exit_srv_callback)
        self.set_pan_tilt_srv = self.create_service(SetBool, '~/set_pan_tilt', self.set_pan_tilt_callback)
        self.set_chassis_following_srv = self.create_service(SetBool, '~/set_chassis_following', self.set_chassis_following_callback)

        self.create_service(SetBool, '~/set_running', self.dummy_srv_callback)
        self.create_service(SetPoint, '~/set_target_color', self.dummy_srv_callback)
        self.create_service(Trigger, '~/get_target_color', self.dummy_srv_callback)
        self.create_service(SetFloat64, '~/set_threshold', self.dummy_srv_callback)
        self.create_service(SetString, '~/set_large_model_target_color', self.dummy_srv_callback)
        self.create_service(Trigger, '~/init_finish', self.dummy_srv_callback)

        self.callback_group = ReentrantCallbackGroup()
        self.image_sub = None
        self.window_created = False

        # Button exit control
        self.should_exit = False

        # CSV log
        self.csv_path = '/home/ubuntu/ros2_ws/src/app/ours_tracking_log.csv'
        self.csv_file = None
        self.csv_writer = None

        self.last_timestamp = None
        self.last_x = None
        self.last_y = None
        self.continuous_track_count = 0
        self.total_loss_count = 0

        self.get_logger().info(f'\033[1;33mPress ESC or q key to exit and save data at any time\033[0m')
        self.experiment_start_time = None

    def init_ukf(self):
        """ Initialize Unscented Kalman Filter (UKF)"""
        # Define state transition equation: assume constant velocity linear motion model
        def fx(x, dt):
            F = np.eye(8)
            F[0, 4] = dt  # x = x + vx*dt
            F[1, 5] = dt  # y = y + vy*dt
            F[2, 6] = dt  # w = w + vw*dt
            F[3, 7] = dt  # h = h + vh*dt
            return F @ x

        # Define observation equation: can only measure position, width, and height (first 4 states)
        def hx(x):
            return x[0:4]

        # Set Sigma points
        points = MerweScaledSigmaPoints(n=8, alpha=0.1, beta=2., kappa=-5)
        # Initialize filter (state dimension 8, measurement dimension 4)
        self.ukf = UKF(dim_x=8, dim_z=4, fx=fx, hx=hx, dt=0.04, points=points)
        self.ukf_initialized = False

    def get_center_lab(self, image, x, y, size=10):
        h, w = image.shape[:2]
        x1, y1 = max(0, int(x - size)), max(0, int(y - size))
        x2, y2 = min(w, int(x + size)), min(h, int(y + size))
        patch = image[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        lab_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        return np.median(lab_patch, axis=(0, 1))

    def track_by_color(self, cv_image):
        if self.target_lab is None:
            return None, 0, 0

        h, w = cv_image.shape[:2]
        image = cv2.resize(cv_image, self.pro_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        image = cv2.GaussianBlur(image, (5, 5), 5)

        min_color = [int(self.target_lab[0] - 50 * self.threshold * 2),
                     int(self.target_lab[1] - 50 * self.threshold),
                     int(self.target_lab[2] - 50 * self.threshold)]
        max_color = [int(self.target_lab[0] + 50 * self.threshold * 2),
                     int(self.target_lab[1] + 50 * self.threshold),
                     int(self.target_lab[2] + 50 * self.threshold)]

        mask = cv2.inRange(image, tuple(min_color), tuple(max_color))
        eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        contour_area = [(c, math.fabs(cv2.contourArea(c))) for c in contours]
        contour_area = [c for c in contour_area if c[1] > 40]

        candidates_count = len(contour_area)
        circle = None

        if len(contour_area) > 0:
            if self.last_target_pos is None:
                contour, area = max(contour_area, key=lambda c_a: c_a[1])
                circle = cv2.minEnclosingCircle(contour)
            else:
                last_x, last_y = self.last_target_pos
                last_x_scaled = last_x * self.pro_size[0] / w
                last_y_scaled = last_y * self.pro_size[1] / h

                circles = [cv2.minEnclosingCircle(c[0]) for c in contour_area]
                circle_dist = [(c, math.sqrt(((c[0][0] - last_x_scaled) ** 2) +
                                            ((c[0][1] - last_y_scaled) ** 2)))
                              for c in circles]
                circle, dist = min(circle_dist, key=lambda c: c[1])

                if dist >= 100:
                    contour, area = max(contour_area, key=lambda c_a: c_a[1])
                    circle = cv2.minEnclosingCircle(contour)

        if circle is not None:
            self.lost_target_count = 0
            (x, y), r = circle
            x = x / self.pro_size[0] * w
            y = y / self.pro_size[1] * h
            r = r / self.pro_size[0] * w

            self.last_target_pos = (x, y)
            return (x, y), r, candidates_count
        else:
            self.lost_target_count += 1
            if self.lost_target_count > 10:
                self.last_target_pos = None
            return None, 0, candidates_count

    def track_by_csrt(self, cv_image):
        if not self.csrt_active or self.csrt_tracker is None:
            return None, 0

        ok, bbox = self.csrt_tracker.update(cv_image)
        if ok:
            x, y, w, h = bbox
            if (20 < x+w/2 < 620 and 20 < y+h/2 < 460 and w > 10 and h > 10):
                return (x + w/2, y + h/2), max(w, h) / 2

        return None, 0

    def update_servo_pid(self, x, y, img_w, img_h):
        if abs(x - img_w / 2) < self.pan_tilt_x_threshold:
            x = img_w / 2
        if abs(y - img_h / 2) < self.pan_tilt_y_threshold:
            y = img_h / 2

        self.servo_x_pid.SetPoint = img_w / 2
        self.servo_x_pid.update(x)
        servo_x_output = int(self.servo_x_pid.output)
        servo_x_output = np.clip(servo_x_output, -30, 30)

        self.servo_x += servo_x_output
        self.servo_x = np.clip(self.servo_x, self.servo_min_x, self.servo_max_x)

        self.servo_y_pid.SetPoint = img_h / 2
        self.servo_y_pid.update(y)
        servo_y_output = int(self.servo_y_pid.output)
        servo_y_output = np.clip(servo_y_output, -30, 30)

        self.servo_y -= servo_y_output
        self.servo_y = np.clip(self.servo_y, self.servo_min_y, self.servo_max_y)

        return self.servo_x, self.servo_y

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_point = (x, y)
            self.target_locked = False

    def dummy_srv_callback(self, request, response):
        response.success = True
        return response

    def set_pan_tilt_callback(self, request, response):
        self.enable_ptz = request.data
        if not self.enable_ptz:
            self.publish_servo(1500, 1500)
        response.success = True
        return response

    def set_chassis_following_callback(self, request, response):
        self.enable_chassis = request.data
        if not self.enable_chassis:
            self.send_twist(0.0, 0.0, 0.0)
            self.pid_yaw.clear()
            self.pid_dist.clear()
            self.last_linear_x = 0.0
            self.last_angular_z = 0.0
        else:
            self.get_logger().info('\033[1;32mSmart following enabled (Anti-interference + PTZ synergy)\033[0m')
        response.success = True
        return response

    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32mUltimate anti-interference mode ready, please click the target!\033[0m')
        self.get_logger().info('\033[1;33mPress ESC or q key to exit at any time\033[0m')

        with self.lock:
            self.is_running = True
            self.should_exit = False
            self.target_locked = False
            self.click_point = None
            self.target_lab = None
            self.enable_ptz = True
            self.enable_chassis = False
            self.servo_x = 1500
            self.servo_y = 1500

            self.servo_x_pid.clear()
            self.servo_y_pid.clear()
            self.pid_yaw.clear()
            self.pid_dist.clear()

            self.publish_servo(1500, 1500)
            self.last_linear_x = 0.0
            self.last_angular_z = 0.0
            self.lost_target_count = 0
            self.last_target_pos = None

            self.csrt_active = False
            self.csrt_tracker = None
            self.csrt_lost_count = 0
            self.csrt_frames_active = 0

            #  State reset UKF
            self.ukf_initialized = False

            self.experiment_start_time = time.time()
            self.last_timestamp = None
            self.last_x = None
            self.last_y = None
            self.continuous_track_count = 0
            self.total_loss_count = 0

            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                'Frame', 'Timestamp', 'Time_sec',
                'X', 'Y', 'Width', 'Height',
                'dT', 'dX', 'dY', 'Velocity',
                'Tracking_Mode', 'Is_Lost', 'Continuous_Count', 'Total_Loss_Count',
                'Error_X', 'Error_Y',
                'Servo_X', 'Servo_Y', 'Servo_Offset',
                'Chassis_Linear_X', 'Chassis_Angular_Z', 'Chassis_Compensation',
                'Frame_Time_ms', 'FPS', 'CSRT_Active', 'Candidates_Count'
            ])

        if self.image_sub is None:
            self.image_sub = self.create_subscription(Image, 'image_raw', self.image_callback, 1, callback_group=self.callback_group)
        response.success = True
        return response

    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32mExit tracking mode\033[0m')
        self.cleanup_and_exit()
        response.success = True
        return response

    def cleanup_and_exit(self):
        with self.lock:
            self.is_running = False
            self.target_locked = False

            self.send_twist(0.0, 0.0, 0.0)
            self.pid_yaw.clear()
            self.pid_dist.clear()
            self.last_linear_x = 0.0
            self.last_angular_z = 0.0

            self.publish_servo(1500, 1500)
            self.servo_x = 1500
            self.servo_y = 1500

            self.servo_x_pid.clear()
            self.servo_y_pid.clear()

            self.publish_rgb(0, 0, 0)

            self.csrt_active = False
            self.csrt_tracker = None
            self.ukf_initialized = False

            if self.csv_file is not None:
                self.csv_file.close()
                self.csv_file = None
                self.csv_writer = None
                self.get_logger().info(f'\033[1;32mCSV data saved: {self.csv_path}\033[0m')

        if self.image_sub is not None:
            self.destroy_subscription(self.image_sub)
            self.image_sub = None

    def publish_servo(self, servo_x, servo_y):
        msg = SetPWMServoState()
        state_x = PWMServoState()
        state_x.id = [2]
        state_x.position = [int(servo_x)]
        state_x.offset = [0]
        state_y = PWMServoState()
        state_y.id = [1]
        state_y.position = [int(servo_y)]
        state_y.offset = [0]
        msg.state = [state_x, state_y]
        msg.duration = 0.02
        self.servo_pub.publish(msg)

    def send_twist(self, linear_x, linear_y, angular_z):
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = float(linear_x), float(linear_y), float(angular_z)
        self.cmd_vel_pub.publish(t)

    def publish_rgb(self, r, g, b):
        msg = RGBStates()
        msg.states = [RGBState(index=1, red=r, green=g, blue=b), RGBState(index=2, red=r, green=g, blue=b)]
        self.rgb_pub.publish(msg)

    def smooth_value(self, current, last, factor=0.5):
        return last * (1 - factor) + current * factor

    def control_chassis(self, x, y, servo_x):
        yaw_error = x - self.x_stop
        dist_error = y - self.y_stop

        self.pid_yaw.update(yaw_error)
        self.pid_dist.update(dist_error)

        angular_z = common.set_range(self.pid_yaw.output, -4.0, 4.0)
        linear_x = common.set_range(self.pid_dist.output, -1.0, 1.0)

        servo_offset = servo_x - self.SERVO_CENTER
        chassis_compensation = 0.0

        if abs(servo_offset) > self.SERVO_OFFSET_THRESHOLD:
            chassis_compensation = servo_offset * self.SERVO_ANGLE_FACTOR
            chassis_compensation = common.set_range(chassis_compensation, -2.0, 2.0)

            angular_z += chassis_compensation
            angular_z = common.set_range(angular_z, -4.0, 4.0)

            if abs(chassis_compensation) > 0.5:
                linear_x *= 0.7

        if abs(linear_x) < 0.05:
            linear_x = 0.0
        if abs(angular_z) < 0.1:
            angular_z = 0.0

        linear_x = self.smooth_value(linear_x, self.last_linear_x, 0.5)
        angular_z = self.smooth_value(angular_z, self.last_angular_z, 0.5)

        self.last_linear_x = linear_x
        self.last_angular_z = angular_z

        return linear_x, angular_z, chassis_compensation

    def image_callback(self, ros_image):
        if not self.is_running:
            return

        if self.should_exit:
            self.cleanup_and_exit()
            return

        frame_start_time = time.time()

        if not self.window_created:
            cv2.namedWindow('tracking_debug', cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback('tracking_debug', self.on_mouse)
            self.window_created = True

        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        result_image = np.copy(cv_image)
        h, w = cv_image.shape[:2]

        with self.lock:
            self.frame_count += 1

            # YOLO low frequency running
            run_yolo = (not self.target_locked) or (self.frame_count % self.YOLO_INTERVAL == 0)
            yolo_boxes = []
            if run_yolo:
                results = self.model.predict(cv_image, classes=[32, 41, 44, 47, 64], conf=0.25, imgsz=640, verbose=False)
                yolo_boxes = [box.xywh[0].cpu().numpy() for box in results[0].boxes] if results[0].boxes else []

            if not self.target_locked:
                for box in yolo_boxes:
                    xc, yc, bw, bh = box
                    cv2.rectangle(result_image, (int(xc-bw/2), int(yc-bh/2)), (int(xc+bw/2), int(yc+bh/2)), (255, 150, 100), 1)

            # Initialize target
            if self.click_point is not None and not self.target_locked:
                cx, cy = self.click_point
                self.target_lab = self.get_center_lab(cv_image, cx, cy)

                if self.target_lab is not None:
                    self.target_locked = True
                    self.lost_target_count = 0
                    self.last_target_pos = (cx, cy)

                    self.get_logger().info(f'Target locked, LAB=[{self.target_lab[0]:.0f}, {self.target_lab[1]:.0f}, {self.target_lab[2]:.0f}]')
                    self.publish_rgb(0, 255, 0)

                    #  [UKF] Initialize covariance and initial state
                    self.ukf.x = np.array([cx, cy, 60, 60, 0, 0, 0, 0])
                    self.ukf_initialized = True

                self.click_point = None

            target_pos = None
            target_radius = 0
            tracking_mode = "NONE"
            candidates_count = 0

            if self.target_locked:
                # 1. LAB tracking
                lab_pos, lab_radius, candidates_count = self.track_by_color(cv_image)

                if lab_pos is not None:
                    target_pos, target_radius = lab_pos, lab_radius
                    tracking_mode = "LAB"
                    self.csrt_lost_count = 0

                    # YOLO auxiliary calibration
                    if run_yolo and len(yolo_boxes) > 0:
                        tx, ty = target_pos
                        min_dist = 9999
                        best_yolo = None
                        for yb in yolo_boxes:
                            dist = math.hypot(yb[0]-tx, yb[1]-ty)
                            if dist < 100 and dist < min_dist:
                                min_dist = dist
                                best_yolo = yb

                        if best_yolo is not None:
                            target_pos = (best_yolo[0], best_yolo[1])
                            target_radius = max(best_yolo[2], best_yolo[3]) / 2
                            tracking_mode = "YOLO"
                else:
                    self.csrt_lost_count += 1

                # 2. CSRT assist
                if self.csrt_lost_count >= self.CSRT_TRIGGER_THRESHOLD and not self.csrt_active:
                    self.get_logger().info('LAB lost, CSRT intervened')
                    self.csrt_active = True
                    self.csrt_frames_active = 0

                    if self.last_target_pos is not None:
                        x, y = self.last_target_pos
                        w_csrt = h_csrt = 80
                        bbox_csrt = (int(x-w_csrt/2), int(y-h_csrt/2), int(w_csrt), int(h_csrt))

                        try: self.csrt_tracker = cv2.TrackerCSRT_create()
                        except: self.csrt_tracker = cv2.legacy.TrackerCSRT_create()
                        self.csrt_tracker.init(cv_image, bbox_csrt)

                if self.csrt_active:
                    csrt_pos, csrt_radius = self.track_by_csrt(cv_image)
                    self.csrt_frames_active += 1

                    if csrt_pos is not None:
                        if target_pos is not None:
                            self.get_logger().info('LAB recovered, CSRT exited')
                            self.csrt_active = False
                            self.csrt_tracker = None
                            self.csrt_frames_active = 0
                        else:
                            target_pos, target_radius = csrt_pos, csrt_radius
                            tracking_mode = "CSRT"

                    if self.csrt_frames_active > self.CSRT_MAX_DURATION:
                        self.get_logger().info('CSRT timeout, switch back to LAB')
                        self.csrt_active = False
                        self.csrt_tracker = None
                        self.csrt_frames_active = 0

            # Data processing and control
            dT, dX, dY, velocity = 0.0, 0.0, 0.0, 0.0
            is_lost = 1
            error_x, error_y = 0.0, 0.0
            linear_x, angular_z = 0.0, 0.0
            chassis_compensation = 0.0
            servo_offset = 0.0

            final_x, final_y = 0.0, 0.0
            final_w, final_h = 0.0, 0.0

            if target_pos is not None:
                is_lost = 0
                self.continuous_track_count += 1

                tx, ty = target_pos
                tw = th = target_radius * 2

                # ==========================================
                #  UKF Core Workspace: Predict and smooth coordinates based on current measurements!
                # ==========================================
                if self.ukf_initialized:
                    self.ukf.predict()
                    self.ukf.update(np.array([tx, ty, tw, th]))

                    # Pass the absolute smoothed coordinates filtered by UKF to the underlying PTZ and chassis!
                    final_x, final_y = self.ukf.x[0], self.ukf.x[1]
                    final_w, final_h = self.ukf.x[2], self.ukf.x[3]
                else:
                    final_x, final_y = tx, ty
                    final_w, final_h = tw, th

                current_timestamp = time.time()
                if self.last_timestamp is not None and self.last_x is not None:
                    dT = current_timestamp - self.last_timestamp
                    dX = final_x - self.last_x
                    dY = final_y - self.last_y
                    if dT > 0:
                        velocity = math.hypot(dX, dY) / dT

                self.last_timestamp = current_timestamp
                self.last_x = final_x
                self.last_y = final_y

                # Draw tracking box
                if tracking_mode == "LAB": color = (0, 255, 0)
                elif tracking_mode == "CSRT": color = (255, 165, 0)
                elif tracking_mode == "YOLO": color = (0, 0, 255)
                else: color = (128, 128, 128)

                # Draw the final filtered circle box
                cv2.circle(result_image, (int(final_x), int(final_y)), int(final_w/2), color, 2)
                cv2.circle(result_image, (int(final_x), int(final_y)), 5, (0, 0, 255), -1)

                error_x = final_x - w/2
                error_y = final_y - h/2

                # PTZ control
                if self.enable_ptz:
                    self.update_servo_pid(final_x, final_y, w, h)
                    self.publish_servo(self.servo_x, self.servo_y)
                    servo_offset = self.servo_x - self.SERVO_CENTER

                # Chassis control
                if self.enable_chassis:
                    linear_x, angular_z, chassis_compensation = self.control_chassis(final_x, final_y, self.servo_x)
                    self.send_twist(linear_x, 0.0, angular_z)

                # HUD display
                status = f"UKF | {tracking_mode}"
                if self.csrt_active: status += " [CSRT]"
                if abs(servo_offset) > self.SERVO_OFFSET_THRESHOLD: status += f" [Comp{chassis_compensation:.2f}]"
                if candidates_count > 1: status += f" [Interference{candidates_count-1}]"

                cv2.putText(result_image, f"{status} | FPS: {1.0/(time.time()-frame_start_time):.1f}",
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(result_image, f"Servo: X={self.servo_x} Y={self.servo_y} | Press ESC/q to exit",
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(result_image, f"Base: Vx={linear_x:.2f} Wz={angular_z:.2f}",
                           (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            else:
                # Target lost
                if self.target_locked and self.lost_target_count > 10:
                    is_lost = 1
                    self.total_loss_count += 1
                    self.continuous_track_count = 0
                    self.target_locked = False
                    self.publish_rgb(255, 0, 0)

                    if self.enable_chassis:
                        self.send_twist(0.0, 0.0, 0.0)
                        self.pid_yaw.clear()
                        self.pid_dist.clear()
                        self.last_linear_x = 0.0
                        self.last_angular_z = 0.0

                    self.csrt_active = False
                    self.csrt_tracker = None
                    self.ukf_initialized = False # Turn off UKF prediction when unlocked

            # CSV log (real-time write)
            if self.frame_count % 10 == 0 and self.csv_writer is not None:
                frame_end_time = time.time()
                frame_time_ms = (frame_end_time - frame_start_time) * 1000
                fps = 1.0 / (frame_end_time - frame_start_time) if (frame_end_time - frame_start_time) > 0 else 0.0
                time_sec = frame_end_time - self.experiment_start_time if self.experiment_start_time else 0.0

                self.csv_writer.writerow([
                    self.frame_count,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    round(time_sec, 4),
                    round(final_x, 2), round(final_y, 2),
                    round(final_w, 2), round(final_h, 2),
                    round(dT, 4), round(dX, 2), round(dY, 2), round(velocity, 2),
                    tracking_mode, is_lost,
                    self.continuous_track_count, self.total_loss_count,
                    round(error_x, 2), round(error_y, 2),
                    int(self.servo_x), int(self.servo_y), round(servo_offset, 2),
                    round(linear_x, 3), round(angular_z, 3), round(chassis_compensation, 3),
                    round(frame_time_ms, 2), round(fps, 2), int(self.csrt_active),
                    candidates_count
                ])
                # Flush to disk immediately
                self.csv_file.flush()

        cv2.imshow('tracking_debug', result_image)

        # Key detection: ESC or q to exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            self.get_logger().info('\033[1;33mExit key detected, saving data...\033[0m')
            self.should_exit = True

        self.result_publisher.publish(self.bridge.cv2_to_imgmsg(result_image, "rgb8"))

def main():
    node = ObjectTrackingNode('object_tracking')
    try:
        rclpy.spin(node)
    finally:
        node.send_twist(0.0, 0.0, 0.0)
        node.publish_servo(1500, 1500)
        if node.csv_file is not None:
            node.csv_file.close()
            node.get_logger().info('\033[1;32mData saved\033[0m')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()