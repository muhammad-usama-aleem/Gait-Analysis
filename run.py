import os
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

FRAME_END = 400
PIXELS_PER_METER = 200

mode = False
upBody = False
smooth = True
detectioncon = 0.5
trackcon = 0.5
pose = mp.solutions.pose.Pose(static_image_mode=mode,
                              smooth_segmentation=smooth,
                              min_detection_confidence=detectioncon,
                              min_tracking_confidence=trackcon)
FLOOR_LINE = 420
left_foot_index = 0
right_foot_index = 0
MIN_DIS = 5


def prepare_landmark_point(point, img):
    h, w = img.shape[:2]
    return (int(point.x*w), int(point.y*h))


def get_info_from(left_foot_heel, right_foot_heel):
    x1, y1 = left_foot_heel
    x2, y2 = right_foot_heel

    step_length = (abs(x1-x2))/PIXELS_PER_METER
    stride_length = 2*step_length
    stride_width = (abs(y1-y2))/PIXELS_PER_METER
    # print(step_length)
    return [step_length,
            stride_length,
            stride_width,
            ]


def findpose(img, draw=True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if not results or not results.pose_landmarks:
        return None

    h, w = img.shape[:2]
    landmarks = results.pose_landmarks.landmark
    left_foot_heel = prepare_landmark_point(landmarks[29], img)
    right_foot_heel = prepare_landmark_point(landmarks[30], img)

    cv2.line(img, (0, FLOOR_LINE), (w, FLOOR_LINE),
             (0, 0, 255), 2,)  # ! Y axis

    if results.pose_landmarks:
        if draw:
            mp.solutions.drawing_utils.draw_landmarks(
                img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

    return img, left_foot_heel, right_foot_heel


def findleg(img, draw=True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if not results or not results.pose_landmarks:
        return None
    h, w = img.shape[:2]
    landmarks = results.pose_landmarks.landmark
    # print('landmark', landmarks[23: 29])
    return (landmarks[24].x, landmarks[24].y), (landmarks[26].x, landmarks[26].y), (landmarks[28].x, landmarks[28].y)


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle

    return angle


def graph_plot(angles, name):
    fig = plt.figure()
    name = "Graph_" + str(name)
    angles = np.array(angles)
    plt.plot(angles)
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.savefig(name, dpi=720)
    plt.show()


src = "./video.mp4"  # ! for opening this video file
#src = 0  # ! for opening the camera
video = cv2.VideoCapture(src)

index = 0
fps = int(video.get(cv2.CAP_PROP_FPS))
num_of_sec_taken = 0
starting_foot_is_right = None
num_of_frames = 0

total_time = 0
num_times_happened = 0
first_starting_leg = None
start_move_left_heel, start_move_right_heel = 0, 0
angle = []
total_graphs_plotted = 0
while True:
    ok, frame = video.read()
    if not ok:
        break

    if len(os.path.splitext(str(src))) == 2:  # ! src is a file
        w,h = frame.shape[:2]
        limit = w // 2  + 170
        frame = frame[:, :limit]  # ! take left half the image
        #frame = frame[:, limit:]  # ! take right half the image

    data = findpose(frame)
    if data:
        img, left_foot_heel, right_foot_heel = data

    first, second, third = findleg(frame)
    angle.append(calculate_angle(first, second, third))
    print("ANGLES are", angle)
    cv2.imshow('img', img)

    right_distance = abs(right_foot_heel[1] - FLOOR_LINE)
    left_distance = abs(left_foot_heel[1] - FLOOR_LINE)
    # print(right_distance, left_distance)
    #! the person is standing on both legs
    if abs(left_foot_heel[0] - right_foot_heel[0]) <= MIN_DIS:
        num_of_frames += 1
        print('SKIP #1')
        continue

    if starting_foot_is_right is None:
        if right_distance <= MIN_DIS:
            starting_foot_is_right = True

        elif left_distance <= MIN_DIS:
            starting_foot_is_right = False

        first_starting_leg = starting_foot_is_right
        start_move_left_heel, start_move_right_heel = left_foot_heel, right_foot_heel

        #! to show the movement faster otherwise the frame won't change a lot
        # index += MIN_DIS
        # print('SKIP #2')
        # continue

    #  print(
    #      "\nright_distance:", right_distance,
    #      "\nleft_distance:", left_distance,
    #      "\nstarting_foot_is_right:", starting_foot_is_right,
    #      "\nright_foot_index:", right_foot_index,
    #      "\nleft_foot_index:", left_foot_index,
    #  )

    if starting_foot_is_right:
        left_foot_index += 1
        y_diff = abs(left_foot_heel[0] - start_move_left_heel[0])
        if left_distance <= MIN_DIS and y_diff > 10:  # ! the left foot has landed
            start_move_left_heel = left_foot_heel
            t = round(left_foot_index/fps, 2)
            print("Left Step Time:", t, "seconds")
            left_foot_index = 0
            total_time += t
            num_times_happened += 1
            # if starting_foot_is_right == first_starting_leg:
            # num_times_happened += 1
            #     print(num_times_happened)

            starting_foot_is_right = False
    else:
        right_foot_index += 1
        y_diff = abs(right_foot_heel[0] - start_move_right_heel[0])
        if right_distance <= MIN_DIS and y_diff > 10:  # ! the right foot has landed
            start_move_right_heel = right_foot_heel
            t = round(right_foot_index/fps, 2)
            print("Right Step Time:", t, "seconds")
            right_foot_index = 0

            total_time += t
            num_times_happened += 1
            # if starting_foot_is_right == first_starting_leg:
            #     num_times_happened += 1
            #     print(num_times_happened)

            starting_foot_is_right = True

    if num_times_happened == 2:
        print('Stride Time:', total_time, "seconds")
        num_times_happened = 0
        total_time = 0

    step_length, stride_length, stride_width = get_info_from(
        left_foot_heel, right_foot_heel)

    print(f"step_length: {step_length}",
          f"stride_length: {stride_length}",
          f"stride_width: {stride_width}",
          "="*20, "\n",
          sep="\n")

    # cv2.waitKey()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

graph_plot(angle, total_graphs_plotted)
angle = []
total_graphs_plotted += 1
