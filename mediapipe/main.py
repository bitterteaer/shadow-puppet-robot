# 未留接口，方便观看代码
import cv2
import mediapipe as mp
import time
import numpy as np
from extends import toLegal, zb_x, zb_y, angle, angle1, cut_angle
from chuankou import open_ser, send_msg

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("test01.mp4")

count_frame = 0
pTime = 0
i = 0

open_ser()  # 打开串口
while True:
    # read VideoCapture frame
    success, img = cap.read()

    # continue frame
    if count_frame < 1:
        count_frame += 1
        continue
    count_frame = 0

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        cx = results.pose_landmarks.landmark[1].x
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            # print(id, lm, "坐标")
            zb_x[id] = float(results.pose_landmarks.landmark[id].x)
            zb_y[id] = float(results.pose_landmarks.landmark[id].y)
            # zb_z[id] = float(results.pose_landmarks.landmark[id].z)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"frame:{i}  " + str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.imshow("Image", cv2.resize(img, (720,480)))
    cv2.waitKey(1)

    jd = np.zeros([6])
    jd[0] = angle(14, 12, 24)
    jd[1] = angle(16, 14, 12)
    jd[2] = angle(23, 11, 13)
    jd[3] = angle(11, 13, 15)
    jd[4] = angle1(26, 24)
    jd[5] = angle1(25, 23)
    # print(i, "i")
    # 数据合法化
    adata = 'a' + toLegal(jd[3])
    bdata = 'b' + toLegal(cut_angle(jd[2]))
    cdata = 'c' + toLegal(jd[5])
    ddata = 'd' + toLegal(jd[4])
    edata = 'e' + toLegal(cut_angle(jd[0]))
    fdata = 'f' + toLegal(jd[1])
    the_data_to_stm = adata + bdata + cdata + ddata + edata + fdata + 'Z'
    print("#------------- send data -----------------------")
    print("# i", i)
    # print('#', jd)
    print('# ' + the_data_to_stm)
    print("#-------------------------------------------------")
    send_msg(the_data_to_stm)
    # time.sleep(0.5)

    i += 1
