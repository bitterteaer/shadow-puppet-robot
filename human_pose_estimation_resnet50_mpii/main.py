import cv2
import paddlehub as hub
import serial
import math
# 导入sympy
from sympy import *


# ------------- 串口操作 --------------------- start
# 串口打开函数
def open_ser():
    port = '/dev/ttyUSB1'  # 串口号
    baudrate = 9600  # 波特率
    try:
        global ser
        ser = serial.Serial(port, baudrate, timeout=0.5)
        if (ser.isOpen() == True):
            print("串口打开成功")
    except Exception as exc:
        print("串口打开异常", exc)


# 数据发送
def send_msg(the_data):
    try:
        # send_datas = input("请输入要发送的数据\n")
        send_datas = str(the_data)

        ser.write(str(send_datas).encode("gbk"))
        print("已发送数据:", send_datas)
    except Exception as exc:
        print("发送异常", exc)


# 接收数据
def read_msg():
    try:
        print("等待接收数据")
        while True:
            data = ser.read(ser.in_waiting).decode('gbk')
            if data != '':
                break
        print("已接受到数据:", data)
    except Exception as exc:
        print("读取异常", exc)


# 关闭串口
def close_ser():
    try:
        ser.close()
        if ser.isOpen():
            print("串口未关闭")
        else:
            print("串口已关闭")
    except Exception as exc:
        print("串口关闭异常", exc)


# ------------- 串口操作 --------------------- end


# ---------- 关键点转换角度 --------------- start
def fun(y1, y2):
    return solve([y1, y2], ['x', 'y'])


def init(A, B, C):
    if A[0] - B[0] == 0:
        return -1
    k1 = (A[1] - B[1]) / (A[0] - B[0])
    k2 = -1 * k1
    b1 = A[1] - k1 * A[0]
    b2 = C[1] - k2 * C[0]

    x = Symbol('x')
    y = Symbol('y')
    y1 = k1 * x + b1 - y
    y2 = k2 * x + b2 - y

    return fun(y1, y2)


def to_angle(A, B, C):
    '''
        cosA = (b^2+c^2-a^2)/(2bc)
        A = arcos((b^2+c^2-a^2)/(2bc))
    '''
    a = ((B[0] - C[0]) ** 2 + (B[1] - C[1]) ** 2) ** 0.5
    b = ((A[0] - C[0]) ** 2 + (A[1] - C[1]) ** 2) ** 0.5
    c = ((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2) ** 0.5
    # print(a,b,c)
    if (2 * b * c) == 0:
        return -1
    resual = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
    if resual > 1:
        resual = 1
    elif resual < -1:
        resual = -1

    return math.degrees(math.acos(resual))


def to_left_shoulder(resual):
    left_elbow = resual['left_elbow']
    left_shoulder = resual['left_shoulder']
    A = left_shoulder
    B = left_elbow
    C = [left_shoulder[0], left_shoulder[1] + 100]
    the_data = to_angle(A, B, C)

    return the_data


def to_right_shoulder(resual):
    left_elbow = resual['right_elbow']
    left_shoulder = resual['right_shoulder']
    A = left_shoulder
    B = left_elbow
    C = [left_shoulder[0], left_shoulder[1] + 100]
    the_data = to_angle(A, B, C)

    return the_data


def to_right_elbow(resual):
    right_elbow = resual['right_elbow']
    right_shoulder = resual['right_shoulder']
    right_wrist = resual['right_wrist']
    A = right_shoulder
    B = right_elbow
    C = right_wrist
    # print(A,B,C)
    x = Symbol('x')
    y = Symbol('y')
    # print(init(A, B, C))
    if init(A, B, C) == -1:
        return 0
    if init(A, B, C) == []:
        return 0
    # print(init(A, B, C))
    B = [init(A, B, C)[x], init(A, B, C)[y]]
    the_data = to_angle(A, B, C)

    return the_data


def to_left_elbow(resual):
    right_elbow = resual['left_elbow']
    right_shoulder = resual['left_shoulder']
    right_wrist = resual['left_wrist']
    A = right_shoulder
    B = right_elbow
    C = right_wrist

    x = Symbol('x')
    y = Symbol('y')
    B = [init(A, B, C)[x], init(A, B, C)[y]]
    the_data = to_angle(A, B, C)

    return the_data


def to_right_foot(resual):
    left_elbow = resual['right_knee']
    left_shoulder = resual['right_hip']
    A = left_shoulder
    B = left_elbow
    C = [left_shoulder[0], left_shoulder[1] + 100]
    the_data = to_angle(A, B, C)

    return the_data


def to_left_foot(resual):
    left_elbow = resual['left_knee']
    left_shoulder = resual['left_hip']
    A = left_shoulder
    B = left_elbow
    C = [left_shoulder[0], left_shoulder[1] + 100]
    the_data = to_angle(A, B, C)

    return the_data


# ---------- 关键点转换角度 --------------- end


def out_points(img):
    '''
        全部肢体位置的键
        踝:'left_ankle','right_ankle'
        膝盖:'left_knee','right_knee'
        髋:'left_hip','right_hip'
        骨盆:'pelvis',胸部:'thorax',上颈:'upper_neck',头顶:'head_top'
        手腕:'right_wrist','left_wrist'
        肘:'right_elbow','left_elbow'
        肩膀:'right_shoulder','left_shoulder'
    '''
    pose_estimation = hub.Module(name="human_pose_estimation_resnet50_mpii")
    result = pose_estimation.keypoint_detection(images=[img], visualization=True)
    points = result[0]['data']
    return points


def standardization(data):
    data = str(int(data+0.5))
    if len(data) == 1:
        data = '00'+data
    elif len(data) == 2:
        data = '0' + data
    else:
        pass

    return data


if __name__ == "__main__":
    is_send = False  # 是否发送数据到stm32机器人

    if is_send is True:
        open_ser()  # 打开串口
        send_msg('a000b000c000d000e000f000Z')  # 为机器人复位

    capture = cv2.VideoCapture("test01.mp4")  # 读取视频, 0: 从摄像头

    count_frame = 0
    while capture.isOpened():
        ret, image = capture.read()

        if count_frame < 1:
            count_frame+=1
            continue
        count_frame = 0

        '''
        识别人体关键点坐标数据格式： 
        resual: OrderedDict([
            ('left_ankle', [243, 790]), 
            ('left_knee', [243, 700]),
            ('left_hip', [232, 580]), 
            ('right_hip', [294, 590]), 
            ('right_knee', [294, 710]),
            ('right_ankle', [289, 800]),
            ('pelvis', [260, 580]),
            ('thorax', [260, 430]), 
            ('upper_neck', [260, 400]), 
            ('head_top', [266, 280]),
            ('right_wrist', [136, 580]), 
            ('right_elbow', [175, 510]),
            ('right_shoulder', [209, 430]), 
            ('left_shoulder', [323, 430]),
            ('left_elbow', [334, 510]),
            ('left_wrist', [323, 590])
        ])
        '''
        resual = out_points(image)
        print(resual)

        '''
        # 关键点坐标转化为角度
        a_angle = standardization(to_left_elbow(resual))  # 肩膀到手肘方向与手肘到手腕方向上的夹角（左）
        b_angle = standardization(to_left_shoulder(resual))  # 肩膀到手肘方向与肩膀竖直方向上的夹角（左）
        c_angle = standardization(to_left_foot(resual))  # 盆骨到膝盖方向与盆骨竖直方向上的夹角（左）
        d_angle = standardization(to_right_foot(resual))  # 盆骨到膝盖方向与盆骨竖直方向上的夹角（右）
        e_angle = standardization(to_right_shoulder(resual))  # 肩膀到手肘方向与肩膀竖直方向上的夹角（右）
        f_angle = standardization(to_right_elbow(resual))  # 肩膀到手肘方向与手肘到手腕方向上的夹角（右）
        print("---------- send data -------------")
        print(f'a{a_angle}b{b_angle}c{c_angle}d{d_angle}e{e_angle}f{f_angle}Z')
        print("----------------------------------")

        if is_send is True:
            send_msg(f'a{a_angle}b{b_angle}c{c_angle}d{d_angle}e{e_angle}f{f_angle}Z')  # 发送数据到机器人
        '''

        cv2.imshow("image", cv2.resize(image, (720,480)))
        cv2.waitKey(1)

    if is_send is True:
        close_ser()  # 关闭串口通讯
    # 释放摄像头
    capture.release()
    # 关闭所有窗口
    cv2.destroyAllWindows()