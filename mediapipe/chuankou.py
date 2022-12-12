from email.mime import base
import serial

# ----------------------------- 串口操作 ---------------------------------------------
# 串口打开函数
def open_ser():
    port = 'com1'  # 串口号
    baudrate = 9600  # 波特率
    try:
        global ser
        # ser = serial.Serial(port,baudrate,timeout=0.5)
        ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.5)  # 使用USB连接串行口
        if (ser.isOpen() == True):
            print("串口打开成功")
    except Exception as exc:
        print("串口打开异常", exc)


# 数据发送
def send_msg(data):
    try:
        # send_datas = input("请输入要发送的数据\n")
        send_datas = str(data)
        # ser.write(strsend_datas).encode("gbk")
        ser.write(send_datas.encode())
        print("已发送数据:", send_datas)
    except Exception as exc:
        pass
    # print("发送异常", exc)


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


# 查找串口
def find_com():
    plist = list(serial.tools.list_ports.comports())

    if len(plist) <= 0:
        print("The Serial port can't find!")
    else:
        plist_0 = list(plist[0])
        serialName = plist_0[0]
        serialFd = serial.Serial(serialName, 9600, timeout=60)
        print("check which port was really used >", serialFd.name)


open_ser()
send_msg(f'a180b025c000d000e025f180Z')
close_ser()