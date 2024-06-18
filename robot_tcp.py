
'''
1. 导⼊ socket 模块
2 socket创建⼀个套接字
3. bind绑定ip和port
4. listen使套接字变为可以被动链接
5. accept等待客户端的链接
6. recv/send接收发送数据
7. 发送数据到客户端
8. 中止与当前客户端的连接
9. 关闭服务端套接字
'''

# 1. 导⼊ socket 模块
from socket import *
import pyautogui
import time
import os
import shutil
from main_cshi import maim
# if os.path.exists(r"C:\Users\hr\Desktop\0527"):
#
#    shutil.move(r"C:\Users\hr\Desktop\0527",  "C:/Users/hr/Desktop/AUTO_DATA/"+str(time.time())+".ply")
#
#
# 2. socket创建⼀个套接字

def tcp_socket_client():
    i = 0
    t = 10
    c = 2559
    k = 1559
    c1 = 2559
    k1 = 1559
    x = c1 / c
    y = k1 / k
    cc = 0
    while True:
        tcp_socket_server = socket(AF_INET, SOCK_STREAM)
        # 3. bind绑定ip和port
        tcp_socket_server.bind(('', 8787))
        # 4. listen使套接字变为可以被动链接
        tcp_socket_server.listen(128)

        # 5. accept等待客户端的链接
        socket_client, ip_port = tcp_socket_server.accept()
        print("新的客户端来了:", ip_port)

        # 循环目的: 为同一个客户端 服务多次
        while True:
            # 6. recv接收客户端发送的数据
            recv_data = socket_client.recv(1024)
            # 判断接收到客户端的信息是否为空，为空则退出
            if not recv_data:
                print("客户端连接断开")
                break

            elif recv_data.decode("gbk") == 'start_v':
                print("收到客户端信息：%s" % recv_data.decode("gbk"))
                print("收到start")
                ##开始采集
                time.sleep(5)
                pyautogui.moveTo(2335 * x, 239 * y)
                time.sleep(2)
                pyautogui.click()
                msg = 'start_s'.strip()
                socket_client.sendall(msg.encode("gbk"))
                # continue
            elif recv_data.decode("gbk") == 'over_v':
                i = i + 1
                print("收到客户端信息：%s" % recv_data.decode("gbk"))
                print("收到完成采集信号，等待融合")
                print('执行第%d按钮' % i)
                print("开始执行按钮")
                time.sleep(3)
                ##结束
                pyautogui.moveTo(2436*x, 240*y)
                time.sleep(0.1)
                pyautogui.click()
                time.sleep(2)
                #融合按钮
                pyautogui.moveTo(211*x, 87*y)
                time.sleep(0.1)
                pyautogui.click()
                time.sleep(2)
                ##点距确认
                pyautogui.moveTo(1695*x, 438*y)
                time.sleep(1)
                pyautogui.click()
                time.sleep(80)
                # ##导出模型
                pyautogui.moveTo(1122*x, 67*y)
                time.sleep(0.7)
                pyautogui.click()
                time.sleep(2)

                ##点云模型
                pyautogui.moveTo(1106*x, 158*y)
                time.sleep(1)
                pyautogui.click()
                time.sleep(10)

                ##保存
                pyautogui.moveTo(1037*x, 755*y)
                time.sleep(1)
                pyautogui.click()
                time.sleep(1)

                ##保存指定文件夹

                pyautogui.moveTo(1288*x, 776*y)
                time.sleep(1)
                pyautogui.click()
                time.sleep(1)
                ##扫描
                pyautogui.moveTo(51*x, 103*y)
                time.sleep(1)
                pyautogui.click()
                time.sleep(2)
                or_path1 = r"rob_data/"
                save_path = r"data/book_seam_dataset/ceshi/"
                for filename in os.listdir(or_path1):
                    if filename.endswith(".ply"):
                        or_path = os.path.join(or_path1, filename)
                        save_path = os.path.join(save_path, filename)
                        shutil.move(or_path, save_path)
                        cc = maim(save_path)
                        print(or_path1)
                print('CC:', cc)
                time.sleep(1)

        if i < t and cc=='ok':
            msg = 'start_s'.strip()
            socket_client.sendall(msg.encode("gbk"))
        else:
            msg = 'over_s'.strip()
            socket_client.sendall(msg.encode("gbk"))
            # tcp_socket_server.close()
            break


