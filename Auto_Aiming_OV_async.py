import numpy as np
import time
import numpy as np
import torch
import win32con
import win32gui
import win32ui
import yaml
from openvino.runtime import Core, Tensor
from utils.augmentations import letterbox
from utils.general import (cv2, non_max_suppression, scale_boxes)
import pydirectinput

# 以下为GRAB_screen的初始化参数
xywh = [640, 284, 640, 640]
screen_center = [(xywh[0] + xywh[2]) / 2, (xywh[1] + xywh[3]) / 2]
hwin = win32gui.GetDesktopWindow()
hwindc = win32gui.GetWindowDC(hwin)
srcdc = win32ui.CreateDCFromHandle(hwindc)
memdc = srcdc.CreateCompatibleDC()
bmp = win32ui.CreateBitmap()
bmp.CreateCompatibleBitmap(srcdc, xywh[2], xywh[3])
memdc.SelectObject(bmp)
colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]


def MOUSE_move(data):
    choose_close = data.get()
    if choose_close[1] != 0:  # 此处是用来防止归位后 x,y 都等于“0”的情况
        '''pydirectinput.moveRel(xOffset=int(choose_close[1] - xywh[2] / 2),
                              yOffset=int(choose_close[2] - xywh[3] / 2), relative=True)'''
        print("[x y mouse_move_x mouse_move_y ]", choose_close[0], choose_close[1], choose_close[2],
              int(choose_close[1] - (xywh[2] / 2)), int(choose_close[2] - (xywh[3] / 2)))


def grab_screen():
    memdc.BitBlt((0, 0), (xywh[2], xywh[3]), srcdc, (xywh[0], xywh[1]), win32con.SRCCOPY)
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, np.uint8)
    img.shape = (xywh[3], xywh[2], 4)
    im0 = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    im0[:, :, [0, 1, 2]] = im0[:, :, [2, 1, 0]]
    im_letterbox, _1, _2 = letterbox(im0, auto=False)  # 缩放为 （640 640）大小
    im_tensor = Tensor(cv2.dnn.blobFromImage(im_letterbox, 1 / 255.0, swapRB=False))
    return im0, im_letterbox, im_tensor  # 原图 缩放图 缩放后进行tensor化的图


# Load COCO Label from yolov5/data/coco.yaml
with open('./data/coco.yaml', 'r', encoding='utf-8') as f:
    result = yaml.load(f.read(), Loader=yaml.FullLoader)
class_list = result['names']
# CONFIGS
DEVICE = "GPU.0"  # 使用GPU进行预测.0代表的是任务管理器里GPU的代号
DETECTIONS = (0,)  # 检测的类型，0为人
MAX_DETECT = 10  # 最大检测目标
CONF = 0.5  # 置信度
IOU = 0.45
# 初始化openvino 推理内核
core = Core()
# 一系列初始化参数
net = core.compile_model("yolov5s.xml", DEVICE)
input_node = net.inputs[0]
output_node = net.outputs[0]
# 创建当前推理和后帧处理
infer_curr = net.create_infer_request()
infer_next = net.create_infer_request()
# 获得截取的 <原图> <缩放图> <用于用于神经网络的tensor> 用于当前推理
frame_curr, frame_curr_letterbox, frame_curr_tensor = grab_screen()
# 定义当前推理并且以异步形式开始
infer_curr.set_tensor(input_node, frame_curr_tensor)
infer_curr.start_async()  # 此时第一帧推理已经开始
while True:
    start = time.perf_counter()
    choose_close = [0, 0, 2147483647]
    # 获得截取的 <原图> <缩放图> <用于用于神经网络的tensor> 用于下帧推理
    frame_next, frame_next_letterbox, frame_next_tensor = grab_screen()
    # 定义 <下次> 推理
    infer_next.set_tensor(input_node, frame_next_tensor)
    # 等待 <当前> 推理完成
    t1 = time.perf_counter()
    infer_curr.wait()       # 经过测试，程序执行到此处的时(0.018s)早已完成了推理，所以前面实际上对程序造成了堵塞
    t2 = time.perf_counter()
    # 开始下次推理
    infer_next.start_async()
    # 获取 <当前> 推理的结果
    infer_result = infer_curr.get_tensor(output_node)
    # 推理结果转换
    data = torch.tensor(infer_result.data)
    # 此处执行NMS非极大值抑制
    dets = non_max_suppression(data, CONF, IOU, DETECTIONS, max_det=MAX_DETECT)[0].numpy()
    # 获得原始边框坐标，置信度，id值（也就是种类）
    bboxes, scores, class_ids = dets[:, :4], dets[:, 4], dets[:, 5]  # x y x y s c
    # 将原始边框坐标“映射”到未缩放的图上，获得真实坐标
    bboxes = scale_boxes(frame_curr_letterbox.shape[:-1], bboxes, frame_curr.shape[:-1]).astype(int)
    # show bbox of detections
    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        color = colors[int(class_id) % len(colors)]
        cv2.rectangle(frame_curr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        '''cv2.rectangle(frame_curr, (bbox[0], bbox[1] - 20), (bbox[2], bbox[1]), color, -1)'''
        '''cv2.putText(frame_curr, class_list[class_id], (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (255, 255, 255))'''
        distance = ((bbox[0] + bbox[2]) / 2 - xywh[2] / 2) ** 2 + ((bbox[1] + bbox[3]) / 2 - xywh[2] / 2) ** 2
        choose_close = choose_close if choose_close[2] < distance else [(bbox[0] + bbox[2]) / 2,
                                                                        (bbox[1] + bbox[3]) / 2, distance]

    # show FPS
    # cv2.imshow("THIS IS SYNC RESULT", frame_curr)
    # cv2.waitKey(1)  # 1 millisecond
    infer_curr, infer_next = infer_next, infer_curr  # 交换图像
    frame_curr, frame_curr_letterbox = frame_next, frame_next_letterbox  # 交换处理网络
    end = time.perf_counter()
    print((end - start), "fps", " ; Detections:", str(len(class_ids)), choose_close, (t2 - start))
