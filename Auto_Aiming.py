# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
import multiprocessing
from multiprocessing import Queue
import argparse
import os
import sys
import time
from pathlib import Path
import pydirectinput
import torch
import numpy as np
import win32gui
import win32ui
import win32con
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
# 以下为GRAB_screen的初始化参数
xywh = [600, 180, 640, 512]
hwin = win32gui.GetDesktopWindow()
hwindc = win32gui.GetWindowDC(hwin)
srcdc = win32ui.CreateDCFromHandle(hwindc)
memdc = srcdc.CreateCompatibleDC()
bmp = win32ui.CreateBitmap()
bmp.CreateCompatibleBitmap(srcdc, xywh[2], xywh[3])
memdc.SelectObject(bmp)

# 此处需要手动输入长宽（用于MOUSE_move）
screen_w, screen_h = 640, 500


def IMG_show(img_list):
    while True:
        if img_list.full():     # 冗余，防止因其他原因使进程阻塞进而导致推理过程暂停
            for x in range(4):
                img_list.get()
        cv2.imshow("THIS IS ASYNC RESULT", img_list.get())
        cv2.waitKey(1)  # 1 millisecond


def MOUSE_move(data):
    while True:
        choose_close = data.get()
        if choose_close[1] != 0:    # 此处是用来防止归位后 x,y 都等于“0”的情况
            pydirectinput.moveRel(xOffset=int(choose_close[1] - screen_w / 2),
                                  yOffset=int(choose_close[2] - screen_h / 2), relative=True)
            print("ASYNC[x y mouse_move_x mouse_move_y ]", choose_close[0], choose_close[1], choose_close[2],
                  int(choose_close[1] - (screen_w / 2)), int(choose_close[2] - (screen_h / 2)))
            # 判断是否在指定的范围内，如满足则点下左键
            if choose_close[0] <= 200:
                pydirectinput.click()
                # time.sleep(1.5)


def GRAB_screen():
    memdc.BitBlt((0, 0), (xywh[2], xywh[3]), srcdc, (xywh[0], xywh[1]), win32con.SRCCOPY)
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, np.uint8)
    img.shape = (xywh[3], xywh[2], 4)
    im0s = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    im = np.asarray(im0s)  # 转换为np.array形式[w,h,c]
    im = im.swapaxes(0, 2)  # 交换为[c,h,w]
    im = im.swapaxes(1, 2)  # 交换为[c,w,h]
    im[[0, 1, 2]] = im[[2, 1, 0]]  # 将RGB 转换为 BGR
    return im, im0s

@smart_inference_mode()
def run(
        show_img,
        mouse_move,
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        img_size=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=2,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img_sync=False,  # show results
        view_img_async=False,  # show results in async
        move_mouse_sync=False,
        move_mouse_async=False,
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    img_size = check_img_size(img_size, s=stride)  # check image size
    # Data loader
    bs = 1  # batch_size
    dataset = LoadScreenshots(source, img_size=img_size, stride=stride, auto=pt)
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *img_size))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    choose_close = [10000000000, 0, 0]
    end = time.perf_counter()
    '''for path, im, im0s, vid_cap, s in dataset:'''
    while True:
        im, im0s = GRAB_screen()
        s = "debug:"
        path = "0"
        start = time.perf_counter()
        time_count = start - end
        end = time.perf_counter()
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        # Inference
        with dt[1]:
            predict = model(im, augment=augment, visualize=False)
        # NMS
        with dt[2]:
            predict = non_max_suppression(predict, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(predict):  # per image
            seen += 1
            p, im0 = path, im0s.copy()
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=int(2), example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    ppl_list = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()    # 返回的是坐标框中心的xywh
                    # 因为没有训练专门模型，此处用来排除手的干扰
                    if ppl_list[0] > 0.7 and ppl_list[1] > 0.7:
                        continue
                    # 获得检测框的中心，以及该中心距离屏幕中心的距离
                    ppl_x, ppl_y = int((ppl_list[0]) * screen_w), int((ppl_list[1] - ppl_list[3] * 0.25) * screen_h)
                    ppl_distance = int(ppl_x - screen_w / 2) ** 2 + int(ppl_y - screen_h / 2) ** 2      # 用来判断与头的距离
                    # 用来保存在一张图片中距离中心点最近的坐标框
                    if ppl_distance < choose_close[0]:
                        choose_close[0], choose_close[1], choose_close[2] = ppl_distance, ppl_x, ppl_y
                    # 对图片进行添加标签
                    if view_img_async or view_img_sync:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')  # 标签
                        annotator.box_label(xyxy, label, color=colors(int(conf), True))  # 坐标 标签 颜色（置信度）
                    # 移动鼠标（自瞄部分，这一块需要使用管理员权限，并在游戏内打开，不然有可能会失效）
                if move_mouse_async:    # 异步
                    mouse_move.put(choose_close)
                if move_mouse_sync:     # 同步
                    if choose_close[1] != 0:
                        pydirectinput.moveRel(xOffset=int(choose_close[1] - screen_w / 2),
                                                  yOffset=int(choose_close[2] - screen_h / 2), relative=True)
                        print("SYNC[x y mouse_move_x mouse_move_y ]", choose_close[0], choose_close[1], choose_close[2],
                                int(choose_close[1] - (screen_w / 2)), int(choose_close[2] - (screen_h / 2)))
                        if choose_close[0] <= 200:
                            pydirectinput.click()
                            # time.sleep(1.5)
                choose_close = [10000000000, 0, 0]
            # Stream results
            if view_img_async:  # 异步
                show_img.put(annotator.result())
            if view_img_sync:   # 同步
                cv2.imshow("THIS IS SYNC RESULT", annotator.result())
                cv2.waitKey(1)  # 1 millisecond

        # Print time
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}inference: {dt[1].dt * 1E3:.1f}ms, total: {time_count* 1E3:.2f}ms")

    # 由于是死循环所以这个选项没有用
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'screen', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--img-size', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img-sync', default=False, action='store_true', help='view-img-sync')
    parser.add_argument('--view-img-async', default=False, action='store_true', help='view-img-async')
    parser.add_argument('--move-mouse-sync', default=False, action='store_true', help='move-mouse-sync')
    parser.add_argument('--move-mouse-async', default=False, action='store_true', help='move-mouse-async')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(show_img_0, mouse_move_0, opt_0):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(show_img_0, mouse_move_0, **vars(opt_0))


if __name__ == "__main__":
    opt = parse_opt()
    show_img = Queue(5)
    mouse_move = Queue()
    MAIN = multiprocessing.Process(target=main, args=(show_img, mouse_move, opt))
    SHOW = multiprocessing.Process(target=IMG_show, args=(show_img,))
    MOVE = multiprocessing.Process(target=MOUSE_move, args=(mouse_move,))
    MAIN.start()
    SHOW.start()
    MOVE.start()


'''
测试数据（3060 Laptop，i7 12700h）：

##########################################显示图片#################################################
无目标时(无鼠标移动)(调用win32api截图):
    异步显示图片:
        debug:512x640 (no detections), inference: 6.6ms, total: 20.23ms
        debug:512x640 (no detections), inference: 6.6ms, total: 20.77ms
        debug:512x640 (no detections), inference: 9.9ms, total: 14.29ms
        debug:512x640 (no detections), inference: 6.0ms, total: 20.88ms
    同步显示图片:
        debug:512x640 (no detections), inference: 9.4ms, total: 28.12ms
        debug:512x640 (no detections), inference: 9.3ms, total: 27.71ms
        debug:512x640 (no detections), inference: 9.3ms, total: 34.73ms
        debug:512x640 (no detections), inference: 9.3ms, total: 28.24ms
        
##########################################移动鼠标################################################
无实时预览时(3个目标)(关闭鼠标暂停)
    同步移动鼠标
         SYNC[x y mouse_move_x mouse_move_y ] 305 304 243 -16 -7
        screen 0 (LTWH): 640,304,640,512: 512x640 4 persons, inference: 12.5ms, total: 128.05ms
            SYNC[x y mouse_move_x mouse_move_y ] 305 304 243 -16 -7
        screen 0 (LTWH): 640,304,640,512: 512x640 4 persons, inference: 6.0ms, total: 112.29ms
            SYNC[x y mouse_move_x mouse_move_y ] 305 304 243 -16 -7
        screen 0 (LTWH): 640,304,640,512: 512x640 4 persons, inference: 12.0ms, total: 124.37ms
            SYNC[x y mouse_move_x mouse_move_y ] 305 304 243 -16 -7
        screen 0 (LTWH): 640,304,640,512: 512x640 4 persons, inference: 23.9ms, total: 145.60ms
    异步移动鼠标
        暂时不考虑，pydirectinput 效率似乎比较低下，会有很明显的滞后

##########################################屏幕截图####################################################
    使用yolo自带的屏幕获取(拖动森林图片移动)
        screen 0 (LTWH): 640,304,640,512: 512x640 (no detections), inference: 13.5ms, total: 21.08ms
        screen 0 (LTWH): 640,304,640,512: 512x640 (no detections), inference: 8.9ms, total: 27.76ms
        screen 0 (LTWH): 640,304,640,512: 512x640 (no detections), inference: 10.1ms, total: 20.67ms
        screen 0 (LTWH): 640,304,640,512: 512x640 (no detections), inference: 8.4ms, total: 20.55ms
    调用win32api获取屏幕(拖动森林图片移动)
        debug:512x640 (no detections), inference: 8.9ms, total: 21.25ms
        debug:512x640 (no detections), inference: 11.7ms, total: 19.96ms
        debug:512x640 (no detections), inference: 8.9ms, total: 20.19ms
        debug:512x640 (no detections), inference: 10.5ms, total: 20.80ms
'''
