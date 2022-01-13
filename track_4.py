# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def read_kph(frame):

    img = frame
    text = 0
    if img is not None:
        img = cv2.resize(img,(1280,720))
        img = img[0:600,360:1100]
        x_rect, y_rect, w_rect, h_rect = 245,464,254,135#251,480,250,100 #247,464,250,100 #
        img = cv2.rectangle(img,(x_rect, y_rect, w_rect, h_rect),color=(0,0,0),thickness=-1)
        img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        #lower_orange, upper_orange = np.array([0,231,230]), np.array([13,255,255]) #np.array([0,202,138]), np.array([255,255,255]) # not 1,9,10
        #lower_orange, upper_orange = np.array([0,201,241]), np.array([8,255,255])  # 10
        lower_orange, upper_orange = np.array([0,185,205]), np.array([255,255,255])  # 9
        mask = cv2.inRange(img_hsv,lower_orange, upper_orange)
        kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE , (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,kernel)
        contours = cv2.findContours(mask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)[-2]
        for idx, contour in enumerate(contours):
            contour = np.array(contour)
            bbox = cv2.boundingRect(contour)
            area = bbox[-1]*bbox[-2]

            if area < 300 or max(bbox[-1],bbox[-2])<150:
                continue

            rot_rect = cv2.minAreaRect(contour)
            (cx,cy), (h,w), rot_angle = rot_rect

            rbox = np.int0(cv2.boxPoints(rot_rect))
            cv2.drawContours(img, [rbox], 0, (0,255,0), 1)
            org=(int(cx)-10,int(cy)-10)
            org_rect = (int(x_rect+w_rect/5), int(y_rect+  h_rect*3/4))

            x_l, y_l = contour[np.argmin(contour[:,0,0])][0]
            x_r, y_r = contour[np.argmax(contour[:,0,0])][0]
            cv2.circle(img,(int(cx),int(cy)),1,(255,0,0))

            
            text = np.arctan((y_r-y_l)/(x_r-x_l))/3.14*180*1.10233245 + 22.97893819           #1.16340775+21.41352538
            img = cv2.putText(img, text=str(np.round(text,1)), org = org_rect, fontFace = 1, fontScale=4, color=(255,255,255), thickness = 3, lineType=16)
            img = cv2.line(img,(x_l,y_l),(x_r,y_r),(0,0,255),2)
            #cv2.imshow(files[i],img)
            #cv2.waitKey()
            #cv2.destroyAllWindows()

            #cv2.imwrite("region_result.png", img)

            # Display the resulting frame
            img = cv2.resize(img,(185,150))
        #print(text)
        
    return img,float(text)

def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok= \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    capkph = cv2.VideoCapture(opt.kph)
    print(opt.source)
    dis_default = int(opt.s)
    p1 = opt.p1
    p2 = opt.p2
    p3 = opt.p3
    p4 = opt.p4
    
    srp = open('result.txt','w')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    
    tr = 0 
    speed = {}
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        #speed = speed[:1000]
        tr += 1
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1
        ret,frame_kph = capkph.read()
        
        imgkph, real_speed = read_kph(frame_kph)
        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs) > 0:

                    
                    #print(outputs)
                    #print(confs)
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        #id i outputs[-2]
                        if str(output[-2]) not in speed:
                            speed[str(output[-2])] = [0,0,0,0,0,0,0.]
                        ####
                        st = 0
                        yy = output[3]
                        p = np.array([p1,p2,p3,p4])*4/3
                        for i in range(1,len(p)+1):
                            if yy>int(p[i-1]):
                                st = i
                        if speed[str(output[-2])][0]==0 or speed[str(output[-2])][0]==st :
                            if speed[str(output[-2])][0]!=st:
                                speed[str(output[-2])][1] = tr
                            speed[str(output[-2])][0] = st
                            
                        elif speed[str(output[-2])][2]==0 or speed[str(output[-2])][2]==st :
                            if speed[str(output[-2])][2]!=st:
                                speed[str(output[-2])][3] = tr
                            speed[str(output[-2])][2] = st
                            
                        elif speed[str(output[-2])][4]==0 or speed[str(output[-2])][4]==st :
                            if speed[str(output[-2])][4]!=st:
                                speed[str(output[-2])][5] = tr
                            speed[str(output[-2])][4] = st
                        
                        elif speed[str(output[-2])][4]!=st :
                            speed[str(output[-2])][4] = st
                            speed[str(output[-2])][5] = tr
                        
                        st_1 = speed[str(output[-2])][0]
                        st_2 = speed[str(output[-2])][2]
                        st_3 = speed[str(output[-2])][4]
                        
                        dis = 8*(speed[str(output[-2])][4]-speed[str(output[-2])][2])
                        if st_1<st_2<st_3 and speed[str(output[-2])][6]==0 :
                            #print(dis_default,id_speed[identities[kk],5],id_speed[identities[kk],3],real_speed)
                            speed[str(output[-2])][6] = round(-30*dis*3.6/(speed[str(output[-2])][5]-speed[str(output[-2])][3])+real_speed,1)
                            
                        elif st_1>st_2>st_3>0 and speed[str(output[-2])][6]==0:
                            speed[str(output[-2])][6] = round(30*dis*3.6/(speed[str(output[-2])][5]-speed[str(output[-2])][3])+real_speed,1)
                        ####
                        #print(speed)
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        c = int(cls)  # integer class

                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True),speed = speed[str(output[-2])][6])

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))
                    
                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort.increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            try:
                im0[930:,1735:] = imgkph
            except:
                continue
            if show_vid:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--kph', type=str,default='', help='kph')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--p1',type=int, default=537,help='limitation')
    parser.add_argument('--p2',type=int, default=560,help='limitation')
    parser.add_argument('--p3',type=int, default=607,help='limitation')
    parser.add_argument('--p4',type=int, default=754,help='limitation')
    parser.add_argument('--s', type=int,default=30, help='distance_between_2_point')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
