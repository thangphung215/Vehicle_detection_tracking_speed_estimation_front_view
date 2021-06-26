import sys
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, \
    check_imshow
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0),clss=None,speed=0):
    print(type(clss))
    for i, box in enumerate(bbox):
        #classes = ['vehicle']
        classes = ['bike','car','truck','bus','person']
        c = np.int(clss[i])
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label =  classes[c] + str('-{}{:d}'.format("", id))
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        t_size2 = cv2.getTextSize(str(speed[i]), cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] +1 , y1 + t_size[1] + 1), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

        
        cv2.rectangle(img, (x1, y2), (x1 + t_size2[0] +1 , y2 + t_size[1] + 1), color, -1)
        cv2.putText(img, str(round(speed[i],2)), (x1, y2 + t_size2[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    height,width = int(img.shape[0]),int(img.shape[1])
    lines = [451*2,437*2,416*2,378*2,293*2,29*2]
    #for i in range(len(lines)):
    #    img = cv2.line(img,(lines[i],0),(lines[i],height-1),(255,0,0),2)
    return img

def find_iou(a,b):
    ious = np.zeros((len(a),len(b)))
    for i in range(len(a)):
        axmin, aymin, axmax, aymax = a[i]
        area_a = (axmax-axmin)*(aymax-aymin)
        for j in range(len(b)):
            bxmin, bymin, bxmax, bymax = b[j]
            area_b = (bxmax-bxmin)*(bymax-bymin)
            
            xmin = max(axmin,bxmin)
            ymin = max(aymin,bymin)
            xmax = min(axmax,bxmax)
            ymax = min(aymax,bymax)
            
            area_ab = (xmax-xmin)*(ymax-ymin)
            
            if ymax<ymin or xmax<xmin:
                iou = 0
            else:
                iou = area_ab/(area_a + area_b - area_ab)
            ious[i,j] = iou
    return ious
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
    out, source, weights, view_vid, save_vid, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_vid, opt.save_vid, opt.save_txt, opt.img_size
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
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if view_vid:
        view_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'
    track_th = 0
    id_speed = np.zeros((1000,10)) #[st_1, t_1, st_2, t_2, st_3, t_3, speed, time_out, y2,speed_2]
    point = np.zeros((1,2))
    e = 0
    ee = 0
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        
        t_start = time.time()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        ret,frame_kph = capkph.read()
        
        imgkph, real_speed = read_kph(frame_kph)
        
            
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        ttt=0
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                new_det = det[:,:4].cpu().numpy()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []
                clss = np.zeros((len(det),))
                bbox_raw = np.zeros((len(det),4))
                # Adapt detections to deep sort input format

                bb = 0
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    x1,y1,x2,y2 = xyxy
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    clss[bb] = cls.item()
                    
                    bbox_raw[bb] = x1.item(),y1.item(),x2.item(),y2.item()
                    bb += 1

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                #print(clss)
                #print(bbox_raw.shape)
                
            
                ############

                ############
                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0)

                # draw boxes for visualization
                clss_new = np.zeros((len(outputs),))
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    
                    
                    #print(bbox_xyxy.shape)
                    ious = find_iou(bbox_xyxy,bbox_raw)
                    ious_max = np.max(ious,axis=1)
                    ious_argmax = np.argmax(ious,axis=1)
                    
                    for iii in range(len(ious_max)):
                        if ious_max[iii] > 0.7:
                            clss_new[iii] = clss[ious_argmax[iii]]
                    #print(clss_new)
                    
                    #compute_speed
                    
                    #[x,y,x,y,distance,t_point,t_time,speed,side]
                    #p = [p1,p2] #p = [580,650]
                    
                    for kk in range(len(identities)): #[st_1, t_1, st_2, t_2, st_3, t_3, speed, t_in, y2,speed_2]
                        #BOT
                        st = 0
                        #id_speed[identities[kk],7] += 1
                        
                        """
                        if id_speed[identities[kk],8]==0:
                            id_speed[identities[kk],7] = e
                            id_speed[identities[kk],8] = np.exp(bbox_xyxy[kk][3]/1080*(-6.53237221)+7.66756723)
                        elif (e-id_speed[identities[kk],7])%60==0:
                            id_speed[identities[kk],9] = (np.exp(bbox_xyxy[kk][3]/1080*(-6.53237221)+7.66756723)- id_speed[identities[kk],8])/(e-id_speed[identities[kk],7])*30
                            #id_speed[identities[kk],8] = np.exp(bbox_xyxy[kk][3]/1080*(-6.53237221)+7.66756723)
                            #id_speed[identities[kk],7] = e
                            id_speed[identities[kk],6] = round((id_speed[identities[kk],9])*3.6+real_speed,2)"""
                        
                        #p = np.array([537,560,607,754])*4/3
                        p = np.array([p1,p2,p3,p4])*4/3
                        for i in range(1,len(p)+1):
                            if bbox_xyxy[kk][3]>int(p[i-1]):
                                st = i
                                
                            
                                               
                        if id_speed[identities[kk],0]==0 or id_speed[identities[kk],0]==st :
                            if id_speed[identities[kk],0]!=st:
                                id_speed[identities[kk],1] = e
                            id_speed[identities[kk],0] = st
                            
                        elif id_speed[identities[kk],2]==0 or id_speed[identities[kk],2]==st :
                            if id_speed[identities[kk],2]!=st:
                                id_speed[identities[kk],3] = e
                            id_speed[identities[kk],2] = st
                            
                        elif id_speed[identities[kk],4]==0 or id_speed[identities[kk],4]==st :
                            if id_speed[identities[kk],4]!=st:
                                id_speed[identities[kk],5] = e
                            id_speed[identities[kk],4] = st
                        
                        elif id_speed[identities[kk],4]!=st :
                            #id_speed[identities[kk],0] = id_speed[identities[kk],2]
                            #id_speed[identities[kk],1] = id_speed[identities[kk],3]
                            #id_speed[identities[kk],2] = id_speed[identities[kk],4]
                            #id_speed[identities[kk],3] = id_speed[identities[kk],5]
                            id_speed[identities[kk],4] = st
                            id_speed[identities[kk],5] = e

                        st_1 = id_speed[identities[kk],0]
                        st_2 = id_speed[identities[kk],2]
                        st_3 = id_speed[identities[kk],4] 
                        dis = 8*(id_speed[identities[kk],4]-id_speed[identities[kk],2])
                        if st_1<st_2<st_3 and id_speed[identities[kk],6]==0 :
                            #print(dis_default,id_speed[identities[kk],5],id_speed[identities[kk],3],real_speed)
                            id_speed[identities[kk],6] = round(-30*dis*3.6/(id_speed[identities[kk],5]-id_speed[identities[kk],3])+real_speed,1)
                            
                        elif st_1>st_2>st_3>0 and id_speed[identities[kk],6]==0:
                            id_speed[identities[kk],6] = round(30*dis*3.6/(id_speed[identities[kk],5]-id_speed[identities[kk],3])+real_speed,1)
                        
                        
                        



                            
 
                        
                    
                    for pp in range(len(bbox_xyxy)):
                        point = np.concatenate((point,[[int((bbox_xyxy[pp,0]+bbox_xyxy[pp,2])/2),int((bbox_xyxy[pp,1]+bbox_xyxy[pp,3])/2)]]),axis=0)
                    draw_boxes(im0, bbox_xyxy, identities, offset=(0, 0),clss=clss_new,speed=id_speed[identities[:],6])      
                    if len(id_speed)>10: print(id_speed[max(identities)-10:max(identities)])
                    else: print(id_speed[:10])
                    print(imgkph.shape)
                    try:
                        im0[930:,1735:] = imgkph
                    except:
                        continue
                    
                    
                    for i in range(len(p)):
                        im0 = cv2.line(im0,(720,int(p[i])),(1200,int(p[i])),(0,255,0),2)
                    """if len(point)>100:
                        for pp in range(len(point)-100,len(point)):
                            im0 = cv2.circle(im0,(int(point[pp,0]),int(point[pp,1])),0,(0,0,255),2)
                    else:
                        for pp in range(len(point)):
                            im0 = cv2.circle(im0,(int(point[pp,0]),int(point[pp,1])),0,(0,0,255),2)"""

                    

                    
                if ttt==1:
                    break
                track_th += 1
                #break
                # Write MOT compliant results to file
                """if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,bbox_top, bbox_w, bbox_h, -1, -1, id_speed[identity,6], -1))  # label format
                """
            else:
                continue
                #deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            """if view_vid:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration"""

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
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)
        #print('---------{}s'.format(time.time()-t_start))
        e += 1
        ee += time.time()-t_start

            
        
    if save_txt :
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    print(ee/e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/weights/yolov5s.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--kph', type=str,default='', help='kph')
    parser.add_argument('--output', type=str, default='./inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--p1',type=int, default=535,help='limitation')
    parser.add_argument('--p2',type=int, default=535,help='limitation')
    parser.add_argument('--p3',type=int, default=535,help='limitation')
    parser.add_argument('--p4',type=int, default=535,help='limitation')
    parser.add_argument('--s', type=int,default=30, help='distance_between_2_point')
    parser.add_argument('--img-size', type=int, default=1280,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-vid',default=False, action='store_false',
                        help='display results')
    parser.add_argument('--save-vid',default=True, action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', default=True,action='store_true',
                        help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[0,1,2,3,4], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)
