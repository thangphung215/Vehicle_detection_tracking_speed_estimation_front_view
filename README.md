# Vehicle_detection_tracking_speed_estimation_front_view
Vehicle detection using YOLOv3,4,5 ---- Tracking using DeepSORT -----Speed Estimating using personal method 

A basic application using 2D image to estimate the speed of other vehicles around, can be developed for autonomous car with 3D detection \
![result_2](https://user-images.githubusercontent.com/59309335/123516449-530a8880-d6c6-11eb-8eb8-a853b1b6fc6a.png)

## Demo video
[Video_40s](https://www.youtube.com/watch?v=CZPUt3wOUQM)

## Code at the time update
[GDrive](https://drive.google.com/drive/folders/1xlSZ2gjnOQgw_pHjTVa02zXkfuU4Reu5?usp=sharing)

## Paper
[YOLOv3](https://arxiv.org/pdf/1804.02767v1.pdf)
[YOLOv4](https://arxiv.org/pdf/2004.10934v1.pdf)
[DeepSORT](https://arxiv.org/pdf/1703.07402v1.pdf)
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the environment with 'requirements.txt'.

```bash
pip install -r requirements.txt
```
Another way is installing the environment of Anaconda with 'environment.yml'

## Usage
[Dataset](https://drive.google.com/file/d/1RLM-2oQtMRDzjNpKPFuknalF91G8Iut2/view?usp=sharing) includes more than 5k images for training and testing \
[YOLOv3](https://github.com/ultralytics/yolov3) \
[YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)\
[YOLOv5](https://github.com/ultralytics/yolov5) 

1. The given link to the dataset includes train and test, so you need to download and extract it first
2. Then train the dataset with yolov5
3. Remember to replace track.py in [Yolov5 + Deep Sort with PyTorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch) with the track_4.py in my repo.
4. Following the tutorials in [Yolov5 + Deep Sort with PyTorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)
5. Finally run:(Remember to get the right movie path responding to kmph, EX:test_day_10_007.MOV and kph_test_day_10_007.MOV
```bash
python track_4.py --source PATH_TO_THE_MOVIE --kph PATH_TO_THE_KMPH --save-vid --yolo_model PATH_TO_YOLO_CHECKPOINT --img 1280
```

## Acknowledgements
[https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3) \
[https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)\
[https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5) \
[https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)


