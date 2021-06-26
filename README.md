# Vehicle_detection_tracking_speed_estimation_front_view
Vehicle detection using YOLOv3,4,5 ---- Tracking using DeepSORT -----Speed Estimating using personal method 

A basic application using 2D image to estimate the speed of other vehicles around, can be developed for autonomous car with 3D detection \
![result_2](https://user-images.githubusercontent.com/59309335/123516449-530a8880-d6c6-11eb-8eb8-a853b1b6fc6a.png)

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

1. Train the model with dataset given
2. Remember to replace track.py in [Yolov5 + Deep Sort with PyTorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch) with the track.py file in the repo.
3. Following the tutorials in [Yolov5 + Deep Sort with PyTorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)

## Acknowledgements
[https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3) \
[https://github.com/ultralytics/yolov3](https://github.com/WongKinYiu/PyTorch_YOLOv4)\
[https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov5) \
[https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)


