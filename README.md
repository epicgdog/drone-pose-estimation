# Drone Pose Estimation
Using a DJI Tello Drone to estimate 3D poses live. 

## Demo:

https://drive.google.com/file/d/1n22INR8USTzreGKZkm-AEgynRVh18yws/view?usp=sharing

### Current approach

Our approach involves using the flow matching model outlined in [FMPose3D](https://arxiv.org/abs/2602.05755) in order to lift from 2D poses to 3D poses live. 

We loaded YOLO11s-pose (`yolo11s-pose`) as our 2D pose estimator with a custom [One Euro filter](https://gery.casiez.net/1euro/) ([paper](https://doi.org/10.1145/2207676.2208639)) to smooth out poses.
The YOLO model was also pre compiled to CoreML (FP16), which runs on Apple's Neural Engine and is about 1.6x faster than the PyTorch model on the mps device. The 3D model also kept reloading into memory and slowed down inference, which was stopped in these changes. 

These improvements led to 5X increase in FPS. 

### Todo
- achieve steady 30fps
- display the live pose somewhere
- applications (live animations, predictive avoidance, continual learning)
