# Drone Pose Estimation
Using a DJI Tello Drone to estimate 3D poses live. 

## Demo:

https://drive.google.com/file/d/1n22INR8USTzreGKZkm-AEgynRVh18yws/view?usp=sharing

### Current approach

Our approach involves using the flow matching model outlined in FMPose3D in order to lift from 2D poses to 3D poses live. 

We loaded yolosm- as our 2D pose estimator with the custom one euro filter to smooth out poses
The yolo model was also pre compiled and uses the mps device setting from pytorch, speeding up inference that would have taken much longer. The model also kept reloading into memory and slowed down inference which was stopped in these changes. 

These improvements led to 5X increase in FPS. 

### Improvements
- achieve steady 30fps
- display the live pose somewhere
- applications (live animations, predictive avoidance, continual learning)
