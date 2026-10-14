# Drone Pose Estimation
Using a DJI Tello Drone to estimate 3d poses using FM models. 


### Basic Approach:

The way I will approach this is to record videos from the drone. Once finished, I will run the 2d pose estimation on it, then see the 3d model pose estimation. 

Recording video:
- Use the opencv VideoWriter class that basically reads each frame and writes to a file. It checks every 1/30th of a second to write to the frame; runs on a separate thread

Inference:
- Use the opencv VideoCapture class to get the frames of the video 
- Stack frames and pass through first through 2d pose detection
- After, pass those keyframes of 2d pose to 3d pose for estimation
