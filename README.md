# Stereo\_Filter

ROS Package to expose OpenCV's WLS Filter on Stereo Input.

[Demo Video](https://youtu.be/FA1Y25zjlRo)

```bash
ROS_NAMESPACE=stereo rosrun uvc_camera uvc_stereo_node _left/device:=/dev/video1 _right/device:=/dev/video2
ROS_NAMESPACE=stereo rosrun stereo_filter stereo_filter left:=left/image_raw right:=right/image_raw
```
