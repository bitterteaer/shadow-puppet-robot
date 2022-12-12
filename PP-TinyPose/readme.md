### 运行命令

```sh
# 预测一张图片
python3 deploy/python/det_keypoint_unite_infer.py --det_model_dir=output_inference/picodet_s_320_pedestrian --keypoint_model_dir=output_inference/tinypose_128x96 --image_file={your image file} --device=GPU

# 预测多张图片
python3 deploy/python/det_keypoint_unite_infer.py --det_model_dir=output_inference/picodet_s_320_pedestrian --keypoint_model_dir=output_inference/tinypose_128x96 --image_dir={dir of image file} --device=GPU

# 预测一个视频
python3 deploy/python/det_keypoint_unite_infer.py --det_model_dir=output_inference/picodet_s_320_pedestrian --keypoint_model_dir=output_inference/tinypose_128x96 --video_file={your video file} --device=GPU
```



```sh
python python/det_keypoint_unite_infer.py --det_model_dir=picodet_v2_s_192_pedestrian --keypoint_model_dir=tinypose_128x96 --video_file=test01.mp4 --device=CPU
```



### github链接

https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose



### 关键点位置

```
COCO keypoint Description:
    0: "Nose",
    1: "Left Eye",
    2: "Right Eye",
    3: "Left Ear",
    4: "Right Ear",
    5: "Left Shoulder,
    6: "Right Shoulder",
    7: "Left Elbow",
    8: "Right Elbow",
    9: "Left Wrist",
    10: "Right Wrist",
    11: "Left Hip",
    12: "Right Hip",
    13: "Left Knee",
    14: "Right Knee",
    15: "Left Ankle",
    16: "Right Ankle"

AI Challenger Description:
    0: "Right Shoulder",
    1: "Right Elbow",
    2: "Right Wrist",
    3: "Left Shoulder",
    4: "Left Elbow",
    5: "Left Wrist",
    6: "Right Hip",
    7: "Right Knee",
    8: "Right Ankle",
    9: "Left Hip",
    10: "Left Knee",
    11: "Left Ankle",
    12: "Head top",
    13: "Neck"
```

