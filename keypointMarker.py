# 功能： 关键点标记
# 状态： 完成

# 0.导包
import os
import cv2
import numpy as np

# 输入：video_name, input_video

# 1. 创建导入信息，输入路径
# input_dir = "D:/code/Learn-Segment-and-Track-Anything-main/frame"
# input_video = "D:/code/Learn-Segment-and-Track-Anything-main/test_input/test_video_c.mp4"
# output_dir = "D:/code/Learn-Segment-and-Track-Anything-main/test_out"

input_dir = f'{os.path.join(os.path.dirname(__file__), "results")}/test_masks'
input_video = f'{os.path.join(os.path.dirname(__file__), "input")}/test.mp4'
output_dir = f'{os.path.join(os.path.dirname(__file__), "results")}/test_mark_video'


# def keypoint_marker(input_dir, input_video, output_dir):
def keypoint_marker():
    input_dir = f'{os.path.join(os.path.dirname(__file__), "results")}/test_masks'
    input_video = f'{os.path.join(os.path.dirname(__file__), "results")}/test_track.mp4'
    output_dir = f'{os.path.join(os.path.dirname(__file__), "results")}'

    # 获取原视频信息
    cap = cv2.VideoCapture(input_video)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # 2.创建输出信息 用于将处理好的帧输出视频
    out_video_dir = output_dir + "/" + "test_video_mark.mp4"
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_video = cv2.VideoWriter(out_video_dir, fourcc, 30, (frame_width, frame_height), True)

    # cap = cv2.VideoCapture(input_video)
    # 3. 循环输入帧
    file_list = os.listdir(input_dir)
    num_mask = len(file_list)
    for i in range(1, num_mask):
        img = cv2.imread(input_dir + "/" + str(i) + ".png")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, img_origin = cap.read()

        # 4. 处理每一帧
        # 4.1 获取当前帧信息与轮廓
        img_h, img_w = img_gray.shape
        contours, cnt = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 4.2 创建空表用于操作
        img_temp = np.ones([img_h, img_w])
        x_centers = []
        y_centers = []

        # 4.3 对每个物体轮廓进行操作
        for j in range(len(contours)):
            M = cv2.moments(contours[j])
            x_center = int(M["m10"] / (M["m00"] + 0.01))
            y_center = int(M["m01"] / (M["m00"] + 0.01))
            x_centers.append(x_center)
            y_centers.append(y_center)

            # 4.4 对物体中心点进行标记
            img_temp = cv2.circle(img_temp, (x_center, y_center), 3, 0, -1)
            img_gray = cv2.circle(img_gray, (x_center, y_center), 3, 0, -1)

            # cv2.imshow("img", img_temp)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        # 4.5 将中心点绘制在原图像
        # img_origin[:,:,0] *= np.uint8(img_temp)  # R
        # img_origin[:,:,1] *= np.uint8(img_temp)  # G
        img_origin[:, :, 2] *= np.uint8(img_temp)  # B

        # 4.6 逐帧写入输出视频
        out_video.write(img_origin)
        # out_video.write(img_temp)

    print('finished keypoint mark.')

    out_video.release()

    cap.release()

# keypoint_marker()






