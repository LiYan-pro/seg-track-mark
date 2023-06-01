# 重要代码
# 功能：分割和跟踪的函数与函数接口
# 状态：已完成

# import os
import cv2
from model_args import segtracker_args,sam_args,aot_args
# from PIL import Image
# from aot_tracker import _palette
import numpy as np
import torch
# import imageio
# from scipy.ndimage import binary_dilation
# from test_SegTracker import SegTracker


# 获取视频首帧
# 输入：源视频
# 返回：outputs=[input_video_first_frame, origin_frame, drawing_board]
def get_meta_from_video(input_video):
    print("get meta information of input video")
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _, first_frame = cap.read()

    cap.release()

    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    return first_frame, first_frame, first_frame


# 初始化SegTracker
# 输入：aot_model = "r50-deaotl", sam_gap, max_obj_num, points_per_side, origin_frame
# 输出：outputs =[Seg_Tracker, input_video_first_frame, click_state]
def init_SegTracker(aot_model, sam_gap, max_obj_num, points_per_side, origin_frame):
    if origin_frame is None:
        return None, origin_frame, [[], []]

    # reset aot args
    aot_args["model"] = aot_model
    # aot_args["model_path"] = aot_model2ckpt[aot_model]
    aot_args["model_path"] = "./ckpt/R50_DeAOTL_PRE_YTB_DAV.pth"

    # reset sam args
    segtracker_args["sam_gap"] = sam_gap
    segtracker_args["max_obj_num"] = max_obj_num
    segtracker_args["min_new_obj_iou"] = 0.2
    sam_args["generator_args"]["points_per_side"] = points_per_side

    Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
    Seg_Tracker.restart_tracker()
    return Seg_Tracker, origin_frame, [[], []]


# from test_seg_track_anything import draw_mask
from seg_track_anything import draw_mask
# 使用SAM分割首帧的全部目标
# 输入：Seg_Tracker, aot_model, origin_frame, sam_gap, max_obj_num, points_per_sid
# 输出：outputs=[Seg_Tracker,input_video_first_frame]
def segment_everything(Seg_Tracker, aot_model, origin_frame, sam_gap, max_obj_num, points_per_side):
    if Seg_Tracker is None:
        Seg_Tracker, _, _ = init_SegTracker(aot_model, sam_gap, max_obj_num, points_per_side, origin_frame)

    frame_idx = 0

    with torch.cuda.amp.autocast():
        pred_mask = Seg_Tracker.seg(origin_frame)
        torch.cuda.empty_cache()
        # gc.collect()
        Seg_Tracker.add_reference(origin_frame, pred_mask, frame_idx)

        masked_frame = draw_mask(origin_frame.copy(), pred_mask)
        # masked_frame = (origin_frame*0.3+colorize_mask(pred_mask)*0.7).astype(np.uint8)

    return Seg_Tracker, masked_frame


# 鼠标交互函数
# 输入：
# 输出：
def interact_point(input_video_first_frame):
    # current_index = 0
    # cv2.setMouseCallback("image", mouse_click)
    input_point = []
    input_label = []
    # input_stop = False

    image_display = input_video_first_frame.copy()
    display_info = f'test.mp4 \n' \
                   f'| Press s to save | Press w to predict | Press d to next image | Press a to previous image \n' \
                   f'| Press space to clear | Press q to remove last point '

    y0, dy = 20, 30
    for i, txt in enumerate(display_info.split('\n')):
        y = y0 + i * dy
        cv2.putText(image_display, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)


# convert points input to prompt state
def get_prompt(input_point, input_label):
    # cv2.namedWindow("image")
    # cv2.setMouseCallback("image", mouse_click)
    # input_point = [[666, 468], [693, 541], [647, 515], [662, 510], [655, 457]]
    # input_label = [1, 0, 0, 0, 1]

    prompt = {
        "prompt_type": ["click"],
        "input_point": input_point,
        "input_label": input_label,
        "multimask_output": "True",
    }
    return prompt


# from test_SegTracker import SegTracker
from SegTracker import SegTracker
def refine_acc_prompt(Seg_Tracker, prompt, origin_frame):
    # Refine acc to prompt
    predicted_mask, masked_frame = Seg_Tracker.refine_first_frame_click(
                                                      origin_frame=origin_frame,
                                                      points=np.array(prompt["input_point"]),
                                                      labels=np.array(prompt["input_label"]),
                                                      multimask=prompt["multimask_output"],
                                                    )

    with torch.cuda.amp.autocast():
        # Reset the first frame's mask
        frame_idx = 0
        Seg_Tracker.restart_tracker()
        Seg_Tracker.add_reference(origin_frame, predicted_mask, frame_idx)

    return masked_frame


# 交互式点选分割图像
# 输入：Seg_Tracker, origin_frame, point_prompt, click_state, aot_model, sam_gap, max_obj_num, points_per_side
# 输出：outputs=[Seg_Tracker, input_video_first_frame, click_state]
def sam_refine(Seg_Tracker, origin_frame, click_state, aot_model, sam_gap, max_obj_num, points_per_side, input_point, input_label):
    '''
    Args:
        template_frame: PIL.Image
        point_prompt: flag for positive or negative button click
        click_state: [[points], [labels]]
    '''

    if Seg_Tracker is None:
        Seg_Tracker, _, _ = init_SegTracker(aot_model, sam_gap, max_obj_num, points_per_side, origin_frame)

    # prompt for sam model
    prompt = get_prompt(input_point, input_label)
    print(prompt)

    # Refine acc to prompt
    masked_frame = refine_acc_prompt(Seg_Tracker, prompt, origin_frame)

    return Seg_Tracker, masked_frame, click_state


from seg_track_anything import tracking_objects_in_video
# 跟踪视频目标
# 输入：Seg_Tracker, input_video
# 输出：outputs=[output_video, output_mask]
def tracking_objects(Seg_Tracker, input_video):
    return tracking_objects_in_video(Seg_Tracker, input_video)


# 主函数：用于分割跟踪目标
# 输入：视频
def seg_track(input_point, input_label, input_video):
    import argparse, os
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid', default=f'{os.path.join(os.path.abspath(__file__))}/../input/test.mp4',
                        help="test video dir OR put test video into this dir.")
    parser.add_argument('--output', default=f'{os.path.join(os.path.abspath(__file__))}/../results/test_out.mp4',
                        help="output test video dir.")
    parser.add_argument('--sam_gap', '--gap', default=999, type=int, help="Gap counts between frames.")
    parser.add_argument('--max_obj_num', default=1, type=int, help="Max object number.")
    parser.add_argument('--points_per_sid', default=50, type=int, help="Points per side.")
    parser.add_argument('--aot_model', default="r50_deaotl", help="Aot model: DeAOT.")
    parser.add_argument('--ckpt_aot', default=f'{os.path.join(os.path.abspath(__file__), "ckpt")}'
                                              f'/R50_DeAOTL_PRE_YTB_DAV.pth', help="AOT model CKPT dir.")
    parser.add_argument('--ckpt_sam', default=f'{os.path.join(os.path.abspath(__file__), "ckpt")}'
                                              f'/sam_vit_b_01ec64.pth', help="SAM model CKPT dir.")

    args = parser.parse_args()

    # 初始化参数
    input_video = input_video
    aot_model = args.aot_model
    sam_gap = args.sam_gap
    max_obj_num = args.max_obj_num
    points_per_side = args.points_per_sid
    # 输入视频首帧，用于点选分割
    input_video_first_frame, origin_frame, drawing_board = get_meta_from_video(input_video)
    # 初始化SAM分割
    Seg_Tracker, input_video_first_frame, click_state = \
        init_SegTracker(aot_model, sam_gap, max_obj_num, points_per_side, origin_frame)
    # _ = Seg_Tracker.to(device="cuda")

    # predictor = SamPredictor(sam)

    interact_point(input_video_first_frame)
    # 采用sam进行局部分割
    Seg_Tracker, masked_frame, click_state = sam_refine(Seg_Tracker, origin_frame, click_state, aot_model, sam_gap,
                                                        max_obj_num, points_per_side, input_point, input_label)
    # 采用SAM分割全部
    # Seg_Tracker, masked_frame = segment_everything(Seg_Tracker, aot_model, origin_frame, sam_gap, max_obj_num, points_per_side)

    output_video, output_mask = tracking_objects(Seg_Tracker, input_video)

"""
if __name__ == "__main__":
    input_point = [[666, 468], [693, 541], [647, 515], [662, 510], [655, 457]]
    input_label = [1, 0, 0, 0, 1]
    seg_track(input_point, input_label)
"""