# 功能：将文件夹下的图片输出为视频
# 状态：实现中

import cv2
import os
import argparse  # 导包


parse = argparse.ArgumentParser()  # 初始化一个对象
parse.add_argument('--input_dir', '--in_', help="Direction of input sequence path.")  # 设置参数
parse.add_argument('--output_dir', '--out_', help="Direction of output video path.")
args = parse.parse_args()  # 保存命令行输入参数
input_dir = args.input_dir  # 将命令行参数保存
output_dir = args.output_dir

path_list = os.listdir(input_dir)
# path_list.sort(key=lambda x: int(x.split('d')[1].split('.')[0]))  # 根据实际文件名称进行修改
# path_list.pop(0)  # 如果文件夹中第一个文件不是图片的话，将第一个弹出
path_list.sort(key=lambda x: int(x.split('.')[0]))
img = cv2.imread(input_dir + '/' + path_list[0])
height, width, channel = img.shape
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out_vid = cv2.VideoWriter((output_dir + '/test.mp4'), fourcc, 30, (width, height), True)

for img_names in path_list:
    img_name = f'{input_dir}/{img_names}'
    frame = cv2.imread(img_name)
    out_vid.write(frame)

out_vid.release()


"""
if __name__ == '__maim__':
    input_dir = 'D:\\data\\dj1'
    output_dir = 'D:\\code\\samt_v01\\results'
    img2vid(input_dir, output_dir)

    parse = argparse.ArgumentParser()
    parse.add_argument('--input_dir', '--in_', help="Direction of input sequence path.")
    parse.add_argument('--output_dir', '--out_', help="Direction of output video path.")
    args = parse.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
"""



