import torch
from PIL import Image
import cv2
import os

# 设置图片目录和视频输出路径
# img_dir = './exprs/ism/ironman_mine'
# output_path = 'my_ironman.mp4'


def imgs_to_video(img_dir, output_path):
    # 读取图片目录中的所有图片文件
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')])
    # 获取第一张图片的宽度和高度

    img_files.sort(key=lambda x:int(x.split('iter_')[1].split('_')[0]))
    first_img = cv2.imread(os.path.join(img_dir, img_files[0]))
    height, width, _ = first_img.shape

    # 创建视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, 10, (width, height))

    # 逐个读取图片并写入视频
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        video_writer.write(img)

    # 释放资源
    video_writer.release()

def imgs_to_video_simple(img_dir, output_path):
    # 读取图片目录中的所有图片文件
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')])
    # 获取第一张图片的宽度和高度

    # img_files.sort(key=lambda x:int(x.split('iter_')[1].split('_')[0]))
    first_img = cv2.imread(os.path.join(img_dir, img_files[0]))
    height, width, _ = first_img.shape

    # 创建视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, 10, (width, height))

    # 逐个读取图片并写入视频
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        video_writer.write(img)

    # 释放资源
    video_writer.release()

if __name__ == "__main__":
    imgs_to_video(img_dir = './exprs/sds/ironman',
            output_path = './exprs/sds/ironman/ironman.mp4')

