# -*encoding: utf-8 *-

import os
import numpy as np
import cv2
import tkinter as tk
from typing import List
from PIL import Image
from PyPDF2 import PdfMerger

def order_points(pts:List):
    pts = np.array(pts)
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image:np.array, pts:List):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth-1 , 0],
        [maxWidth-1 , maxHeight-1 ],
        [0, maxHeight-1 ]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def click_event(event, x, y, flags, param):
    img, points = param[0], param[1]
    if event == cv2.EVENT_LBUTTONDOWN:
        # 在图像上画一个圆来标记点击的位置
        cv2.circle(img, (x,y), 8, (255,255,0), -1)
        points.append((x, y))
        cv2.imshow('image', img)

def transform(img_path:str='example.jpg'):
    image:np.array = cv2.imread(img_path)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    points = []
    cv2.setMouseCallback('image', click_event, [image, points])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    warped:np.array = four_point_transform(image, points)
    # cv2.imshow('warped', warped)
    # cv2.imwrite('transformed.jpg', warped)

    return warped

def gaussian(image:np.array, radius:int) -> np.array:
    return cv2.GaussianBlur(image, (radius, radius), 0)

def divide(image:np.array, base:np.array) -> np.array:
    # dim1:x dim2:y dim3:channel
    return image/base

# def get_red(image:np.array) -> np.array:
#     # Convert BGR image to HSV
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     # Define range of red color in HSV
#     lower_red = np.array([0,70,50])
#     upper_red = np.array([10,255,255])

#     # Threshold the HSV image to get only red colors
#     mask = cv2.inRange(hsv, lower_red, upper_red)

#     # Bitwise-AND mask and original image
#     red_part = cv2.bitwise_and(image, image, mask=mask)
#     return red_part


# def set_red(gray_image:np.array, red_part:np.array) -> np.array:

#     # Create a 3-channel image from the grayscale image
#     gray_image_3channel = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

#     # Subtract the red part from the grayscale image
#     result = cv2.subtract(gray_image_3channel, red_part)

#     # Add the red part back to the result
#     result = cv2.add(result, red_part)

#     return result

# def adaptive_thresh(image, win_size, ratio=0.15):
#     # 对图像矩阵进行均值平滑
#     image_mean = cv2.blur(image, win_size)
#     # 原图像矩阵与平滑结果做差
#     out = image - (1.0-ratio) * image_mean
#     # 当差值大于或等于0时，输出值为255，反之输出值为0
#     out[out >= 0] = 255
#     out[out < 0] = 0
#     out = out.astype(np.uint8)
#     return out

def scan(image:np.array, gaussian_radius:int, threshold_val:float):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    base = gaussian(image, gaussian_radius)
    mixed_image = divide(image, base)
    # red_part = get_red(image)
    # mixed_image = mixed_image.astype(np.float32)
    # cv2.imshow('mixed', mixed_image)
    # mixed_image = cv2.cvtColor(mixed_image, cv2.COLOR_BGR2GRAY)
    # mixed_image = cv2.normalize(mixed_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    _, threshold = cv2.threshold(mixed_image, threshold_val, 255, cv2.THRESH_BINARY)
    # threshold = adaptive_thresh(mixed_image, (19, 19), 0.15)
    # return set_red(threshold, red_part)
    return threshold


def main(path:str, save_path:str, gaussian_radius:int=19, threshold_val:float=0.98):
    transformed = transform(path)
    scanned = scan(transformed, gaussian_radius, threshold_val)

    # a4_size = (210, 297)
    # resized = cv2.resize(scanned, a4_size)

    cv2.imwrite(save_path, scanned)
    cv2.imshow(path, scanned)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_pdf(jpg_files):

    # 将JPG文件转换为PDF文件
    pdf_files = []
    for jpg_file in jpg_files:
        img = Image.open(jpg_file)
        pdf_file = jpg_file.replace('.jpg', '.pdf')
        img.save(pdf_file, "PDF", resolution=1000.0)
        # print(f'[INFO] {jpg_file} saved as {pdf_file} successfully')
        print(f'[DEBUG] img{img.size}  pdf{pdf_file}')
        pdf_files.append(pdf_file)

    # 使用PyPDF2库将PDF文件合并
    merger = PdfMerger()
    for pdf_file in pdf_files:
        merger.append(pdf_file)

    # 保存合并后的PDF文件
    merger.write("merged.pdf")
    merger.close()

    # 删除临时的PDF文件
    for pdf_file in pdf_files:
        os.remove(pdf_file)

def auto_scan(gaussian_radius, threshold_val): # a wrapper to filter all `.jpg` files in the current directory
    current_directory = os.getcwd()
    files_in_directory = os.listdir(current_directory)
    jpg_files = [file for file in files_in_directory if file.endswith(".jpg")]
    # print(jpg_files)
    os.makedirs('output', exist_ok=True)
    for file in jpg_files:
        save_path = os.path.join('output', f'out_{file}')
        try:
            main(file, save_path, gaussian_radius, threshold_val)
            print(f'[INFO] processed {file} successfully')
        except Exception as e:
            print(f'[Error] processing {file}: {e}')

    print("[INFO] All files processed successfully")
    jpg_scanned_dir = os.path.join(current_directory, 'output')
    os.chdir(jpg_scanned_dir)
    jpg = os.listdir()
    save_pdf(jpg)

def tk_init():
    def get_value():
        # 获取Spinbox组件中的值
        gaussian_radius:str = gaussian.get()
        threshold_val:str = threshold.get()
        print(f'[INFO] gaussian_radius: {gaussian_radius}, threshold_val: {threshold_val}')
        root.destroy()
        auto_scan(int(gaussian_radius), float(threshold_val))

    root = tk.Tk()

    # 创建一个Spinbox组件
    gaussian = tk.Spinbox(root, from_=1, to=60, increment=2, font=('微软雅黑', 30), )
    threshold = tk.Spinbox(root, from_=0.1, to=1.0, increment=0.001, font=('微软雅黑', 30))
    gaussian.delete(0, tk.END)
    gaussian.insert(tk.END, '19')
    threshold.delete(0, tk.END)
    threshold.insert(tk.END, '0.98')

    gaussian_text = tk.Label(root, text="高斯模糊半径（默认19）", font=('微软雅黑', 30))
    threshold_text = tk.Label(root, text="阈值（默认0.98）", font=('微软雅黑', 30))

    gaussian_text.pack(expand=True, padx=20, pady=8)
    gaussian.pack(expand=True, padx=20, pady=8)
    threshold_text.pack(expand=True, padx=20, pady=8)
    threshold.pack(expand = True, padx=20, pady=8)

    # 创建一个Button组件，点击时会调用get_value函数
    button = tk.Button(root, text="OK", command=get_value, font=('微软雅黑', 30))
    button.pack()

    # 运行Tk窗口
    root.mainloop()
    
if __name__ == '__main__':
    tk_init()

