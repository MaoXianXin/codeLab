from autokeras.image.image_supervised import ImageClassifier
from autokeras.image.image_supervised import load_image_dataset
import os
import csv
import shutil
from PIL import Image
import cv2

train_dir = './101_ObjectCategories/train'
class_dirs = [i for i in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, i))]

# resize图片大小
def resize_img(path):
    im1 = cv2.imread(path)
    if len(im1.shape) < 3 or im1.shape[2] == 1:
        print('image shape lower than 3dim or third channel is 1')
    im2 = cv2.resize(im1, (224, 224), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(path, im2)

# 1.重命名图片
def rename_img():
    for current_class in class_dirs:
        for image in os.listdir(os.path.join(train_dir, current_class)):
            src = os.path.join(train_dir + '/' + current_class, image)
            resize_img(src)
            dst = os.path.join(train_dir + '/' + current_class, current_class + '_' + image)
            os.rename(src, dst)

# 3.移动图片位置
def move_img_loc():
    for current_class in class_dirs:
        for image in os.listdir(os.path.join(train_dir, current_class)):
            source  = os.path.join(train_dir + '/' + current_class, image)
            destination = os.path.join(train_dir, image)
            shutil.move(source, destination)
    for current_class in class_dirs:
        os.rmdir(os.path.join(train_dir, current_class))

# 2.制作CSV文件
def create_csv():
    with open('./101_ObjectCategories/train/label.csv', 'w') as train_csv:
        fieldnames = ['File Name', 'Label']
        writer = csv.DictWriter(train_csv, fieldnames=fieldnames)
        writer.writeheader()
        label = 0
        for current_class in class_dirs:
            for image in os.listdir(os.path.join(train_dir, current_class)):
                writer.writerow({'File Name': str(image), 'Label': label})
            label += 1
        train_csv.close()

if __name__ == '__main__':
    # for current_class in class_dirs:
    #     for image in os.listdir(os.path.join(train_dir, current_class)):
    #         img_path = os.path.join(train_dir + '/' + current_class, image)
    #         resize_img(img_path)
    rename_img()
    create_csv()
    move_img_loc()