#-*-coding:utf8-*-
__author__ = '万壑'

from read_data import read_name_list,read_file
from train_model import Model
import cv2

def test_onePicture(path):
    model= Model()
    model.load()
    img = cv2.imread(path)
    img = cv2.resize(img, (64*2, 64*2))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    picType,prob = model.predict(img)
    if picType != -1:
        name_list = read_name_list('data')
        print(name_list[picType],prob)
    else:
        print("Don't know this person")

#读取文件夹下子文件夹中所有图片进行识别
def test_onBatch(path):
    model= Model()
    model.load()
    index = 0
    img_list, label_lsit, counter = read_file(path)
    for img in img_list:
        picType,prob = model.predict(img)
        if picType != -1:
            index += 1
            name_list = read_name_list('data')
            print(name_list[picType])
        else:
            print(" Don't know this person")

    return index

if __name__ == '__main__':
    test_onePicture('ltt.jpg')
    test_onePicture('czq.jpg')
    test_onePicture('tz.jpg')
    test_onePicture('5.pgm')
	test_onePicture('7.pgm')
    #2 3 1 5 7
