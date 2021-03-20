import cv2
import numpy as np
from imutils import paths

# 在图片中检测到行人的位置画矩形


def draw_person(src, person_xy):
    x, y, w, h = person_xy
    cv2.rectangle(src, (x, y), (x+w, y+h), (0, 255, 255), 2)


def get_svm_detector(svm):
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)


path = r"INRIAPerson\Test\pos"  # 选择要进行检测的图片文件夹

# loop over the test dataset
for imagePath in paths.list_images(path):
    image = cv2.imread(imagePath)
    # 调用
    hog = cv2.HOGDescriptor()
    hog.load('myHogDector.bin')

    # hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    (rects, weights) = hog.detectMultiScale(
        image, 0, winStride=(16, 16), padding=(8, 8), scale=1.05
    )

    # draw the bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
