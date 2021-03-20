import random
from sklearn import metrics
import cv2
import numpy as np
from imutils import paths


def load_images(dirname):
    img_list = []
    for imagePath in paths.list_images(dirname):
        img_list.append(cv2.imread(imagePath))
    return img_list


# 从每一张没有人的原始图片中随机裁出n张64*128的图片作为负样本
def sample_neg(full_neg_lst, neg_list, size):
    random.seed(1)
    width, height = size[1], size[0]
    for i in range(len(full_neg_lst)):
        for j in range(2):  # 这里是n值
            y = int(random.random() * (len(full_neg_lst[i]) - height))
            x = int(random.random() * (len(full_neg_lst[i][0]) - width))
            neg_list.append(full_neg_lst[i][y:y + height, x:x + width])
    return neg_list


def computeHOGs(img_lst, gradient_lst, wsize=(128, 64)):
    hog = cv2.HOGDescriptor()
    hog.winSize = wsize
    for img in img_lst:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (wsize[1], wsize[0]))
        gradient_lst.append(hog.compute(img))


def get_svm_detector(svm):
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)


# 主程序
# 计算HOG特征
neg_list = []
pos_list = []
gradient_lst = []
labels = []
hard_neg_list = []
pos_list = load_images(
    r'pedestrians128x64')
full_neg_lst = load_images(
    r'INRIAPerson\Train\neg')
sample_neg(full_neg_lst, neg_list, [128, 64])
computeHOGs(pos_list, gradient_lst)
[labels.append(1) for _ in range(len(pos_list))]
computeHOGs(neg_list, gradient_lst)
[labels.append(-1) for _ in range(len(neg_list))]

# 训练SVM
svm = cv2.ml.SVM_create()
svm.setCoef0(0.0)
svm.setDegree(3)
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
svm.setTermCriteria(criteria)
svm.setGamma(0)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setNu(0.5)
svm.setP(0.1)
svm.setC(0.01)
svm.setType(cv2.ml.SVM_EPS_SVR)
svm.train(np.array(gradient_lst), cv2.ml.ROW_SAMPLE, np.array(labels))

_, y_pred = svm.predict(np.array(gradient_lst))

y_pred[y_pred >= 0] = 1
y_pred[y_pred < 0] = -1

print(metrics.accuracy_score(np.array(labels),
                             np.array(y_pred).astype(np.int32)))  # 训练集上的精度

# 保存训练结果
hog = cv2.HOGDescriptor()
hog.setSVMDetector(get_svm_detector(svm))
hog.save('myHogDector.bin')
