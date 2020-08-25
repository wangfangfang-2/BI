from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns # 数据可视化的包
from sklearn.tree import DecisionTreeClassifier

# 加载数据
digits = load_digits()
data = digits.data
# 查看数据集大小
print(data.shape)
# 获取第一张图片的像素数
print(digits.images[0])
# 将25%的数据作为测试集，其余作为训练集
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=27)
# 采用Z-Score规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)



clf = DecisionTreeClassifier(random_state=0,splitter='best',criterion='gini') # sklearn默认使用基尼Gini系数

clf.fit(train_ss_x,train_y)

predict_y = clf.predict(test_ss_x)
print('CART算法准确率: %0.4lf' % accuracy_score(test_y, predict_y))

#以下为输出结果部分
(1797, 64)
[[ 0.  0.  5. 13.  9.  1.  0.  0.]
 [ 0.  0. 13. 15. 10. 15.  5.  0.]
 [ 0.  3. 15.  2.  0. 11.  8.  0.]
 [ 0.  4. 12.  0.  0.  8.  8.  0.]
 [ 0.  5.  8.  0.  0.  9.  8.  0.]
 [ 0.  4. 11.  0.  1. 12.  7.  0.]
 [ 0.  2. 14.  5. 10. 12.  0.  0.]
 [ 0.  0.  6. 13. 10.  0.  0.  0.]]
CART算法准确率: 0.8800
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns # 数据可视化的包
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# 加载数据
digits = load_digits()
data = digits.data

split_rate = []
accuracy = []

for i in np.arange(0.05, 0.41, 0.01):
    train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=i, random_state=27)
    ss = preprocessing.StandardScaler()
    train_ss_x = ss.fit_transform(train_x)
    test_ss_x = ss.transform(test_x)

    clf = DecisionTreeClassifier(random_state=666,splitter='random',criterion='gini') # sklearn默认使用基尼Gini系数
    clf.fit(train_ss_x,train_y)

    predict_y = clf.predict(test_ss_x)
    print('test_size:%0.2f,' % i, 'CART算法准确率: %0.4lf' % accuracy_score(test_y, predict_y))
    
    split_rate.append(i)
    accuracy.append(accuracy_score(test_y, predict_y))

plt.figure(figsize=(16,4)) 
plt.plot(split_rate,accuracy)
plt.xlabel('Test_size')
plt.ylabel('Accuracy')
plt.title('CART on mnist') 
plt.xlim(0.05,0.4)
x_major_locator=MultipleLocator(0.01)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
'''zhangyuxi@ZhangdeMacBook-Pro wangfang-BI % python3 CC.py '''
'''test_size:0.05, CART算法准确率: 0.8778
test_size:0.06, CART算法准确率: 0.8796
test_size:0.07, CART算法准确率: 0.8571
test_size:0.08, CART算法准确率: 0.8472
test_size:0.09, CART算法准确率: 0.8457
test_size:0.10, CART算法准确率: 0.8667
test_size:0.11, CART算法准确率: 0.8737
test_size:0.12, CART算法准确率: 0.8704
test_size:0.13, CART算法准确率: 0.8675
test_size:0.14, CART算法准确率: 0.8690
test_size:0.15, CART算法准确率: 0.8296
test_size:0.16, CART算法准确率: 0.8090
test_size:0.17, CART算法准确率: 0.8366
test_size:0.18, CART算法准确率: 0.8426
test_size:0.19, CART算法准确率: 0.8655
test_size:0.20, CART算法准确率: 0.8417
test_size:0.21, CART算法准确率: 0.8492
test_size:0.22, CART算法准确率: 0.8687
test_size:0.23, CART算法准确率: 0.8768
test_size:0.24, CART算法准确率: 0.8449
test_size:0.25, CART算法准确率: 0.8356
test_size:0.26, CART算法准确率: 0.8333
test_size:0.27, CART算法准确率: 0.8395
test_size:0.28, CART算法准确率: 0.8373
test_size:0.29, CART算法准确率: 0.8372
test_size:0.30, CART算法准确率: 0.8148
test_size:0.31, CART算法准确率: 0.8369
test_size:0.32, CART算法准确率: 0.8403
test_size:0.33, CART算法准确率: 0.8418
test_size:0.34, CART算法准确率: 0.8216
test_size:0.35, CART算法准确率: 0.8490
test_size:0.36, CART算法准确率: 0.8377
test_size:0.37, CART算法准确率: 0.8241
test_size:0.38, CART算法准确率: 0.8463
test_size:0.39, CART算法准确率: 0.8388
test_size:0.40, CART算法准确率: 0.8359'''



