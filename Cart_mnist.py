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