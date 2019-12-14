#  2019/12/13
# 加入类别标签以及聚类信息
# 完整的包含分类模块的可以参考《findwk.py》or 《fcmgan.py》
# train loss ~= 0.05, test ~= 0.12  很明显训练《测试，说明过拟合，因此新思路或许有效
# 误差为table 2 中的内容
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pandas as pd

# %% System Parameters
# 1. Mini batch size
mb_size = 128
# 2. Missing rate
p_miss = 0.2
# 3. Hint rate
p_hint = 0.9  # 后面函数生成的M大约有9成数值为1
# 4. Loss Hyperparameters
alpha = 10  # 越大越关注完整数据的损失
beta = 0.0
# 5. Train Rate
train_rate = 0.8

# Data generation
Data = np.loadtxt('E:\PyCharm 2018.2.1\my project\Letter_with_label.csv', delimiter=",",
                  skiprows=1)
Data = Data[:2000, :]  # 修改成只取前面2000行进行计算

# 加载聚类中心和隶属度
sample_u = pd.read_excel('E:\PyCharm 2018.2.1\my project\sample_u.csv')
sample_v = pd.read_excel('E:\PyCharm 2018.2.1\my project\sample_v.csv')
sample_u = np.array(sample_u)
sample_u = sample_u.transpose()  # [n,c]
sample_v = np.array(sample_v)    # [c,d]

# Parameters
No = len(Data)
Dim = len(Data[0, :-1])

# Hidden state dimensions
H_Dim1 = Dim
H_Dim2 = Dim

# Normalization (0 to 1)
Min_Val = np.zeros(Dim)
Max_Val = np.zeros(Dim)

for i in range(Dim):
    Min_Val[i] = np.min(Data[:, i])
    Data[:, i] = Data[:, i] - np.min(Data[:, i])
    Max_Val[i] = np.max(Data[:, i])
    Data[:, i] = Data[:, i] / (np.max(Data[:, i]) + 1e-6)

# %% Missing introducing
p_miss_vec = p_miss * np.ones((Dim, 1))

Missing = np.zeros((No, Dim))

for i in range(Dim):
    A = np.random.uniform(0., 1., size=[len(Data), ])
    B = A > p_miss_vec[i]
    Missing[:, i] = 1. * B

# %% Train Test Division

idx = np.random.permutation(No)

Train_No = int(No * train_rate)
Test_No = No - Train_No

# Train / Test Features
trainX = Data[idx[:Train_No], :-1]
testX = Data[idx[Train_No:], :-1]

# ######### 训练，测试的标签 #########
trainY = Data[idx[:Train_No], -1]
testY = Data[idx[Train_No:], -1]

# Train / Test Missing Indicators
trainM = Missing[idx[:Train_No], :]
testM = Missing[idx[Train_No:], :]


# %% Necessary Functions

# 1. Xavier Initialization Definition
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# Hint Vector Generation
def sample_M(m, n, p):
    A = np.random.uniform(0., 1., size=[m, n])  # 生成0~1之间的随机数
    B = A > p
    C = 1. * B  # 将逻辑值转化为0跟1！！！
    return C

# 定义求矩阵向量间欧式距离的函数
def EuclideanDistances(A, B):
    A = np.array(A)
    B = np.array(B)

    BT = B.transpose()
    vecProd = np.dot(A,BT)
    # SqA = A**2
    SqA = A * A
    sumSqA = np.matrix(np.sum(SqA, axis=1))  # axis =1 对同一行求和
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    SqED[SqED<0]=0.0
    ED = np.sqrt(SqED)
    return ED


# 定义一个求隶属度U的函数（在这里我让v不变，求出生成样本在该中心下的隶属度，与初始化的隶属度做对比）
def find_U(x, v):
    # x = np.array(x)
    # v = np.array(v)

    # n = len(x)
    # c = len(v)
    n = 128
    c = 2
    u = np.zeros((n, c))
    distance_matrix = EuclideanDistances(x, v)
    for j in range(0, c):
        for i in range(0, n):
            dummy = 0.0
            for k in range(0, c):
                dummy += (distance_matrix[i, j] / distance_matrix[i, k]) ** (2 / (2 - 1))
            u[i, j] = 1 / dummy
    return u

'''
GAIN Consists of 3 Components
- Generator
- Discriminator
- Hint Mechanism
'''
# %% GAIN Architecture

# %% 1. Input Placeholders
# 1.1. Data Vector
X = tf.placeholder(tf.float32, shape=[None, Dim])
# 1.2. Mask Vector
M = tf.placeholder(tf.float32, shape=[None, Dim])
# 1.3. Hint vector
H = tf.placeholder(tf.float32, shape=[None, Dim])
# 1.4. X with missing values
New_X = tf.placeholder(tf.float32, shape=[None, Dim])
# ##### Data Label ######
Y = tf.placeholder(tf.float32, shape=[None, 1])  # 暂时只考虑二分类情况
V = tf.placeholder(tf.float32, shape=[None, Dim])  # 聚类的中心向量
sample_U = tf.placeholder(tf.float32, shape=[None, 2])  # [n, c]

# %% 2. Discriminator
D_W1 = tf.Variable(xavier_init([Dim * 2, H_Dim1]))  # Data + Hint as inputs
D_b1 = tf.Variable(tf.zeros(shape=[H_Dim1]))

D_W2 = tf.Variable(xavier_init([H_Dim1, H_Dim2]))
D_b2 = tf.Variable(tf.zeros(shape=[H_Dim2]))

D_W3 = tf.Variable(xavier_init([H_Dim2, Dim]))
D_b3 = tf.Variable(tf.zeros(shape=[Dim]))  # Output is multi-variate

# %%为避免相乘出错，直接生成新的变量好了
D_W3_Y = tf.Variable(xavier_init([H_Dim2, 1]))
D_b3_Y = tf.Variable(tf.zeros(shape=[1]))  # Output is multi-variate

theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
theta_Y = [D_W1, D_W2, D_W3, D_W3_Y, D_b1, D_b2, D_b3, D_b3_Y]
# theta_Y = [D_W1[:Dim, :], D_W2, D_W3_Y, D_b1, D_b2, D_b3_Y]

# %% 3. Generator
G_W1 = tf.Variable(xavier_init([Dim * 2, H_Dim1]))  # Data + Mask as inputs (Random Noises are in Missing Components)
G_b1 = tf.Variable(tf.zeros(shape=[H_Dim1]))
G_W2 = tf.Variable(xavier_init([H_Dim1, H_Dim2]))
G_b2 = tf.Variable(tf.zeros(shape=[H_Dim2]))
G_W3 = tf.Variable(xavier_init([H_Dim2, Dim]))
G_b3 = tf.Variable(tf.zeros(shape=[Dim]))
theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]


# %% GAIN Function

# %% 1. Generator
def generator(new_x, m):
    inputs = tf.concat(axis=1, values=[new_x, m])  # Mask + Data Concatenate  # %%%行数不变，列数*2 %%%%% ##3
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)  # [0,1] normalized Output
    return G_prob


# %% 2. Discriminator
def discriminator(new_x, h):
    # 可以把h加上一些东西：

    inputs = tf.concat(axis=1, values=[new_x, h])  # Hint + Data Concatenate    # %%%行数不变，列数*2 %%%%% ##
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)  # [0,1] Probability Output  #每个特征的概率都输出
    ## 增加下面这几句 ##
    D_h1_ = tf.nn.relu(tf.matmul(inputs[:, :Dim], D_W1[:Dim, :]) + D_b1)
    D_h2_ = tf.nn.relu(tf.matmul(D_h1_, D_W2) + D_b2)
    Y_logit = tf.matmul(D_h2_, D_W3_Y) + D_b3_Y
    Y_pred = tf.nn.sigmoid(Y_logit)  # 判断数据类别(0或者1)
    return D_prob, Y_pred


# %% 3. Other functions
# Random sample generator for Z
def sample_Z(m, n):
    return np.random.uniform(0., 0.01, size=[m, n])

# Mini-batch generation
def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx


# %% Structure
# Generator
G_sample = generator(New_X, M)

# Combine with original data
Hat_New_X = New_X * M + G_sample * (1 - M)

# Discriminator
D_prob, Y_pred = discriminator(Hat_New_X, H)  # %%%多了一个Y_pred作为输出参数
Y_real = Y

# %% Loss
# 要修改这些
D_loss1 = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1 - M) * tf.log(1. - D_prob + 1e-8))
G_loss1 = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))
MSE_train_loss = tf.reduce_mean((M * New_X - M * G_sample) ** 2) / tf.reduce_mean(M)  # %%这里是连续型的变量

matrix_U = find_U(Hat_New_X, V)  # 生成的样本丢进去得到隶属度矩阵[n,c]
# sample_U 真实样本的隶属度
# euclidean = tf.sqrt(tf.reduce_sum(tf.square(x3-x4), 2))
D_U_train_loss = tf.sqrt(tf.reduce_sum(tf.square(matrix_U - sample_U), 2))  # 新增一个d的loss

# beta 应该跟隶属度有关系
D_loss = D_loss1 + beta * D_U_train_loss  # 第二项应该侧重于生成数据如果符合聚类的给1，否则给0
# D_loss = D_loss1
G_loss = G_loss1 + alpha * MSE_train_loss

# define y loss
Y_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_real, logits=Y_pred))
# %%%  define Y_solver ###
Y_solver = tf.train.AdamOptimizer().minimize(Y_loss, var_list=theta_Y)

# %% MSE Performance metric
MSE_test_loss = tf.reduce_mean(((1 - M) * X - (1 - M) * G_sample) ** 2) / tf.reduce_mean(1 - M)
# %%这里只计算缺失后数据填充产生的误差

# %% Solver
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

# Sessions
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# %% Start Iterations
for it in tqdm(range(5000)):  # 5000->500
    # %% Inputs
    mb_idx = sample_idx(Train_No, mb_size)
    X_mb = trainX[mb_idx, :]
    # %%% 增加的 ####
    Y_mb = trainY[mb_idx]

    Z_mb = sample_Z(mb_size, Dim)
    M_mb = trainM[mb_idx, :]
    H_mb1 = sample_M(mb_size, Dim, 1 - p_hint)
    H_mb = M_mb * H_mb1

    New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

    # _, D_loss_curr = sess.run([D_solver, D_loss1], feed_dict={M: M_mb, New_X: New_X_mb, H: H_mb})
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={M: M_mb, New_X: New_X_mb, H: H_mb,
                                                             V: sample_v, sample_U: sample_u})
    _, G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = sess.run(
        [G_solver, G_loss1, MSE_train_loss, MSE_test_loss],
        feed_dict={X: X_mb, M: M_mb, New_X: New_X_mb, H: H_mb})

    # %% Intermediate Losses
    if it % 100 == 0:
        print('Iter: {}'.format(it))
        print('Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr)))
        print('Test_loss: {:.4}'.format(np.sqrt(MSE_test_loss_curr)))

# %% Final Loss

Z_mb = sample_Z(Test_No, Dim)
M_mb = testM
X_mb = testX  # %%%真实的数据 ####
New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce  # %%%%%缺失数据，但加了随机噪声  ####
MSE_final, Sample = sess.run([MSE_test_loss, G_sample], feed_dict={X: testX, M: testM, New_X: New_X_mb})

print('Final Test RMSE: ' + str(np.sqrt(MSE_final)))



