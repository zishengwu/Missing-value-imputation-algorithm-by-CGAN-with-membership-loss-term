# %% Packages
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

# %% System Parameters
# 1. Mini batch size
mb_size = 10
# 2. Missing rate
p_miss = 0.1
# 3. Hint rate
p_hint = 0.9           # 后面函数生成的M大约有9成数值为1
# 4. Loss Hyperparameters
alpha = 10
# 5. Train Rate
train_rate = 0.8

c = 4
# car:4  wine:7   german:2   vs:4
# %% Data

Data = pd.read_excel('E:\PyCharm 2018.2.1\my project\dataset\Vehicle Silhouettes\Vehicle Silhouettes.xlsx')
Data = np.array(Data)

# 给一个常数来改变训练的多少
fortrain = 690

# 加载训练样本的聚类中心和隶属度
sample_u = pd.read_excel('E:\\PyCharm 2018.2.1\\my project\\dataset\\Vehicle Silhouettes\\u0.1.xlsx')
sample_v = pd.read_excel('E:\\PyCharm 2018.2.1\\my project\\dataset\\Vehicle Silhouettes\\v0.1.xlsx')
sample_u = np.array(sample_u)
sample_v = np.array(sample_v)
sample_u = sample_u.astype(np.float32)
sample_v = sample_v.astype(np.float32)

# Parameters
No = len(Data)
Dim = len(Data[0, :-c])

# Hidden state dimensions
H_Dim1 = Dim
H_Dim2 = Dim


# %% Missing introducing
p_miss_vec = p_miss * np.ones((Dim, 1))

Missing = np.zeros((No, Dim))

for i in range(Dim):
    A = np.random.uniform(0., 1., size=[len(Data), ])
    B = A > p_miss_vec[i]
    Missing[:, i] = 1. * B

# Train / Test Features
trainX = Data[:fortrain, :-c]
testX = Data[fortrain:, :-c]


# Train / Test Missing Indicators
trainM = Missing[:fortrain, :]
testM = Missing[fortrain:, :]


# 1. Xavier Initialization Definition
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# Hint Vector Generation
def sample_M(m, n, p):
    A = np.random.uniform(0., 1., size=[m, n])   # 生成0~1之间的随机数
    B = A > p
    C = 1. * B   # 将逻辑值转化为0跟1！！！
    return C


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



# %% 2. Discriminator
D_W1 = tf.Variable(xavier_init([Dim * 2, H_Dim1]))  # Data + Hint as inputs
D_b1 = tf.Variable(tf.zeros(shape=[H_Dim1]))

D_W2 = tf.Variable(xavier_init([H_Dim1, H_Dim2]))
D_b2 = tf.Variable(tf.zeros(shape=[H_Dim2]))

D_W3 = tf.Variable(xavier_init([H_Dim2, Dim]))
D_b3 = tf.Variable(tf.zeros(shape=[Dim]))  # Output is multi-variate

theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

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
    inputs = tf.concat(axis=1, values=[new_x, h])  # Hint + Data Concatenate    # %%%行数不变，列数*2 %%%%% ##
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)  # [0,1] Probability Output  #每个特征的概率都输出
    return D_prob


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
D_prob = discriminator(Hat_New_X, H)  # %%%多了一个Y_pred作为输出参数

# %% Loss
D_loss1 = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1 - M) * tf.log(1. - D_prob + 1e-8))
G_loss1 = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))
MSE_train_loss = tf.reduce_mean((M * New_X - M * G_sample) ** 2) / tf.reduce_mean(M)    # %%这里是连续型的变量

D_loss = D_loss1
G_loss = G_loss1 + alpha * MSE_train_loss

# %% MSE Performance metric
MSE_test_loss = tf.reduce_mean(((1 - M) * X - (1 - M) * G_sample) ** 2) / tf.reduce_mean(1 - M)
# %%这里只计算缺失后数据填充产生的误差

# %% Solver
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)


# Sessions
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# %% Iterations

# %% Start Iterations
for it in tqdm(range(5000)):

    # %% Inputs
    mb_idx = sample_idx(fortrain, mb_size)
    X_mb = trainX[mb_idx, :]
    Z_mb = sample_Z(mb_size, Dim)
    M_mb = trainM[mb_idx, :]
    H_mb1 = sample_M(mb_size, Dim, 1 - p_hint)
    H_mb = M_mb * H_mb1

    New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

    _, D_loss_curr = sess.run([D_solver, D_loss1], feed_dict={M: M_mb, New_X: New_X_mb, H: H_mb})
    _, G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = sess.run(
        [G_solver, G_loss1, MSE_train_loss, MSE_test_loss],
        feed_dict={X: X_mb, M: M_mb, New_X: New_X_mb, H: H_mb})


    # %% Intermediate Losses
    if it % 100 == 0:
        print('Iter: {}'.format(it))
        print('Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr)))
        print('Test_loss: {:.4}'.format(np.sqrt(MSE_test_loss_curr)))

# %% Final Loss

Z_mb = sample_Z(No-fortrain, Dim)
M_mb = testM
X_mb = testX  # %%%真实的数据 ####


New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce  # %%%%%缺失数据，但加了随机噪声  ####
MSE_final, Sample = sess.run([MSE_test_loss, G_sample], feed_dict={X: testX, M: testM, New_X: New_X_mb})

print('Final Test RMSE: ' + str(np.sqrt(MSE_final)))




