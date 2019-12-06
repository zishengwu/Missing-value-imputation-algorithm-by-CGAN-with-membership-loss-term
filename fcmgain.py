# %% Packages
import tensorflow as tf
import numpy as np
from tqdm import tqdm

# %% System Parameters
# 1. Mini batch size
mb_size = 128
# 2. Missing rate
p_miss = 0.2
# 3. Hint rate
p_hint = 0.9
# 4. Loss Hyperparameters
alpha = 10
# 5. Train Rate
train_rate = 0.8

# %% Data

# Data generation
Data = np.loadtxt('E:\PyCharm 2018.2.1\my project\Letter_with_label.csv', delimiter=",",
                  skiprows=1)

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
    A = np.random.uniform(0., 1., size=[m, n])
    B = A > p
    C = 1. * B
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
# ##### Data Label ######
Y = tf.placeholder(tf.float32, shape=[None])  # 暂时只考虑二分类情况



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
# theta_D = [D_W1, D_W2, D_W3, D_W3_Y, D_b1, D_b2, D_b3, D_b3_Y]
theta_Y = [D_W1[:Dim, :], D_W2, D_W3_Y, D_b1, D_b2, D_b3_Y]

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
    D_prob = tf.nn.sigmoid(D_logit)  # [0,1] Probability Output
    ## 增加下面这几句 ##
    D_h1_ = tf.nn.relu(tf.matmul(inputs[:, :Dim], D_W1[:Dim, :]) + D_b1)
    D_h2_ = tf.nn.relu(tf.matmul(D_h1_, D_W2) + D_b2)
    Y_logit = tf.matmul(D_h2_, D_W3_Y) + D_b3_Y
    Y_pred = tf.nn.sigmoid(Y_logit)  ## 判断数据类别 （0或者1）
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
D_loss1 = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1 - M) * tf.log(1. - D_prob + 1e-8))
G_loss1 = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))
MSE_train_loss = tf.reduce_mean((M * New_X - M * G_sample) ** 2) / tf.reduce_mean(M)    # %%这里是连续型的变量

D_loss = D_loss1
G_loss = G_loss1 + alpha * MSE_train_loss

## define y loss
# ylog(y')+(1-y)log(1-y') ：迫使y pred ~= y real
# Y_loss = -tf.reduce_mean(Y_pred * tf.log(Y_real + 1e-8) + (1 - Y_pred) * tf.log(1. - Y_real + 1e-8))
Y_loss = -tf.reduce_mean(Y_real * tf.log(Y_pred + 1e-8) + (1 - Y_real) * tf.log(1. - Y_pred + 1e-8))
# %%%  define Y_solver ###
# Y_solver = tf.train.AdamOptimizer().minimize(Y_loss, var_list=theta_Y)
Y_solver = tf.train.GradientDescentOptimizer(0.01).minimize(Y_loss, var_list=theta_Y)
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
for it in tqdm(range(500)):  # 5000->500

    # %% Inputs
    mb_idx = sample_idx(Train_No, mb_size)
    X_mb = trainX[mb_idx, :]
    # %%% 增加的 ####
    Y_mb = trainY[mb_idx]
    # Y_mb = np.reshape(Y_mb, [128, 1])

    Z_mb = sample_Z(mb_size, Dim)
    M_mb = trainM[mb_idx, :]
    H_mb1 = sample_M(mb_size, Dim, 1 - p_hint)
    H_mb = M_mb * H_mb1

    New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

    _, D_loss_curr = sess.run([D_solver, D_loss1], feed_dict={M: M_mb, New_X: New_X_mb, H: H_mb})
    _, G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = sess.run(
        [G_solver, G_loss1, MSE_train_loss, MSE_test_loss],
        feed_dict={X: X_mb, M: M_mb, New_X: New_X_mb, H: H_mb})
    # # %% 增加 ###
    _, Y_loss_curr = sess.run([Y_solver, Y_loss], feed_dict={Y: Y_mb, M: M_mb, New_X: New_X_mb, H: H_mb})  # 可能还需要其他参数
                                                                            # %%% 增加了M: M_mb,New_X: New_X_mb, H: H_mb

    # %% Intermediate Losses
    if it % 100 == 0:
        print('Iter: {}'.format(it))
        print('Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr)))
        print('Test_loss: {:.4}'.format(np.sqrt(MSE_test_loss_curr)))
        # %%% 增加 ###
        print('Predict_loss: {:.4}'.format(np.sqrt(Y_loss_curr)))
        print()

# %% Final Loss

Z_mb = sample_Z(Test_No, Dim)
M_mb = testM
X_mb = testX  # %%%真实的数据 ####


New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce  # %%%%%缺失数据，但加了随机噪声  ####
MSE_final, Sample = sess.run([MSE_test_loss, G_sample], feed_dict={X: testX, M: testM, New_X: New_X_mb})
New_genera_data = M_mb * X_mb + (1-M_mb) * Sample  # Sample 完全是新生成的，我只要抽取出缺失部分填充的就好！！

print('Final Test RMSE: ' + str(np.sqrt(MSE_final)))

# %%%%% 新增y_mb ########

# un_, test_out = discriminator(New_genera_data, np.zeros([len(X_mb), Dim], dtype=tf.float64))
un_, test_out = discriminator(X, H)
_, test_out = sess.run([un_, test_out], feed_dict={X: New_genera_data, H: np.zeros([len(X_mb), Dim])})
Y_mb = testY  # 真实标签，用来对比生成输出后判别器的准确度

Y_test = test_out > 0.5
# num_error = tf.abs(Y_mb - Y_test)
# num_error = tf.reduce_sum(num_error)
sess.run(tf.global_variables_initializer())  # 这句一定要加，不然无法打印实际值
# print(sess.run(num_error[:5]))
print(testY)
print(test_out)
print(Y_test)


