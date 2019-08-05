
#!get_ipython().system('DATA_PATH=data/data9911/ && NEW_NAME=$(find -name *[0-9].ipynb) && NEW_NAME=${NEW_NAME%.*} && NEW_NAME=${NEW_NAME#./} && unzip -o ${DATA_PATH}data_images.zip  && cp -rf data_images/. .')

from __future__ import print_function
import numpy as np                            #导入numpy库
import matplotlib.pyplot as plt               #导入matplotlib作图库
import pandas as pd                           #导入padans库
import paddle
import paddle.fluid as fluid                  #使用paddle fluid版本深度学习库

import math
import sys


colnames = ['房屋面积']+['房价']                                      #数据是一维的，只有面积与房价的关系
print_data = pd.read_csv('./datasets/data.txt',names = colnames)
print_data.head()

# coding = utf-8 #
global x_raw,train_data,test_data
'''
全局变量的使用，首先声明变量而后赋值
'''

data = np.loadtxt('./datasets/data.txt',delimiter = ',')
x_raw = data.T[0].copy()                                   #暂时没发现该步骤的作用

#axis=0,表示按列计算
#data.shape[0]表示data中一共有多少列
maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(axis=0)/data.shape[0]
print("the raw area :",data[:,0].max(axis = 0))
print(avgs[0])
#归一化，data[:,i]表示第i列的元素

### START CODE HERE ### (≈ 3 lines of code)
data[:,0] = (data[:,0] - avgs[0])/(maximums[0]-minimums[0])   #进限于当前只有一个特征的数据，参考答案写的比较好
feature = 2 
for i in range(feature -1):                                        #适合任意特征的归一化
    data[:,i] = (data[:,i] - avgs[i])/(maximums[i] - minimums[i])
# 标准化操作 
dmean ,dstd  = np.mean(data,axis = 0) ,np.std(data , axis = 0)
for i in range(feature -1):
    data[:,i] = (data[: ,i]  - dmean[i])/dstd[i]


### END CODE HERE ###
print(data)
print('normalization:',data[:,0].max(axis = 0))

ratio = 0.8
offset = int(data.shape[0]*ratio)

### START CODE HERE ### (≈ 2 lines of code)
train_data = data[:offset].copy()
test_data  = data[offset:].copy()
### END CODE HERE ###

print(len(data))
print(len(train_data))


def read_data(data_set):
    """
    一个reader
    Args：
        data_set -- 要获取的数据集
    Return：
        reader -- 用于获取训练集及其标签的生成器generator
    """
    def reader():
        """
        一个reader
        Args：
        Return：
            data[:-1],data[-1:] --使用yield返回生成器
                data[:-1]表示前n-1个元素，也就是训练数据，
                data[-1:]表示最后一个元素，也就是对应的标签
        """
        for data in data_set:
            yield data[:-1],data[-1:]
    return reader
# yield的用法，迭代返回固定格式的数据


test_array = ([10,100],[20,200])
print("test_array for read_data:")
for value in read_data(test_array)():
    print(value)




BATCH_SIZE = 32

# 设置训练reader
train_reader = paddle.batch(
    paddle.reader.shuffle(
        read_data(train_data), 
        buf_size=500),
    batch_size=BATCH_SIZE)

#设置测试 reader
test_reader = paddle.batch(
    paddle.reader.shuffle(
        read_data(test_data), 
        buf_size=500),
    batch_size=BATCH_SIZE)





use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace() 



# dtype='float32'：数据类型为float32
### START CODE HERE ### (≈ 1 lines of code)
x = fluid.layers.data(name='x', shape=[1], dtype='float32')

### END CODE HERE ###


# 标签数据，fluid.layers.data表示数据层,name=’y’：名称为y,输出类型为tensor
# shape=[1]:数据为1维向量
### START CODE HERE ### (≈ 1 lines of code)
y = fluid.layers.data(name='y', shape=[1], dtype='float32')

### END CODE HERE ###

# 输出层，fluid.layers.fc表示全连接层，input=x: 该层输入数据为x
# size=1：神经元个数，act=None：激活函数为线性函数
y_predict = fluid.layers.fc(input=x, size=1, act=None)


### START CODE HERE ### (≈ 2 lines of code)
avg_loss = fluid.layers.mean(fluid.layers.square_error_cost(input=y_predict, label=y))

### END CODE HERE ###


# **定义执行器(参数随机初始化):**
# 
# 首先定义执行器，fulid使用了一个C++类Executor用于运行一个程序，Executor类似一个解析器，Fluid将会使用这样一个解析器来训练和测试模型。
# 

# In[33]:


exe = fluid.Executor(place)


# **配置训练程序:**
# 
# ①全局主程序main program。该主程序用于训练模型。
# 
# ②全局启动程序startup_program。
# 
# ③测试程序test_program。用于模型测试。
# 

# In[34]:


main_program = fluid.default_main_program() # 获取默认/全局主函数
startup_program = fluid.default_startup_program() # 获取默认/全局启动程序

#克隆main_program得到test_program
#有些operator在训练和测试之间的操作是不同的，例如batch_norm，使用参数for_test来区分该程序是用来训练还是用来测试
#该api不会删除任何操作符,请在backward和optimization之前使用
test_program = main_program.clone(for_test=True)


# **优化方法:**
# 
# 损失函数定义确定后，需要定义参数优化方法。为了改善模型的训练速度以及效果，学术界先后提出了很多优化算法，包括： Momentum、RMSProp、Adam 等，已经被封装在fluid内部，读者可直接调用。本次可以用 fluid.optimizer.SGD(learning_rate= ) 使用随机梯度下降的方法优化，其中learning_rate表示学习率，大家可以自己尝试修改。
# 
# 

# In[35]:


# 创建optimizer，更多优化算子可以参考 fluid.optimizer()
learning_rate = 0.01
sgd_optimizer = fluid.optimizer.SGD(learning_rate)
sgd_optimizer.minimize(avg_loss)
print("optimizer is ready")


# **训练模型:**
# 
# 上述内容进行了模型初始化、网络结构的配置并创建了训练函数、硬件位置、优化方法，接下来利用上述配置进行模型训练。
# 
# 
# **创建训练过程:**
# 
# 训练需要有一个训练程序和一些必要参数，并构建了一个获取训练过程中测试误差的函数。必要参数有executor,program,reader,feeder,fetch_list，executor表示之前创建的执行器，program表示执行器所执行的program，是之前创建的program，如果该项参数没有给定的话则默认使用defalut_main_program，reader表示读取到的数据，feeder表示前向输入的变量，fetch_list表示用户想得到的变量或者命名的结果。

# In[36]:


# For training test cost
def train_test(executor, program, reader, feeder, fetch_list):
    accumulated = 1 * [0]
    count = 0
    for data_test in reader():
        outs = executor.run(
            program=program, feed=feeder.feed(data_test), fetch_list=fetch_list)
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)]  # 累加测试过程中的损失值
        count += 1 # 累加测试集中的样本数量
    return [x_d / count for x_d in accumulated] # 计算平均损失


# In[17]:


#定义模型保存路径：
#params_dirname用于定义模型保存路径。
params_dirname = "easy_fit_a_line.inference.model"


# **训练主循环**
# 
# 我们构建一个循环来进行训练，直到训练结果足够好或者循环次数足够多。 如果训练迭代次数满足参数保存的迭代次数，可以把训练参数保存到params_dirname。 设置训练主循环

# In[42]:



#用于画图展示训练cost
from paddle.utils.plot import Ploter
train_prompt = "Train cost"
test_prompt = "Test cost"
plot_prompt = Ploter(train_prompt, test_prompt)
step = 0

# 训练主循环
feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
exe.run(startup_program)

exe_test = fluid.Executor(place)

#num_epochs=100表示迭代训练100次后停止训练。
num_epochs = 100

for pass_id in range(num_epochs):
    for data_train in train_reader():
        avg_loss_value, = exe.run(main_program,
                                  feed=feeder.feed(data_train),
                                  fetch_list=[avg_loss])
        if step % 10 == 0:  # 每10个批次记录并输出一下训练损失
            plot_prompt.append(train_prompt, step, avg_loss_value[0])
            plot_prompt.plot()
            #print("%s, Step %d, Cost %f" %(train_prompt, step, avg_loss_value[0]))
        if step % 100 == 0:  # 每100批次记录并输出一下测试损失
            test_metics = train_test(executor=exe_test,
                                     program=test_program,
                                     reader=test_reader,
                                     fetch_list=[avg_loss.name],
                                     feeder=feeder)
            plot_prompt.append(test_prompt, step, test_metics[0])
            plot_prompt.plot()
            #print("%s, Step %d, Cost %f" %(test_prompt, step, test_metics[0]))
            
            if test_metics[0] < 10.0: # 如果准确率达到要求，则停止训练
                break

        step += 1

        if math.isnan(float(avg_loss_value[0])):
            sys.exit("got NaN loss, training failed.")

        #保存训练参数到之前给定的路径中
        if params_dirname is not None:
            fluid.io.save_inference_model(params_dirname, ['x'], [y_predict], exe)


# 得到的cost函数变化图像大致应是一个收敛的结果：
# <img src='images/result.png' style = "width:400px;height:300px;">

# # 4 - 预测
# 
# 
# 预测器会从params_dirname中读取已经训练好的模型，来对从未遇见过的数据进行预测。  
# 通过fluid.io.load_inference_model，预测器会从params_dirname中读取已经训练好的模型，来对从未遇见过的数据进行预测。

# In[43]:


infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()


# **预测**
# 
# 预测器会从params_dirname中读取已经训练好的模型，来对从未遇见过的数据进行预测。
# 
# - tensor_x:生成batch_size个[0,1]区间的随机数，以 tensor 的格式储存
# - results：预测对应 tensor_x 面积的房价结果
# - raw_x:由于数据处理时我们做了归一化操作，为了更直观的判断预测是否准确，将数据进行反归一化，得到随机数对应的原始数据。

# In[44]:


with fluid.scope_guard(inference_scope):
    [inference_program, feed_target_names, fetch_targets
     ] = fluid.io.load_inference_model(params_dirname, infer_exe) # 载入预训练模型


    batch_size = 2
    tensor_x = np.random.uniform(0, 1, [batch_size, 1]).astype("float32")
    
    print("tensor_x is :" ,tensor_x )
    results = infer_exe.run(
        inference_program,
        feed={feed_target_names[0]: tensor_x},
        fetch_list=fetch_targets) # 进行预测
    raw_x = tensor_x*(maximums[0]-minimums[0])+avgs[0]
    print("the area is:",raw_x)
    print("infer results: ", results[0])
    


# 此处应得到一组预测结果：
# 
# ('the area is:', array([[####],
# 
#        [####]], dtype=float32))
#        
# ('infer results: ', array([[####],
# 
#        [####]], dtype=float32))
# 

# 根据线性模型的原理，补全输出公式，计算a和b的值
# 
# - 提示：已知两点求直线方程
# 

# In[45]:



a = (results[0][0][0] - results[0][1][0]) / (raw_x[0][0]-raw_x[1][0])
b = (results[0][0][0] - a * raw_x[0][0])

print(a,b)


# 预测结果应为：a=6.7,b=-24.42(每次训练结果取随机数，因此得到的结果可能会有一点点偏差，但大致应在这个范围之间）,因此本次模型得到的房屋面积与房价之间的拟合函数为$y=6.7x-24.42$。其中y为预测的房屋价格，x为房屋面积，根据这个公式可以推断：如果有500万的预算，想在该地区购房，房屋面积大概为$\frac{500-(-24.42)}{6.7}=78(m^2)$。

# **（5）绘制拟合图像 **
# 
# 通过训练，本次线性回归模型输出了一条拟合的直线，想要直观的判断模型好坏可将拟合直线与数据的图像绘制出来。
# 
# 

# In[46]:


import numpy as np
import matplotlib.pyplot as plt

def plot_data(data):
    x = data[:,0]
    y = data[:,1]
    y_predict = x*a + b
    plt.scatter(x,y,marker='.',c='r',label='True')
    plt.title('House Price Distributions')
    plt.xlabel('House Area ')
    plt.ylabel('House Price ')
    plt.xlim(0,250)
    plt.ylim(0,2500)
    predict = plt.plot(x,y_predict,label='Predict')
    plt.legend(loc='upper left')
    plt.savefig('result1.png')
    plt.show()

data = np.loadtxt('./datasets/data.txt',delimiter = ',')
plot_data(data)



