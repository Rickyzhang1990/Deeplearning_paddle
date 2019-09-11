import os
import shutil
import paddle as paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
import reader
import math
import numpy as np 

# 获取cats数据
train_reader = paddle.batch(reader.train(), batch_size=26)

# 定义残差神经网络（ResNet）
def resnet50(input, class_dim):
    def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1, act=None, name=None):
        conv = fluid.layers.conv2d(input=input,
                                   num_filters=num_filters,
                                   filter_size=filter_size,
                                   stride=stride,
                                   padding=(filter_size - 1) // 2,
                                   groups=groups,
                                   act=None,
                                   param_attr=ParamAttr(name=name + "_weights"),
                                   bias_attr=False,
                                   name=name + '.conv2d.output.1')
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        return fluid.layers.batch_norm(input=conv,
                                       act=act,
                                       name=bn_name + '.output.1',
                                       param_attr=ParamAttr(name=bn_name + '_scale'),
                                       bias_attr=ParamAttr(bn_name + '_offset'),
                                       moving_mean_name=bn_name + '_mean',
                                       moving_variance_name=bn_name + '_variance', )

    def shortcut(input, ch_out, stride, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return conv_bn_layer(input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck_block(input, num_filters, stride, name):
        conv0 = conv_bn_layer(input=input,
                              num_filters=num_filters,
                              filter_size=1,
                              act='relu',
                              name=name + "_branch2a")
        conv1 = conv_bn_layer(input=conv0,
                              num_filters=num_filters,
                              filter_size=3,
                              stride=stride,
                              act='relu',
                              name=name + "_branch2b")
        conv2 = conv_bn_layer(input=conv1,
                              num_filters=num_filters * 4,
                              filter_size=1,
                              act=None,
                              name=name + "_branch2c")

        short = shortcut(input, num_filters * 4, stride, name=name + "_branch1")

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu', name=name + ".add.output.5")

    depth = [3, 4, 6, 3]
    num_filters = [64, 128, 256, 512]

    conv = conv_bn_layer(input=input, num_filters=64, filter_size=7, stride=2, act='relu', name="conv1")
    conv = fluid.layers.pool2d(input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

    for block in range(len(depth)):
        for i in range(depth[block]):
            conv_name = "res" + str(block + 2) + chr(97 + i)
            conv = bottleneck_block(input=conv,
                                    num_filters=num_filters[block],
                                    stride=2 if i == 0 and block != 0 else 1,
                                    name=conv_name)

    pool = fluid.layers.pool2d(input=conv, pool_size=7, pool_type='avg', global_pooling=True)
    output = fluid.layers.fc(input=pool, size=class_dim, act='softmax')
    return output


def resnet101(input,class_dim):
    def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1, act=None, name=None):
        conv = fluid.layers.conv2d(input=input,
                                   num_filters=num_filters,
                                   filter_size=filter_size,
                                   stride=stride,
                                   padding=(filter_size - 1) // 2,
                                   groups=groups,
                                   act=None,
                                   param_attr=ParamAttr(name=name + "_weights"),
                                   bias_attr=False,
                                   name=name + '.conv2d.output.1')
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        return fluid.layers.batch_norm(input=conv,
                                       act=act,
                                       name=bn_name + '.output.1',
                                       param_attr=ParamAttr(name=bn_name + '_scale'),
                                       bias_attr=ParamAttr(bn_name + '_offset'),
                                       moving_mean_name=bn_name + '_mean',
                                       moving_variance_name=bn_name + '_variance', )

    def shortcut(input, ch_out, stride, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return conv_bn_layer(input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck_block(input, num_filters, stride, name):
        conv0 = conv_bn_layer(input=input,
                              num_filters=num_filters,
                              filter_size=1,
                              act='relu',
                              name=name + "_branch2a")
        conv1 = conv_bn_layer(input=conv0,
                              num_filters=num_filters,
                              filter_size=3,
                              stride=stride,
                              act='relu',
                              name=name + "_branch2b")
        conv2 = conv_bn_layer(input=conv1,
                              num_filters=num_filters * 4,
                              filter_size=1,
                              act=None,
                              name=name + "_branch2c")

        short = shortcut(input, num_filters * 4, stride, name=name + "_branch1")

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu', name=name + ".add.output.5")

    depth = [3, 4, 23, 3]
    num_filters = [64, 128, 256, 512]

    conv = conv_bn_layer(input=input, num_filters=64, filter_size=7, stride=2, act='relu', name="conv1")
    conv = fluid.layers.pool2d(input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

    for block in range(len(depth)):
        for i in range(depth[block]):
            if  block == 2:
                if i == 0:
                    conv_name="res"+str(block+2)+"a"
                else:
                    conv_name="res"+str(block+2)+"b"+str(i)
            else:
                conv_name="res"+str(block+2)+chr(97+i)
            conv = bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1, name=conv_name)
    pool = fluid.layers.pool2d(input=conv, pool_size=7, pool_type='avg', global_pooling=True)
    stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
    out = fluid.layers.fc(input=pool,
                       size=class_dim,
                       act = "softmax",
                       param_attr=fluid.param_attr.ParamAttr(
                       initializer=fluid.initializer.Uniform(-stdv, stdv)))
    return out
        

            
# 定义筛选函数
class Opque(list):
    ##定义队列
    def __init__(self,*,means = 0.9 ,dist = 0.1,steps = 20):
        self.queue = []
        self.means = means
        self.dist  = dist 
        self.steps = steps
    
    def push(self ,x)->None:
        self.queue.append(x)
        
    @property   
    def pop(self):
        self.queue = self.queue[10:]
        
    @property
    def isfull(self):
        return len(self.queue) == self.steps 
    @property
    def isempty(self):
        return len(self.queue) == 0 
    @property
    def isteady(self):
        if np.mean(np.array(self.queue)) >= self.means and np.max(np.array(self.queue)) - np.min(np.array(self.queue)) <= self.dist:
            return True 
        else:
            return False
        
    
        
       
# 定义输入层
image = fluid.layers.data(name='image', shape=[3, 224, 224], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# 获取分类器
model = resnet101(image, 12)

# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

# 定义优化方法
optimizer = fluid.optimizer.Momentum(learning_rate=1e-4,momentum=0.9)
opts = optimizer.minimize(avg_cost)


# 定义一个使用GPU的执行器
place = fluid.CUDAPlace(0)

exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 经过step-1处理后的的预训练模型
pretrained_model_path = 'models/s1_8-16_model/'
# 加载经过处理的模型
fluid.io.load_params(executor=exe, dirname=pretrained_model_path)

# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

# 训练10次
que = Opque(means=0.95 ,dist=0,steps= 20)
break_flag = False
for pass_id in range(20):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, acc])
        # 每100个batch打印一次信息
        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))
        que.push(train_acc[0])
        if que.isfull:
            if que.isteady:
                break_flag = True 
                break 
            else:
                que.pop
        else:   pass
    if break_flag == True:
        break 
print("the optimize Accuracy:%0.5f ,batch_id:%d ,Pass:%d"%(train_acc[0] ,batch_id , pass_id))
print("the Accuracy list is\n%s"%que.queue)


# 保存参数模型
save_pretrain_model_path = 'models/s2_815_model/'
# 删除旧的模型文件
shutil.rmtree(save_pretrain_model_path, ignore_errors=True)
# 创建保持模型文件目录
os.makedirs(save_pretrain_model_path)
# 保存参数模型
fluid.io.save_inference_model(save_pretrain_model_path, feeded_var_names=[image.name], target_vars=[model], executor=exe)


