比赛试题经验  
1、第一次使用的是Resnet50模型，需要注意的是模型需要与pretrain的model参数要一一对应。  
使用resnet50，进行finetuning训练时，可以将minibatch调节比较大，最好成绩是93.3%，后续经过多次  
测试，没有超过93%的成绩，因而使用后续使用了Resnet101模型。  
2、Resnet101模型大体代码结构跟Resnet50相似，网络层数有些不同。  
使用Resnet101无数据扩充情况下，得到最好成绩是94.16%。   
后续使用数据扩充方法，参考[image_enforcing.py]()对原始训练数据扩充了6倍后继续训练。  
同时第一步model-base保证训练分类准确度在90%左右后停止训练，保存第一步模型。用于后续finetuning训练。  
finetuning训练时有几个改变：  
1、将梯度下降方法改为动量梯度下降方法  
2、学习率减小，减小为10-4  
3、根据最优成绩的标准，写了一个判断训练精度平稳性并自动停止训练的方法，见下  
```python
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
```
根据上述函数，提取连续20个平局值在0.95以上并且max-min <= 0.1的训练段，保存最后一次的模型。  
通过以上操作，获得了两个95%准确率，一个95.41%的准确率，其余提交结果均在94%以上。
通过本次实验总结如下：  
1、最初训练使用adam达到快速收敛，后续精调的时候要用momentun或者SGD细调  
2、第二步训练时学习率尽量设置的低一点，minbatch尽量在可承受范围内扩大  
3、保存好每一步的参数，方便继续训练  
