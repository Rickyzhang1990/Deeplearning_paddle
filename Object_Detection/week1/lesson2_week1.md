## RCNN回顾   
Regions with CNN features  
缺点：
训练过程阶段多   
训练时间和空间开销大
目标检测速度慢  
R-CNN需要对每个区域分别执行CNN正向传播过程，并没有考虑共享计算  
SPP NET   
不需要进行resize   
只需要进行一次CNN   
缺点 多阶段；无法向后传播  
TasteRCNN   
1、输入图片和候选框   
2、生成一个conv  feature map   
3、对每个BBox,生成a fix-length feature vector   
4、输出两个信息：k+1分类标签  bBox的位置  
损失函数：将bbox和分特征类的损失函数合二为一   
FastRCNN的主要贡献   
更好的检测效果（mAP）  
训练时但不的，使用多任务loss  
所有的网络的参数都可以训练和更新   
不需要存储特征  
## ROI  pooling技术直观理解   
把多重SPP变成一层的SPP  
输入时途中的一个ROI区域   
输出是一个固定尺寸的特征图   
总结：   
1、用于目标检测任务   
2、对CNN 中的fearture map工作  
3.允许end-to-end的形式训练目标检测系统  
##  映射关系  
卷积不会是特征图变小，pooling是图像面积为原来的一半  
## Faster RCNN  
RPN是单独的一个网络，RPN生成300个候选框  
k-anchor boxes（锚框）  
 

