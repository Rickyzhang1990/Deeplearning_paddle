

   ### 1 LeNet是第一个成功的CNN应用，请问LeNet 没有使用下列哪种结构：  
   A.CNN   B.下采样 C.全连接   :heavy_check_mark:D.残差    
   解释：LeNET是96年由lecun开发的网络模型，是深度神经网络的起始，使用了全连接，卷积池化等技术。残差结构则是有Resnet正式使用。  
   ### 2 AlexNet带来了深度学习技术的蓬勃发展，请问AlexNet 有几个卷积层：  
   A.3  B.4  :heavy_check_mark:C.5  D.6  
   ### 3 AlexNet没有使用的技术有：  
   A.卷积 B.池化 C.dropout :heavy_check_mark:D.1\*1卷积核    
   解释：AlexNET是2012年ImageNET大赛上获得冠军的模型，主要贡献是大量使用了Relu激活函数和dropout技术，1\*1卷积核在GoogleNET中被大量使用。  
   ### 4 使用GPU的主要原因是：  
   :heavy_check_mark: A.加速计算   B.扩大内存   C.增加数据集   D.防止过拟合  
   ### 5 VGG是一种经典的网络结构，下列说法中正确的是：   
   A. VGG-16和VGG-19 没有使用 全连接层  
   B. VGG主要是在宽度方面进行了探索   
   :heavy_check_mark:C. VGG 证明网络深度越深越好，所以可以无限制的加深网络  
   D. VGG 尽可能使用了小卷积核，例如：3\*3、1\*1  
   ### 6 小卷积核有自己优势以下关于小卷积核的表述不正确的是：  
   :heavy_check_mark:A. 卷积核小方便 padding 的使用  
   B. 5\*5 的卷积核可以用两个 3\*3 的卷积核替代   
   C. 7\*7 的卷积核可以用三个 3\*3 的卷积核替代   
   D. 小卷积核相较大卷积核能够加速计算同时能够引入更多的非线性  
   ### 7 GoogleNet 的第一个版本使用了 inception V1 结构，关于这个结构描述不正确的有：  
   :heavy_check_mark:A. 该结构证明网络越深越好   
   B. 该结构进行了宽度方面的探索   
   C. 该结构使用 1\*1 卷积核，并且效果较好  
   D. 该结构使用没有残差技术，而是将若干卷积结果拼接  
   ### 8 下面关于1*1卷积核表述不正确的是：  
   A. 1\*1卷积核可以做到升维  
   B. 1\*1卷积核可以做到降维   
   C. 1\*1卷积核可以做到引入更多的非线性  
   :heavy_check_mark:D.1\*1卷积核除了浪费计算时间毫无作用  
   ### 9 Dropout 技术描述不正确的有：  
   A.延长了计算时间  
   B.有效降低过拟合  
   C.目前已经被美国某公司申请为专利  
   :heavy_check_mark:D.Dropout 每次运算都删除一些神经元，所以网络规模越来越小  
   ### 10 下面网络结构最深的：
   A.LeNet  B.ZF-Net  C.VGG  :heavy_check_mark:D.GoogleNet    
   解释：本道题我答错了，GoogleNET在网络宽度上进行了探索，同时在网络深度上也高于其他网络，VGG中最多只有19层。
   
