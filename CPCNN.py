# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 19:35:09 2021

@author: bq
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

import os
import sys

input1 = sys.argv[1]
input2 = sys.argv[2]
input3 = sys.argv[3]
output1 = sys.argv[4]

##这个函数相当于读取fa格式数据
def readFa(fa):
    '''
    @msg: 读取一个fasta文件
    @param fa {str}  fasta 文件路径
    @return: {generator} 返回一个生成器，能迭代得到fasta文件的每一个序列名和序列
    '''
    with open(fa,'r') as FA:
        seqName,seq='',''
        while 1:
            line=FA.readline()
            line=line.strip('\n')
            if (line.startswith('>') or not line) and seqName:
                yield((seqName,seq))
            if line.startswith('>'):
                seqName = line[1:]
                seq=''
            else:
                seq+=line
            if not line:break
 ##-----------------------------------碱基编码--------------------------------------------------------           
            
 ##读取训练集和测试集
data_ncrna_training=pd.DataFrame(readFa(input1))
data_cds_training=pd.DataFrame(readFa(input2))
data_testing=pd.DataFrame(readFa(input3))
##training,testing给碱基编码用，储存其相关内容，
data_cds_training[2]=0
data_ncrna_training[2]=0
data_testing[2]=0

data_cds_training[3]=0
data_ncrna_training[3]=1
data_testing[3]=0

training=pd.concat([data_cds_training,data_ncrna_training],ignore_index=True)
testing=data_testing

##这两个函数的作用分别是将RNA类型转化为列表与编码
def list_trans(data):
    data.iloc[:,1] = data.iloc[:,1].apply(lambda x :list(x))
    
def encode(x):
    x = np.array(x).reshape(len(x),-1)
    enc = OneHotEncoder()
    enc.fit(x)
    targets = enc.transform(x).toarray()
    return targets
#将训练集和测试集的碱基分割出来，即字符串分割        
list_trans(training)
list_trans(testing)

##开始初步编码
training.iloc[:,2] = training.iloc[:,1].apply(lambda x :encode(x))
testing.iloc[:,2]=testing.iloc[:,1].apply(lambda x :encode(x))


##查看编码之后的数据框
training

### 因为有的RNA里含有N，会编码成(0,0,0,1,0)，不符合我们的要求，将含有N的RNA序列筛选出来，重新编码
def recode1(x):
    idex=[]
    for i in range(0,len(x)):
        if x.iloc[i,2][1].shape ==(5,):
            idex.append(i)
    return(idex)
    
#对测试集和训练集筛选出含有N的RNA
idex_training=recode1(training)
idex_testing=recode1(testing)


#查看筛选出来的RNA，以training为例
training.iloc[idex_training,1]


##定义可以对N进行我们想要的编码的编码函数
def encode2(x):
    y=np.ones((len(x),4))
    for j in range(0,len(x)):
        if x[j]=='N':
            y[j]=np.array([1/4,1/4,1/4,1/4])
        elif x[j]=='A':
            y[j]=np.array([1,0,0,0])
        elif x[j]=='C':
            y[j]=np.array([0,1,0,0])
        elif x[j]=='G':
            y[j]=np.array([0,0,1,0])
        else:
            y[j]=np.array([0,0,0,1])
    return y
##对含有N的RNA重新编码

training.iloc[idex_training,2]=training.iloc[idex_training,1].apply(lambda x:encode2(x))
testing.iloc[idex_testing,2]=testing.iloc[idex_testing,1].apply(lambda x:encode2(x))


##查看含有N的RNA重新编码之后的样子：
training.iloc[idex_training,2]



###由于有全连接层的存在要求输入的样本大小，在这里就是每一条RNA的长度是固定的，选择不影响分类的0填充，使其大小一致。

##得到测试集和训练集中最长的RNA含有多少个碱基，我们把其余的RNA补齐，具体表现为把差的碱基数编码为[0,0,0,0]
a=max(training.iloc[:,1].apply(lambda x:len(x)));print(a)
b=max(testing.iloc[:,1].apply(lambda x:len(x)));print(b)
a=max(a,b);print(a)


##该函数的作用为对每一条RNA进行扩充：具体是：已知最大长度，可以选择堆叠的方法
##堆叠上[0,0,0,0]，堆叠之后的长度都应是a 首先创建一个[0,0,0,0]的，然后再合并
def rshape(x):
    dim=a-len(x)
    y=np.zeros((dim,4))#创建一个dim*4的矩阵，里面用0填充
    x=np.vstack((x,y))#将x和y堆叠成新矩阵
    return x
   


##直接用一个大矩阵存储扩充后的RNA

X0=np.zeros((len(training),a,4))
X1=np.zeros((len(testing),a,4))
Y0=np.zeros((len(training),1))
Y1=np.zeros((len(testing),1))

for i in range(0,len(training)):
    X0[i]=rshape(training.iloc[i,2])
    Y0[i]=training.iloc[i,3]
for i in range(0,len(testing)):
    X1[i]=rshape(testing.iloc[i,2])
    Y1[i]=testing.iloc[i,3]
    
##将Y0,Y1弄成分类的，得到的y0,y1即为最后送入网络的训练，测试标签
from tensorflow.keras.utils import to_categorical
y0= to_categorical(Y0)
y1=to_categorical(Y1)   

#因为送入网络需要有四个维度（只有三个维度会报错）
X0=X0.reshape(len(X0),a,4,1)
X1=X1.reshape(len(X1),a,4,1)                        



#-----------------------------密码子编码--------------------------------------
##读取训练集和测试集
data_ncrna_training=pd.DataFrame(readFa(input1))
data_cds_training=pd.DataFrame(readFa(input2))
data_testing=pd.DataFrame(readFa(input3))
##给编码能力标签及扩充数据框，合成总的训练集，测试集
data_cds_training[2]=0
data_ncrna_training[2]=0
data_testing[2]=0

data_cds_training[3]=0
data_ncrna_training[3]=1
data_testing[3]=0

training_1=pd.concat([data_cds_training,data_ncrna_training],ignore_index=True)
testing_1=data_testing

##分割密码子
import re
def split_condons(x):
    all_codons = re.findall('.{3}',x)
    return all_codons

training_1.iloc[:,1]=training_1.iloc[:,1].apply(lambda x:split_condons(x))
testing_1.iloc[:,1]=testing_1.iloc[:,1].apply(lambda x:split_condons(x))


##查看 
training_1

##先给64个密码子64个标签，且将含有N的编码子的标签设置为65
from sklearn import preprocessing
##找一个含有64种密码子的序列，这个时候才可以设置64个标签，此时le就可以给出密码子的标签
for i in range(0,len(training_1)):
    le = preprocessing.LabelEncoder()
    le.fit(training_1.iloc[i,1])
    if len(le.classes_)==64:
        break
    
    
##检查一下我们用来造标签的那个序列的64个密码子是不是真的密码子，即是否都不含N
le.classes_


##最终的标签函数，有的密码子里含N，我们给其一个标签65，其他的照样编码，这里的X为一条切割密码子后的序列
def lable(x):
    lab = np.zeros(len(x))
    for i in range(0,len(x)):
        if 'N' in x[i]:
            lab[i]=65
        else:
            lab[i]=le.transform(x[i:i+1])
    return lab       

#进行最终标签
training_1.iloc[:,2]=training_1.iloc[:,1].apply(lambda x:lable(x))
testing_1.iloc[:,2]=testing_1.iloc[:,1].apply(lambda x:lable(x))

#查看训练集
training


##进行One-Hot编码：正常密码子正常编码 含N的密码子即标签为65的编码成一个每个元素都是1的64维向量
##运行完以下代码后，b为0~63不同取值，这样才可以有分类的独热向量，一个值对应一个独热向量
b=np.empty([64,1], dtype = float, order = 'C')
for i in range(0,64):
    b[i]=i

#b

#enc即可用来转化独热向量
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()

enc.fit(b)


##training_1[4]放其one-hot编码
training_1[4]=0
testing_1[4]=0


##标签为65的即含有N，我们将其编码为一个含有64个元素的向量，每个元素为1，其它照常编码
def OneHot(x):
    x=x.reshape(-1,1)
    if x[0]==65:
        d=np.ones([1,64], dtype = float, order = 'C')
    else:
        d=enc.transform([x[0]]).toarray() 
    for i in range(1,len(x)):
        if x[i]==65:
            d1=np.ones([1,64], dtype = float, order = 'C')
        else:
            d1=enc.transform([x[i]]).toarray() 
        d=np.vstack((d,d1))    
    return d 

training_1.iloc[:,4]=training_1.iloc[:,2].apply(lambda x:OneHot(x))
testing_1.iloc[:,4]=testing_1.iloc[:,2].apply(lambda x:OneHot(x))

##得到测试集和训练集中含有最多密码子个数的序列，我们把其余的RNA补齐，具体表现为把差的密码子数编码为元素都是0的64维向量
a_1=max(training_1.iloc[:,1].apply(lambda x:len(x)));print(a_1)
b_1=max(testing_1.iloc[:,1].apply(lambda x:len(x)));print(b_1)
a_1=max(a_1,b_1);print(a_1)


##该函数的作用为对每一条RNA的密码子进行扩充：具体是：已知最大长度，可以选择堆叠的方法
##堆叠上64维零向量，堆叠之后的长度都应是a 首先创建一个64维零向量的，然后再合并
def rshape_1(x):
    dim=a_1-len(x)
    y=np.zeros((dim,64))#创建一个dim*4的矩阵，里面用0填充
    x=np.vstack((x,y))#将x和y堆叠成新矩阵
    return x

##直接用一个大矩阵,加上_1区分碱基编码和密码子编码，存储扩充后的RNA

X0_1=np.zeros((len(training_1),a_1,64))
X1_1=np.zeros((len(testing_1),a_1,64))


for i in range(0,len(training_1)):
    X0_1[i]=rshape_1(training_1.iloc[i,4])
    
for i in range(0,len(testing_1)):
    X1_1[i]=rshape_1(testing_1.iloc[i,4])
    
    


#因为送入网络需要有四个维度（只有三个维度会报错）
X0_1=X0_1.reshape(len(X0_1),a_1,64,1)
X1_1=X1_1.reshape(len(X1_1),a_1,64,1)


#--------------网络---------------------------------------
from keras.layers import Activation,Conv2D,Dense
from keras.layers import Dropout,Flatten,Input,MaxPooling2D,concatenate
from keras import Model

import sklearn
from sklearn.metrics import roc_auc_score

M = 100
K = 6

##定义输入的两种类型：
inputA=Input([a,4,1])
inputB=Input([a_1,64,1])
#在A（碱基上的分支）
x = Conv2D(M,[K,4],activation='relu')(inputA)
x = MaxPooling2D([a-K+1,1])(x)
x = Flatten()(x)
x = Model(inputs=inputA, outputs=x)

#在B（密码子）上的分支
y = Conv2D(M,[K,64],activation='relu')(inputB)
y = MaxPooling2D([a_1-K+1,1])(y)
y = Flatten()(y)
y = Model(inputs=inputB, outputs=y)

#结合这两个分支的输出
combined = concatenate([x.output, y.output])

#全连接层
z = Dense(40,activation='relu')(combined)
z = Dense(2,activation='softmax')(z)

##合成整个模型

model = Model(inputs=[x.input, y.input], outputs=z)

model.summary()

#模型编译
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
tf.config.experimental_run_functions_eagerly(True)

#model.fit([X0, X0_1], y0, validation_data=([X1, X1_1],y1), epochs=10, batch_size=100)

model.fit([X0, X0_1],y0,epochs=10, batch_size=100, validation_split=0.2)
Y_pred = model.predict([X1, X1_1], batch_size=1)

fw = open(output1+'Results.txt','w')
for i in range(0,len(y1)):
    fw.write(testing.iloc[i,0]+'\n')
    fw.write(str(Y_pred[i,1])+'\t'+str(Y_pred[i,0])+'\n')
fw.close()

#auc = sklearn.metrics.roc_auc_score(y1, Y_pred)
#fw1 = open('./LncRNA/results_Integrate_Model/Dog_lncRNA_CNN_Long_Results.txt','a')
#fw1.write('K = '+str(K)+', M = '+str(M)+', rep'+str(r)+', AUROC = '+str(auc)+'\n')

