#ML hw1
#second order 
#with normalization
# CO O3 pm2.5 windspeed


import csv
import numpy as np
import math
import sys

pollution_type=18
num_hour=24
num_month=12
num_day=20
num_total=pollution_type*num_hour*num_day*num_month

num_row=num_hour*num_month*num_day
num_col=pollution_type

def getLoss_1st(w,b,data,Loss) : 
    L=0
    for i in range(len(data)):
        L+=(data[i][1]-(b+np.dot(data[i][0],w)))**2
    if len(data)!=0:
        L=math.sqrt(L/len(data))
    print(L)
    Loss.append(L)
    return Loss    

def readTraining_choose(fileName,pollution,pos_pm2) :#can use to choose different pollution
    csvfile=open(fileName,'rb')  
    data = []
    k=0

    for row in csv.reader(csvfile,delimiter=','):
        tmp=[]
        if (k%pollution_type in pollution) and (k != 0):
            row.pop(0)
            row.pop(0)
            row.pop(0)
            for  i in range(len(row)) :       
                if row[i]=='NR':
                    tmp.append(float(0))
                else : 
                    tmp.append(float(row[i]))
            data.append(tmp)
        k+=1
    data=np.array(data,dtype='float')
    data=np.transpose(data)

    data_ref=[]
    data_col=num_month*num_day*(len(pollution))
    
    for i in range(0,data_col,len(pollution)) :
        for j in range(num_hour) :
            data_ref.append(data[j][i:i+len(pollution)])

    data=np.array(data_ref,dtype='float')
    
    data_tuple=[]
    for i in range(0,num_row,480):
        for m in range(i,(i+480-9)) :
            tmp=[]
            for j in range(m,m+9) :
                tmp.extend(data[j])
            
            data_tuple.append([np.array(tmp),data[m+9][pos_pm2]])
        
    return data_tuple

def normalization(data) :
    means=np.zeros(len(data[0][0]))
    var=np.zeros(len(data[0][0]))
    #calculate means
    for i in range (len(data)) :
        for j in range(len(data[0][0])) :
            means[j]+=(data[i][0][j])
        #means[len(means)-1]+=data[i][1]
    means=(means/len(data))
    #calculate variance
    for i in range(len(data)) :
        for j in range (len(data[0][0])) :
            var[j]+=(data[i][0][j]-means[j])**2
        #var[len(means)-1]+=(data[i][1]-means[j])**2
    
    sigma=np.sqrt(var/len(data))
    #normalization
    for i in range (len(data)) :
        for j in range (len(data[0][0])) :
            data[i][0][j]=((data[i][0][j]-means[j])/sigma[j])
        #data[i][1]=((data[i][1]-means[len(means)-1])/sigma[len(means)-1])
    return data,means,sigma

def testResult_choose_1st(w,b,testfile,means,sigma,norm,pollution,pos_pm2,res):#can use to test different pollution type
    test=[]
    test_data=[]
    test_X=open(testfile,'rb')
    k=0
    name=''
    for row in csv.reader(test_X,delimiter=','):
        k+=1
        tmp=[]
        if (k%18) in pollution:
            name=str(row[0])
            row.pop(0)
            row.pop(0)
            for j in range(len(row)) :
                if row[j]=='NR':
                    tmp.append(float(0))
                else : 
                    tmp.append(float(row[j]))
            test.append(tmp)            
        if  k==18:
            k=0
            tmp=np.array(test)# tmp is a len(pollution)*9 array
            tmp=np.transpose(tmp)
            tmp=tmp.flatten()        
            test_data.append([name,tmp])
            test=[]
    #print(len(test_data))
    #print(test_data[0][1])
    if norm==True:
        for i in range(len(test_data)) :
            for j in range(len(test_data[0][1])) :
                test_data[i][1][j]=(test_data[i][1][j]-means[j])/(sigma[j])
    file=open(res,'w')#to record prediction
    outfile=csv.writer(file)

    result=[]
    outfile.writerow(['id','value'])
    print(len(test_data))

    for i in range (len(test_data)):#predict pm2.5
        y=np.dot(test_data[i][1],w)+b 
        outfile.writerow([test_data[i][0],y])  
        
    file.close()

def read_1st_model(model) :
    csvfile=open(model,'rb')
    k=0
    w=[]
    b=0
    for row in csv.reader(csvfile,delimiter=','):
        if k==0 :
            for i in range(len(row)) :
                w.append(float(row[i]))
        if k==1 :
            b=float(row[0])
        k+=1
    w=np.array(w)
    return w,b

def SGD_1st (w,b,batch_size,init_lr,iterator,data_tuple,val_set,train_set,L_arr,L_val) :
    sigma_w=0
    sigma_b=0
    
    sum_wg=0#sum of w_grad**2 
    sum_bg=0#sum of b_grad**2
    
    w_grad=np.zeros(len(data_tuple[0][0]))
    b_grad=0
    
   
    
    #L_arr=getLoss_1st(w,b,train_set,L_arr)
    #L_val=getLoss_1st(w,b,val_set,L_val)
    
    for i in range(iterator):#run iteration
        print('i=',i)
        w_grad=np.zeros(len(w))
        b_grad=0
        n=0
        for j in range(len(data_tuple)): 
            x=0
            tmp=data_tuple[j]
            x=tmp[1]-(b+np.dot(tmp[0],w))
            #print('j=',j)
            w_grad+=(-2*x*tmp[0])
            b_grad+=(-2*x)
            
            n+=1
            
            if ((j%batch_size)==(batch_size-1)) or j==(len(data_tuple)-1) :
                w_grad=w_grad/n
                b_grad=b_grad/n
                
                n=0
                
                sum_wg+=(w_grad**2)
                sum_bg+=(b_grad**2)
                
                sigma_w=np.sqrt(sum_wg)
                sigma_b=math.sqrt(sum_bg)
                w=w-(init_lr/sigma_w)*w_grad
                b=b-(init_lr/sigma_b)*b_grad
                
                w_grad=np.zeros(len(w))
                b_grad=0
        
        #   L_arr=getLoss_1st(w,b,train_set,L_arr)
        #   L_val=getLoss_1st(w,b,val_set,L_val)
        #   if len(L_arr)>=10000 :
        #       wf.writerow(L_arr)
        #       wf.writerow(L_val)
        #       L_arr=[]
        #       L_val=[]
        #       
    return w,b,L_arr,L_val

def gra_ada_1st(w,b,init_lr,iterator,data_tuple,val_set,train_set,L_arr,L_val):
    
    sigma_w=0
    sigma_b=0
    
    sum_wg=0#sum of w_grad**2 
    sum_bg=0#sum of b_grad**2
    
    w_grad=np.zeros(len(data_tuple[0][0]))
    b_grad=0
       
    #L_arr=getLoss_1st(w,b,train_set,L_arr)
    #L_val=getLoss_1st(w,b,val_set,L_val)
    
    for i in range(iterator):
        print('i=',i)
        w_grad=np.zeros(len(w))
        b_grad=0
        for j in range(len(data_tuple)): 
            x=0
            tmp=data_tuple[j]
            x=tmp[1]-(b+np.dot(tmp[0],w))
            #print('x=',x)
            w_grad+=(-2*x*tmp[0])
            b_grad+=(-2*x)
        sum_wg+=(w_grad**2)
        sum_bg+=(b_grad**2)
    
        sigma_w=np.sqrt(sum_wg)
        sigma_b=math.sqrt(sum_bg)
    
        w=w-(init_lr/sigma_w)*w_grad
        b=b-(init_lr/sigma_b)*b_grad
        
        #   L_arr=getLoss_1st(w,b,train_set,L_arr)
        #   L_val=getLoss_1st(w,b,val_set,L_val)
        #   if len(L_arr)>=10000 :
        #       wf.writerow(L_arr)
        #       wf.writerow(L_val)
        #       L_arr=[]
        #       L_val=[]
    return w,b,L_arr,L_val
    
    
    
fileName=sys.argv[1]#'train.csv'
testfile=sys.argv[2]#'test_X.csv'
res=sys.argv[3]#'res.csv'
model='model.csv'

norm=True




pollution=[3,8,10,15,17]
pos_pm2=2

data=readTraining_choose(fileName,pollution,pos_pm2)

means=[]
sigma=[]


data_tuple,means,sigma=normalization(data)
 
val_set=[]
train_set=data_tuple


init_lr=0.04
iterator=1000
batch_size=3


L_arr=[]
L_val=[]


w,b=read_1st_model(model)



w,b,L_arr,L_val=gra_ada_1st(w,b,init_lr,iterator,data_tuple,val_set,train_set,L_arr,L_val)



testResult_choose_1st(w,b,testfile,means,sigma,norm,pollution,pos_pm2,res)

