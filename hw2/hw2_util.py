#util file for machine learning hw2

import csv
import math
import numpy as np



def print_data(data) :
    for i in range (len(data)) :
        print(data[i])

def sigmoid(z):
   # print('z=',z)
    if z<0 :
        return (1-(1/(1+math.exp(z))))
    return (1/(1+math.exp(-z)))

def ln(z):
    #print('z=',z)
    return math.log(z)

def sigfunc_1st(w,b,x):
    return sigmoid(np.dot(w,x)+b)

def sigfunc_2nd(w2,w1,b,x):
    return sigmoid(np.dot(w2,x**2)+np.dot(w1,x)+b)    

def normalization(data):#data is an np.array
    mean=np.mean(data,axis=0)
    std=np.std(data,axis=0)
    for i in range (len(data)):
        for j in range (len(data[0])):
            if std[j]!=0 : 
                data[i][j]=(data[i][j]-mean[j])/std[j]
    return data

def normalization_continuous(data):
    mean=np.mean(data,axis=0)
    std=np.std(data,axis=0)
    for i in range(len(data)):
        for j in range(6):
            if (j != 2) and (std[j] != 0):
                data[i][j]=(data[i][j]-mean[j])/std[j]
    return data
 
def normalization_zero_to_minus(data):
    mean=np.mean(data,axis=0)
    std=np.std(data,axis=0)
    for i in range(len(data)):
        for j in range(len(data[0])):
            if (j in [0,1,3,4,5]) and (std[j] != 0):
                data[i][j]=(data[i][j]-mean[j])/std[j]
            elif data[i][j]==0:
                data[i][j]=-1
    return data 
    
def getLoss_1st(w,b,X,Y,Loss) : 
    L=0
    for i in range(len(X)):
        tmp=sigfunc_1st(w,b,X[i])
       # if tmp >=1 :
       #     print('error1')
       # elif tmp<=0:
       #     print('error 0')
        if Y[i]==1 :
            if tmp==0 :
                L-=-1000000
            else:
                L-=(Y[i]*ln(tmp))
        else :
            if tmp==1 :
                L-=-1000000
            else :
                L-=((1-Y[i])*ln(1-tmp))
    print('L=',L)
    Loss.append(L)  

def getLoss_2nd(w2,w1,b,X,Y,Loss) : 
    L=0
    for i in range(len(X)):
        tmp=sigfunc_2nd(w2,w1,b,X[i])
        if Y[i]==1 :
            if tmp==0 :
                L-=-1000000
            else :
                L-=(Y[i]*ln(tmp))
        else :
            if tmp==1 :
                L-=-1000000
            else :    
                L-=((1-Y[i])*ln(1-tmp))
    print(L)
    Loss.append(L)
    
    
def read_X_train(X_train_csv):
    csvfile=open(X_train_csv,'rb') 
    first_read=True
    X_train=[]
    for row in csv.reader(csvfile,delimiter=','):
        if first_read :
            first_read=False
        else :
            X_train.append(map(float,row))
    X_train=np.array(X_train)
    return X_train

def read_X_test(X_test_csv):
    csvfile=open(X_test_csv,'rb')
    first_read=True
    X_test=[] 
    for row in csv.reader(csvfile,delimiter=','):
        if first_read :
            first_read=False
        else :
            X_test.append(map(float,row))
    X_test=np.array(X_test)
    return X_test

def read_Y_train(Y_train_csv):
    csvfile=open(Y_train_csv)
    Y_train=[]
    for row in csv.reader(csvfile,delimiter=','):
        Y_train.append(map(float,row))
    Y_train=np.array(Y_train)
    return Y_train

  
def read_X_limit(X_csv,age,fnlwgt,sex,cg,cl,hour_per,work_class,edu,marital,occupation,relationship,race,nation) :
    csvfile=open(X_csv,'rb') 
    first_read=True
    X_train=[]
    for row in csv.reader(csvfile,delimiter=','):
        if first_read :
            first_read=False
        else :
            tmp=[]
            #print row
            if age :
                tmp.append(row[0])
            if fnlwgt :
                tmp.append(row[1])
            if sex :
                tmp.append(row[2])
            if cg :
                tmp.append(row[3])
            if cl :
                tmp.append(row[4])
            if hour_per :
                tmp.append(row[5])
            if work_class :
                tmp.extend(row[6:15])
            if edu :
                tmp.extend(row[15:31])
            if marital :
                tmp.extend(row[31:38])
            if occupation :
                tmp.extend(row[38:53])
            if relationship :
                tmp.extend(row[53:59])
            if race :
                tmp.extend(row[59:64])
            if nation:
                tmp.extend(row[64:106])
            #print tmp
            #print tmp
            X_train.append(np.array(map(float,tmp)))
    X_train=np.array(X_train)
    #print X_train
    return X_train

def read_train(train_csv):
    pass
def read_test(test_csv):
    pass

def SGD_2nd (w2,w1,b,batch_size,init_lr,iterator,X_train,Y_train,L_arr,L_val,wf) :

    sigma_w2=np.zeros(len(X_train[0]))
    sigma_w1=np.zeros(len(X_train[0]))
    sigma_b=0
    
    sum_w2g=np.zeros(len(X_train[0]))#sum of w2_grad**2
    sum_w1g=np.zeros(len(X_train[0]))#sum of w1_grad**2 
    sum_bg=0#sum of b_grad**2
    
    w2_grad=np.zeros(len(X_train[0]))
    w1_grad=np.zeros(len(X_train[0]))
    b_grad=0
    
    
    #   getLoss_2nd(w2,w1,b,X_train,Y_train,L_arr)
    #   getLoss_2nd(w2,w1,b,X_train,Y_train,L_val)
    L_arr.append(get_2nd_accuracy(X_train,Y_train,w2,w1,b))    
    
    for i in range(iterator):
        print('i=',i)
        
        w2_grad=np.zeros(len(X_train[0]))
        w1_grad=np.zeros(len(X_train[0]))
        b_grad=0
        n=0
        
        for j in range(len(X_train)): 
        
            x=0
            
            x2=(X_train[j]**2)#x^2
            x=Y_train[j]-sigmoid(np.dot(w2,x2)+np.dot(w1,X_train[j])+b)
            
            #print('j=',j)
            
            w2_grad+=(-2*x*x2)
            w1_grad+=(-2*x*X_train[j])
            b_grad+=(-2*x)
            n+=1
            
            if ((j%batch_size)==(batch_size-1)) or j==(len(X_train)-1) :
            
                w2_grad=w2_grad/n
                w1_grad=w1_grad/n
                b_grad=b_grad/n
                
                n=0
                
                sum_w2g+=(w2_grad)**2
                sum_w1g+=(w1_grad)**2
                sum_bg+=(b_grad)**2
                
                sigma_w2=np.sqrt(sum_w2g)
                sigma_w1=np.sqrt(sum_w1g)
                sigma_b=math.sqrt(sum_bg)
                
                for m in range(len(w2)):
                    if sigma_w2[m]!=0 :
                        w2[m]=w2[m]-(init_lr/sigma_w2[m])*w2_grad[m]
                    if sigma_w1[m]!=0 :
                        w1[m]=w1[m]-(init_lr/sigma_w1[m])*w1_grad[m]
                b=b-(init_lr/sigma_b)*b_grad
                
                w2_grad=np.zeros(len(w2))
                w1_grad=np.zeros(len(w1))
                b_grad=0
        
        #   getLoss_2nd(w2,w1,b,X_train,Y_train,L_arr)
        #   getLoss_2nd(w2,w1,b,X_train,Y_train,L_val)
        if i%10==0 or i==iterator-1 :
            L_arr.append(get_2nd_accuracy(X_train,Y_train,w2,w1,b))
        
        if len(L_arr)>=10000 :
            wf.writerow(L_arr)
            wf.writerow(L_val)
            L_arr=[]
            L_val=[]
    return w2,w1,b,L_arr,L_val

def SGD_1st (w,b,batch_size,init_lr,iterator,X_train,Y_train,L_arr,L_val,wf) :

    sigma_w=np.zeros(len(X_train[0]))
    sigma_b=0
    
    sum_wg=np.zeros(len(X_train[0]))#sum of w_grad**2 
    sum_bg=0#sum of b_grad**2
    
    w_grad=np.zeros(len(X_train[0]))
    b_grad=0
    
   
    
    L_arr.append(get_1st_accuracy(X_train,Y_train,w,b))
    
    for i in range(iterator):
        print('i=',i)
        w_grad=np.zeros(len(w))
        b_grad=0
        n=0
        for j in range(len(X_train)):
            #print('j=',j)
            #print('w=',w)
            #print('X_train[j]=',X_train[j]) 
            #print('dot=',np.dot(X_train[j],w))
            x=Y_train[j]-sigmoid(b+np.dot(X_train[j],w))
            #print(x)
            #print('j=',j)
            w_grad+=(-2*x*X_train[j])
            b_grad+=(-2*x)
            
            n+=1
            
            if ((j%batch_size)==(batch_size-1)) or j==(len(X_train)-1) :
                w_grad=w_grad/n
                b_grad=b_grad/n
                
                n=0
                
                sum_wg+=(w_grad**2)
                sum_bg+=(b_grad**2)
                
                sigma_w=np.sqrt(sum_wg)
                sigma_b=math.sqrt(sum_bg)
                for m in range(len(w_grad)):
                    if sigma_w[m]!=0 :
                        w[m]=w[m]-(init_lr/sigma_w[m])*w_grad[m]
                b=b-(init_lr/sigma_b)*b_grad
                
                w_grad=np.zeros(len(w))
                b_grad=0
        
        #   getLoss_1st(w,b,X_train,Y_train,L_arr)
        #   getLoss_1st(w,b,[],[],L_val)
        if i%10==0 or i==iterator -1:
            L_arr.append(get_1st_accuracy(X_train,Y_train,w,b))
        if len(L_arr)>=10000 :
            wf.writerow(L_arr)
            wf.writerow(L_val)
            L_arr=[]
            L_val=[]
            
    return w,b,L_arr,L_val

    
def result_1st(X_test,w,b,res):
    f=open(res,'w')
    wf=csv.writer(f)
    wf.writerow(['id','label'])
    for i in range(len(X_test)) :
        r=sigfunc_1st(w,b,X_test[i])#np.dot(X_test[i],w)+b
        if r<0.5 :
            r=0
        else :
            r=1
        wf.writerow( [i+1 ,r] )
    f.close()

def result_2nd(X_test,w2,w1,b,res):
    f=open(res,'w')
    wf=csv.writer(f)
    wf.writerow(['id','label'])
    for i in range (len(X_test)):
        r=sigfunc_2nd(w2,w1,b,X_test[i])
        if r<0.5 :
            r=0
        else :
            r=1
        wf.writerow([i+1,r])
    f.close()
    
def read_sec_model(model) :
    csvfile=open(model,'rb')
    k=0
    w2=[]
    w1=[]
    b=0
    for row in csv.reader(csvfile,delimiter=','):
        if k==0 :
            for i in range(len(row)) :
                w2.append(float(row[i]))
        if k==1 :
            for i in range(len(row)) :
                w1.append(float(row[i]))
        if k==2 :
            b=float(row[0])
        k+=1
    w2=np.array(w2)
    w1=np.array(w1)
    return w2,w1,b

def write_sec_model(model,w2,w1,b) :
    f=open(model,'w')
    wf=csv.writer(f)
    wf.writerow(w2)
    wf.writerow(w1)
    wf.writerow(b)
    f.close()
    
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

def write_1st_model(model,w,b) :
    f=open(model,'w')
    wf=csv.writer(f)
    wf.writerow(w)
    wf.writerow(b)
    f.close()

def get_1st_accuracy(X,Y,w,b) :

    Y_tmp=[]
    n=float(0)
    for i in range (len(X)):
        r=sigfunc_1st(w,b,X[i])
        if r<0.5 and Y[i]==1:
            n+=1
        elif r>=0.5 and Y[i]==0 :
            n+=1
    
    accuracy=100-n*100/len(Y)
    print('accuracy:',accuracy)
    return accuracy
    
def get_2nd_accuracy(X,Y,w2,w1,b) :

    n=float(0)
    for i in range (len(Y)):
        r=sigfunc_2nd(w2,w1,b,X[i])
        if r<0.5 and Y[i]==1:
            n+=1
        elif r>=0.5 and Y[i]==0 :
            n+=1
    accuracy=100-n*100/len(Y)
    print('accuracy:',accuracy)

    return accuracy

