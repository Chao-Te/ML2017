#ML hw2
#probability generative model





import hw2_util as ut
import numpy as np
import math
import sys
import csv

def classify_2_class(X_train,Y_train):

    class1=[]#Y_train=1
    class0=[]#Y-train=0
    for i in range(len(X_train)):
        if Y_train[i]==0 :
            class0.append(X_train[i])
        else :
            class1.append(X_train[i])
    class0=np.array(class0)
    class1=np.array(class1)

    return  class0,class1

def use_Gaussian(classn):#return mean and var

    mean=np.mean(classn,axis=0)
    var=np.zeros((len(mean),len(mean)))

    for i in range(len(classn)) : 

        tmp=classn[i]-mean                    
        var+=np.dot(np.reshape(tmp,(len(tmp),1)),np.reshape(tmp,(1,len(tmp))))

    var=var/len(classn)

    return mean ,var

def get_class_probability(class0,class1):

    PC0=float(len(class0))/(len(class0)+len(class1))
    PC1=float(len(class1))/(len(class0)+len(class1))

    return PC0,PC1
    
def matrix_inverse(var):
    if np.linalg.det(var)==0 :
        print 'Singularity  Matrix'
        return np.linalg.pinv(var)
    return np.linalg.inv(var)
 

def get_result(X,mean0,mean1,varinv,PC0,PC1):
    Y=[]
    #print varinv
    print (len(X))
    for i in range (len(X)) :
    
        diff0=X[i]-mean0
        diff1=X[i]-mean1
    
        tmp0=np.dot((np.dot(np.reshape(diff0,(1,len(diff0))),varinv)),np.reshape(diff0,(len(diff0),1)))
        tmp1=np.dot((np.dot(np.reshape(diff1,(1,len(diff1))),varinv)),np.reshape(diff1,(len(diff1),1)))
       
        upper0=PC0*math.exp(-0.5*tmp0)
        upper1=PC1*math.exp(-0.5*tmp1)
        
        if upper0<upper1:
            Y.append(1)
        else :
            Y.append(0)
    Y=np.array(Y)
    return Y
    
def write_result(result,Y) :
    
    f=open(res,'w')
    wf=csv.writer(f)
    wf.writerow(['id','label'])

    for i in range(len(Y)) :
        wf.writerow([i+1,Y[i]])

    f.close()


def get_accuracy(Y,Y_train):
    n=0.0
    for i in range(len(Y)):
        if Y[i]!=Y_train[i]:
            n+=1
    return 1-n/len(Y)









train_csv=sys.argv[1]
test_csv=sys.argv[2]
X_train_csv=sys.argv[3]
Y_train_csv=sys.argv[4]
X_test_csv=sys.argv[5]
res=sys.argv[6]


#model='logistic_model.csv'


X_train=ut.read_X_train(X_train_csv)
Y_train=ut.read_Y_train(Y_train_csv)
X_test=ut.read_X_test(X_test_csv)


class0,class1=classify_2_class(X_train,Y_train)


mean0,var0=use_Gaussian(class0)
mean1,var1=use_Gaussian(class1)

PC0,PC1=get_class_probability(class0,class1)

var=PC0*var0+PC1*var1

#print(var)
varinv=matrix_inverse(var)

Y=get_result(X_train,mean0,mean1,varinv,PC0,PC1)
print(get_accuracy(Y,Y_train))

Y=get_result(X_test,mean0,mean1,varinv,PC0,PC1)

write_result(res,Y) 









