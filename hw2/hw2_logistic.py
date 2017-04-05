#machine learning hw2
#using logistic regression
#2nd
#with normalization
#score=0.85737
import hw2_util as ut
import numpy as np
import math
import sys
import csv



def read_TA_train(X_train_csv):
    csvfile=open(X_train_csv,'rb') 
    first_read=True
    X_train=[]
    for row in csv.reader(csvfile,delimiter=','):
        if first_read :
            first_read=False
        else :
            row.pop(2)
            for i in range (7):
                row.pop(30)
            row.pop(13)
            row.pop()
            row.pop(43)
            row.pop(52)
            X_train.append(map(float,row))
    X_train=np.array(X_train)
    return X_train





train_csv=sys.argv[1]
test_csv=sys.argv[2]
X_train_csv=sys.argv[3]
Y_train_csv=sys.argv[4]
X_test_csv=sys.argv[5]
res=sys.argv[6]


model='logistic_model.csv'

first_train=False
savemodel=False





X_train=ut.read_X_train(X_train_csv)
Y_train=ut.read_Y_train(Y_train_csv)
X_test=ut.read_X_test(X_test_csv)


print(len(X_train))
print(len(Y_train))


#test
#for i in range (len(X_train)):
#    print X_train[i]

X_train=ut.normalization(X_train)
X_test=ut.normalization(X_test)


init_lr=0.001
batch_size=50
iteration=100

L_train=[]
L_val=[]

###test first order
if first_train :
    w2=np.zeros(len(X_train[0]))
    w1=np.zeros(len(X_train[0]))
    b=0
else :
    w2,w1,b=ut.read_sec_model(model)
    #w1,b=ut.read_1st_model(model)



f=open("Loss.csv",'w')
wf=csv.writer(f)


#w1,b,L_train,L_val=ut.SGD_1st(w1,b,batch_size,init_lr,iteration,X_train,Y_train,L_train,L_val,wf)

w2,w1,b,L_train,L_val=ut.SGD_2nd(w2,w1,b,batch_size,init_lr,iteration,X_train,Y_train,L_train,L_val,wf)

wf.writerow(L_train)
wf.writerow(L_val)

f.close()

if savemodel :
    #ut.write_1st_model(model,w1,b)
    ut.write_sec_model(model,w2,w1,b)
#ut.result_1st(X_test,w1,b,res)
ut.result_2nd(X_test,w2,w1,b,res)












