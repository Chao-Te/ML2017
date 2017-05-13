import numpy as np
import math
import matplotlib.pyplot as plt
import csv

from PIL import Image 
from scipy import misc



#####global variable#####

alpha_name=['A','B','C','D','E','F','G'\
            ,'H','I','J','K','L','M']
img_num=['00','01','02','03','04','05','06','07','08','09'\
         ,'10','11','12','13','14','15','16','17','18','19'\
         ,'20','21','22','23','24','25','26','27','28','29'\
         ,'30','31','32','33','34','35','36','37','38','39'\
         ,'40','41','42','43','44','45','46','47','48','49'\
         ,'50','51','52','53','54','55','56','57','58','59'\
         ,'60','61','62','63','64','65','66','67','68','69'\
         ,'70','71','72','73','74']                        


####utils function######

def readImg(imfile,num_subj,num_im):
    data=[]
    for i in range (num_subj):
        for j in range(num_im) :
            imName='./'+imfile+'/'+alpha_name[i]+img_num[j]+'.bmp'
            im = Image.open(imName)
            data.append(np.reshape(np.array(im),64*64))
            
    data=np.array(data).astype(float)
    
    return data


def save_9eigenface(v):
    print(len(v))
    fig = plt.figure(figsize=(8, 8))
    for i in range(len(v)):
        ax = fig.add_subplot(3, 3, i+1)
        #print(filter_imgs[it][i][0].shape)
        ax.imshow(v[i].reshape(64,64),'gray')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.xlabel('{:.3f}'.format(int(i//1)))
        plt.tight_layout()
    fig.savefig('./pca_res/9eigenFace.png')
    
def save_100img(v,figName):
    print(len(v))
    fig = plt.figure(figsize=(8, 8))
    for i in range(len(v)):
        ax = fig.add_subplot(10, 10, i+1)
        #print(filter_imgs[it][i][0].shape)
        ax.imshow(v[i].reshape(64,64),'gray')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        #plt.xlabel('{:.3f}'.format(int(i)))
        plt.tight_layout()
    fig.savefig(figName)

def cal_L2(mat1,mat2):
    m=mat1-mat2
    m=m**2
    return np.sum(m)
    
    
def cal_rmse(data_mat,recon_mat):
    if (data_mat.shape != recon_mat.shape):
        print('error shape not same!!')
    s=0.0
    for i in range(len(data_mat)):
        s+=cal_L2(data_mat[i],recon_mat[i])

    s=s/(10*10*64*64)
    print(s)
    return (100*math.sqrt(s))/256.0
    
def find_recon(eigenvectors, eigenvalues, V,av_img,k):
    U=np.transpose(np.transpose(eigenvectors)[0:k])
    sig=np.diag(eigenvalues[0:k])
    v=V[0:k]
    tmp=np.dot(np.dot(U,sig),v)
    tmp=np.transpose(tmp)
    recon_mat=tmp+av_image
    return recon_mat
    
#####main body##### 

imfile='data'
num_subj=10#read 10 face
num_im=10# number of image for each face


save_avg=False#for question1 first task, if you want to save image set save_avg=True
save_9=False
do_problem2_1=False
do_problem2_2=False
do_problem3=True

data_mat=readImg(imfile,num_subj,num_im)

#####plot average####
av_image=np.mean(data_mat, axis=0)

#print(av_image)
if save_avg:
    misc.imsave('./pca_res/average.png', av_image.reshape(64,64))

#####find svd#####

img_cen=np.transpose(data_mat-av_image)
print(img_cen.shape)
eigenvectors, eigenvalues, V = np.linalg.svd(img_cen, full_matrices=False)



'''
problem1-2
'''
eig_vT=np.transpose(eigenvectors)
if save_9:
    save_9eigenface(eig_vT[0:9])

'''
problem2
'''

####original#####
if do_problem2_1:
    save_100img(data_mat,"./pca_res/original.png")

#####recon#####
if do_problem2_2:
    U=np.transpose(np.transpose(eigenvectors)[0:5])
    sig=np.diag(eigenvalues[0:5])
    v=V[0:5]
    tmp=np.dot(np.dot(U,sig),v)
    tmp=np.transpose(tmp)
    recon_mat=tmp+av_image
    save_100img(recon_mat,"./pca_res/recon.png")

print('finish problem 2')

'''
problem3
'''
k_arr=['k']
error_arr=['error']
if do_problem3:
    for k in range(100):
        tmp=find_recon(eigenvectors, eigenvalues, V,av_image,k)
        error_arr.append(cal_rmse(data_mat,tmp))
        k_arr.append(k)
    f=open("./pca_res/res.csv",'w')
    wf=csv.writer(f)
    wf.writerow(k_arr)
    wf.writerow(error_arr)
    f.close()
print('finish problem3')
