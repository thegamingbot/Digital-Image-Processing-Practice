import cv2
from matplotlib import pyplot as plt
img = cv2.imread("lena.png")
gray = cv2.imread("lena.png",0)
#Salt and Pepper
from skimage.util import random_noise
import numpy as np
def sp(n):
    noise_img=random_noise(gray,mode='s&p',amount=n)
    noise_img=np.array(255*noise_img,dtype='uint8')
    cv2.imwrite('sp_42049.png',noise_img)
#     cv2.imshow('blur',noise_img)
#     cv2.waitKey(1000)
#     cv2.destroyAllWindows()
    return noise_img
#     cv2.imshow('blur',noise_img)

a1=sp(0.1)
a2=sp(0.2)
a3=sp(0.3)
a4=sp(0.4)
a5=sp(0.5)
a6=sp(0.6)
a7=sp(0.7)
a8=sp(0.8)
a9=sp(0.9)
a10=sp(1)
# img=[]
img1=[a1,a2]
img2=[a3,a4]
img3=[a5,a6]
img4=[a7,a8]
img5=[a9,a10]
# plt.imshow(a)
# plt.show()
def show_img(img):
    cv2.imshow('noise',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# show_img(a4)
# print(sp(0.1))
for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(img1[i],'gray')
for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(img2[i],'gray')
for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(img3[i],'gray')    
for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(img4[i],'gray')
for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(img5[i],'gray')


#3x3 filter            


print(a1.shape)
f1=np.zeros([3,3])
f1.fill(1/9)
print("f1 is :\n ",f1)
f1_new=np.zeros([a1.shape[0]+f1.shape[0]-1,a1.shape[1]+f1.shape[1]-1])#new matrix of size m+a-1 and n+b-1
print(f1_new.shape)
n1=np.insert(a1, 0, 0, axis = 1)
n1=np.insert(n1, 0, 0, axis = 0)
n1=np.insert(n1, 513, 0, axis = 1)
# n1=np.insert(n1,1,0,axis=1)
n1=np.insert(n1,513,0,axis=0)
# n1=np.insert(n1,1,0,axis=0)
print("n1 is :\n",n1) # n1 contains the a1 matrix with extra 2 rows and cols of 0's
print(n1.shape)
cv2.imshow("n1 img",n1)
cv2.waitKey(0)
cv2.destroyAllWindows()

#correlation

print(f1.shape)
b=int((f1.shape[0]-1)/2)
a=int((f1.shape[1]-1)/2)
print(a1)
for x in range(0,a1.shape[0]+f1.shape[0]-1):
    for y in range(0,a1.shape[1]+f1.shape[1]-1):
#         print("inside y")
#         f1_new[x,y]=0
        for j in range(-1*b,b):
            for i in range(-1*a,a):
                
                f1_new[x,y]+=np.multiply(n1[x+i,y+j],f1[i,j])
#                 print(f1_new[x,y])
print(f1_new)
# cv2.imshow('pls work',f1_new)
# print('hi')
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.imshow(f1_new,'gray')
plt.show()

#Using inbuilt function
blur = cv2.blur(n1,(3,3))
plt.imshow(blur,'gray'),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()


# 5x5 filter


print(a1.shape)
f2=np.zeros([5,5])
f2.fill(1/25)
print("f2 is :\n ",f2)
f2_new=np.zeros([a1.shape[0]+f2.shape[0]-1,a1.shape[1]+f2.shape[1]-1])#new matrix of size m+a-1 and n+b-1
print(f2_new.shape)
n1=np.insert(a1, 0, 0, axis = 1)
n1=np.insert(n1,0,0,axis=0)
n1=np.insert(n1, 1, 0, axis = 1)
n1=np.insert(n1,1,0,axis=0)
n1=np.insert(n1,514,0,axis=1)
n1=np.insert(n1,515,0,axis=1)
n1=np.insert(n1, 514, 0, axis = 0)
n1=np.insert(n1,515,0,axis=0)

print("n1 is :\n",n1) # n1 contains the a1 matrix with extra 2 rows and cols of 0's
print(n1.shape)
cv2.imshow("n1 img",n1)
cv2.waitKey(0)
cv2.destroyAllWindows()

b=int((f2.shape[0]-1)/2)
a=int((f2.shape[1]-1)/2)
print(a1)
for x in range(0,a1.shape[0]+f2.shape[0]-2):
    for y in range(0,a1.shape[1]+f2.shape[1]-2):
#         print("inside y")
#         f1_new[x,y]=0
        for j in range(-1*b,b):
            for i in range(-1*a,a):
                
                f2_new[x,y]+=np.multiply(n1[x+i,y+j],f2[i,j])
#                 print(f1_new[x,y])
print(f2_new)
# cv2.imshow('pls work',f1_new)
# print('hi')
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.imshow(f2_new,'gray')
plt.show()

#Using inbuilt function
blur = cv2.blur(n1,(5,5))
plt.imshow(blur,'gray'),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

# 7x7 filter


print(a1.shape)
f2=np.zeros([7,7])
f2.fill(1/49)
print("f2 is :\n ",f2)
f2_new=np.zeros([a1.shape[0]+f2.shape[0]-1,a1.shape[1]+f2.shape[1]-1])#new matrix of size m+a-1 and n+b-1
print(f2_new.shape)
n1=np.insert(a1, 0, 0, axis = 1)
n1=np.insert(n1,0,0,axis=0)
n1=np.insert(n1, 1, 0, axis = 1)
n1=np.insert(n1,1,0,axis=0)
n1=np.insert(n1,0,0,axis=1)
n1=np.insert(n1,513,0,axis=0)
n1=np.insert(n1,514,0,axis=1)
n1=np.insert(n1,515,0,axis=1)
n1=np.insert(n1, 514, 0, axis = 0)
n1=np.insert(n1,515,0,axis=0)
n1=np.insert(n1,516,0,axis=0)
n1=np.insert(n1,517,0,axis=1)
print("n1 is :\n",n1) # n1 contains the a1 matrix with extra 2 rows and cols of 0's
print(n1.shape)
cv2.imshow("n1 img",n1)
cv2.waitKey(0)
cv2.destroyAllWindows()

f2_new=np.zeros([a1.shape[0]+f2.shape[0]-1,a1.shape[1]+f2.shape[1]-1])#new matrix of size m+a-1 and n+b-1
b=int((f2.shape[0]-1)/2)
a=int((f2.shape[1]-1)/2)
print(a1)
for x in range(0,a1.shape[0]+f2.shape[0]-3):
    for y in range(0,a1.shape[1]+f2.shape[1]-3):
        for j in range(-1*b,b):
            for i in range(-1*a,a):
                
                f2_new[x,y]+=np.multiply(n1[x+i,y+j],f2[i,j])
#                 print(f1_new[x,y])
print(f2_new)
plt.imshow(f2_new,'gray')
plt.show()


#Using inbuilt function
blur = cv2.blur(n1,(7,7))
plt.imshow(blur,'gray'),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()


#median filter 3x3


print(a1.shape)
f1=np.zeros([3,3])
f1.fill(1/9)
print("f1 is :\n ",f1)
f1_new=np.zeros([a1.shape[0]+f1.shape[0]-1,a1.shape[1]+f1.shape[1]-1])#new matrix of size m+a-1 and n+b-1
print(f1_new.shape)
n1=np.insert(a1, 0, 0, axis = 1)
n1=np.insert(n1, 0, 0, axis = 0)
n1=np.insert(n1, 513, 0, axis = 1)
# n1=np.insert(n1,1,0,axis=1)
n1=np.insert(n1,513,0,axis=0)
# n1=np.insert(n1,1,0,axis=0)
print("n1 is :\n",n1) # n1 contains the a1 matrix with extra 2 rows and cols of 0's
print(n1.shape)
cv2.imshow("n1 img",n1)
cv2.waitKey(0)
cv2.destroyAllWindows()


f2_new=f1_new
# f2_new=a1
plt.imshow(a1,'gray')
plt.show()
li=[]
x=a1
a1=x
for i in range(511):
    for j in range(511):
        if(i!=0 and j!=0 and i!=511 and j!=511):
            li=[a1[i-1,j-1],a1[i-1,j],a1[i-1,j+1],a1[i,j-1],a1[i,j],a1[i,j+1],a1[i+1,j-1],a1[i+1,j],a1[i+1,j+1]]
            li=(sorted(li))
#             print("edian is :",li[int(len(li)/2)])
            f2_new[i,j]=li[int(len(li)/2)]
        elif(i==0 and j==0):
            li=[a1[i,j],a1[i,j+1],a1[i+1,j],a1[i+1,j+1]]
            li=sorted(li)
            f2_new[i,j]=li[int(len(li)/2)]
            
        elif(i==511 and j==511):
            li=[a1[i,j-1],a1[i,j],a1[i-1,j-1],a1[i-1,j]]
            li=sorted(li)
            f2_new[i,j]=li[int(len(li)/2)]
        elif(i==511 and j==0):
            li=[a1[i-1,j],a1[i-1,j+1],a1[i,j],a1[i,j+1]]
            li=sorted(li)
            f2_new[i,j]=li[int(len(li)/2)]
        elif(i==0 and j==511):
            li=[a1[i,j],a1[i,j-1],a1[i+1,j],a1[i+1,j-1]]
            li=sorted(li)
            f2_new[i,j]=li[int(len(li)/2)]
        elif(i==0 or i==511):
            li=[a1[i,j-1],a1[i,j],a1[i,j+1],a1[i+1,j-1],a1[i+1,j],a1[i+1,j+1]]
            li=sorted(li)
            f2_new[i,j]=li[int(len(li)/2)]
        elif(j==0 or j==511):
            li=[a1[i-1,j],a1[i-1,j+1],a1[i,j],a1[i,j+1],a1[i+1,j],a1[i+1,j+1]]
            li=sorted(li)
            f2_new[i,j]=li[int(len(li)/2)]
plt.imshow(f2_new,'gray')
plt.show()



# 5x5 median filter

#Median filter
print(a1.shape)
f2=np.zeros([5,5])
f2.fill(1/25)
print("f2 is :\n ",f2)
f2_new=np.zeros([a1.shape[0]+f2.shape[0]-1,a1.shape[1]+f2.shape[1]-1])#new matrix of size m+a-1 and n+b-1
print(f2_new.shape)
n1=np.insert(a1, 0, 0, axis = 1)
n1=np.insert(n1,0,0,axis=0)
n1=np.insert(n1, 1, 0, axis = 1)
n1=np.insert(n1,1,0,axis=0)
n1=np.insert(n1,514,0,axis=1)
n1=np.insert(n1,515,0,axis=1)
n1=np.insert(n1, 514, 0, axis = 0)
n1=np.insert(n1,515,0,axis=0)

print("n1 is :\n",n1) # n1 contains the a1 matrix with extra 2 rows and cols of 0's
print(n1.shape)
cv2.imshow("n1 img",n1)
cv2.waitKey(0)
cv2.destroyAllWindows()


f2_new=f2_new
# f2_new=a1
plt.imshow(a1,'gray')
plt.show()
li=[]
x=a1
a1=x
for i in range(511):
    for j in range(511):
        if(i!=0 and j!=0 and i!=511 and j!=511):
            li=[a1[i-1,j-1],a1[i-1,j],a1[i-1,j+1],a1[i,j-1],a1[i,j],a1[i,j+1],a1[i+1,j-1],a1[i+1,j],a1[i+1,j+1]]
            li=(sorted(li))
#             print("edian is :",li[int(len(li)/2)])
            f2_new[i,j]=li[int(len(li)/2)]
        elif(i==0 and j==0):
            li=[a1[i,j],a1[i,j+1],a1[i+1,j],a1[i+1,j+1]]
            li=sorted(li)
            f2_new[i,j]=li[int(len(li)/2)]
            
        elif(i==511 and j==511):
            li=[a1[i,j-1],a1[i,j],a1[i-1,j-1],a1[i-1,j]]
            li=sorted(li)
            f2_new[i,j]=li[int(len(li)/2)]
        elif(i==511 and j==0):
            li=[a1[i-1,j],a1[i-1,j+1],a1[i,j],a1[i,j+1]]
            li=sorted(li)
            f2_new[i,j]=li[int(len(li)/2)]
        elif(i==0 and j==511):
            li=[a1[i,j],a1[i,j-1],a1[i+1,j],a1[i+1,j-1]]
            li=sorted(li)
            f2_new[i,j]=li[int(len(li)/2)]
        elif(i==0 or i==511):
            li=[a1[i,j-1],a1[i,j],a1[i,j+1],a1[i+1,j-1],a1[i+1,j],a1[i+1,j+1]]
            li=sorted(li)
            f2_new[i,j]=li[int(len(li)/2)]
        elif(j==0 or j==511):
            li=[a1[i-1,j],a1[i-1,j+1],a1[i,j],a1[i,j+1],a1[i+1,j],a1[i+1,j+1]]
            li=sorted(li)
            f2_new[i,j]=li[int(len(li)/2)]
plt.imshow(f2_new,'gray')
plt.show()


#7x7 median filter


print(a1.shape)
f2=np.zeros([7,7])
f2.fill(1/49)
print("f2 is :\n ",f2)
f2_new=np.zeros([a1.shape[0]+f2.shape[0]-1,a1.shape[1]+f2.shape[1]-1])#new matrix of size m+a-1 and n+b-1
print(f2_new.shape)
n1=np.insert(a1, 0, 0, axis = 1)
n1=np.insert(n1,0,0,axis=0)
n1=np.insert(n1, 1, 0, axis = 1)
n1=np.insert(n1,1,0,axis=0)
n1=np.insert(n1,0,0,axis=1)
n1=np.insert(n1,513,0,axis=0)
n1=np.insert(n1,514,0,axis=1)
n1=np.insert(n1,515,0,axis=1)
n1=np.insert(n1, 514, 0, axis = 0)
n1=np.insert(n1,515,0,axis=0)
n1=np.insert(n1,516,0,axis=0)
n1=np.insert(n1,517,0,axis=1)
print("n1 is :\n",n1) # n1 contains the a1 matrix with extra 2 rows and cols of 0's
print(n1.shape)
cv2.imshow("n1 img",n1)
cv2.waitKey(0)
cv2.destroyAllWindows()

f2_new=f2_new
# f2_new=a1
plt.imshow(a1,'gray')
plt.show()
li=[]
x=a1
a1=x
for i in range(511):
    for j in range(511):
        if(i!=0 and j!=0 and i!=511 and j!=511):
            li=[a1[i-1,j-1],a1[i-1,j],a1[i-1,j+1],a1[i,j-1],a1[i,j],a1[i,j+1],a1[i+1,j-1],a1[i+1,j],a1[i+1,j+1]]
            li=(sorted(li))
#             print("edian is :",li[int(len(li)/2)])
            f2_new[i,j]=li[int(len(li)/2)]
        elif(i==0 and j==0):
            li=[a1[i,j],a1[i,j+1],a1[i+1,j],a1[i+1,j+1]]
            li=sorted(li)
            f2_new[i,j]=li[int(len(li)/2)]
            
        elif(i==511 and j==511):
            li=[a1[i,j-1],a1[i,j],a1[i-1,j-1],a1[i-1,j]]
            li=sorted(li)
            f2_new[i,j]=li[int(len(li)/2)]
        elif(i==511 and j==0):
            li=[a1[i-1,j],a1[i-1,j+1],a1[i,j],a1[i,j+1]]
            li=sorted(li)
            f2_new[i,j]=li[int(len(li)/2)]
        elif(i==0 and j==511):
            li=[a1[i,j],a1[i,j-1],a1[i+1,j],a1[i+1,j-1]]
            li=sorted(li)
            f2_new[i,j]=li[int(len(li)/2)]
        elif(i==0 or i==511):
            li=[a1[i,j-1],a1[i,j],a1[i,j+1],a1[i+1,j-1],a1[i+1,j],a1[i+1,j+1]]
            li=sorted(li)
            f2_new[i,j]=li[int(len(li)/2)]
        elif(j==0 or j==511):
            li=[a1[i-1,j],a1[i-1,j+1],a1[i,j],a1[i,j+1],a1[i+1,j],a1[i+1,j+1]]
            li=sorted(li)
            f2_new[i,j]=li[int(len(li)/2)]
plt.imshow(f2_new,'gray')
plt.show()

