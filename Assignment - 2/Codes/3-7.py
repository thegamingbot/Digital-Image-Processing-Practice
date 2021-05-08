######  Q3 and Q4


import numpy as np
import cv2
import array

def fft(signal):
    F=[]
    M=len(signal)
    for u in range(0,M):
        temp=0
        for x in range(0,M):
            temp+=signal[x]*(np.exp((-2j)*np.pi*u*x/M))
        F.append(temp)
    return(F)

def dft(f):
    M=len(f)
    odd=[]
    even=[]
    G=[]
    H=[]
    if M==1:
        return(fft(f))
    else:
       for i in range(M):
        if i%2==0:
            even.append(f[i])
        else:
            odd.append(f[i])
       G=(fft(even)).copy()
       H=(fft(odd)).copy()
       F1=[]
       F2=[]
       for u in range(0,int(M/2)):
           temp1=G[u]+np.multiply(np.exp(-2j*np.pi*u/M),H[u])
           F1.append(temp1)
       for u in range(int(M/2),M):
           temp2=G[u-int(M/2)]-np.multiply(np.exp(-2j*np.pi*(u-int(M/2))/M),H[u-int(M/2)])
           F1.append(temp2)
#        print("fft function output:",F1)    
       return(F1)

a=[1,2,3,4,5,6,7,8,9,10]
print("inbuilt:",np.fft.fft(a))
print("calling fft:",dft(a))




###########  Q4 (dft used for an image)


import cv2
import numpy as np
import pandas as pd
img = cv2.imread("lenares.png",0)
def find_dft(img):
    row=[]
    col=[]
    result = np.array(img)
    for p in result:
        row.append(dft(p))
    result2 = np.array(row)
    
    for p in result2.T:
        col.append(dft(p))
    result3 = np.array(col)
    print("inbuilt dft2 : ",np.fft.fft2(result))
    return result3.T
print("user defined:  ",find_dft(img))



############  Q5  (magnitude and phase for dog and lena)


import math
import cmath 
dog = cv2.imread("dogres.jpg",0)
def magnitude(matrix):
    mag=[]
    mag2=[]
    li1=[]
    li2=[]
    for i in range(0,matrix.shape[0]):
        li1=[]
        li2=[]
        for j in range(0,matrix.shape[1]):
            li2.append(abs(matrix[i][j]))
            li1.append(((matrix[i][j].real)*(matrix[i][j].real) + (matrix[i][j].imag)*(matrix[i][j].imag))**0.5)
        mag.append(li1)
        mag2.append(li2)
    mag=np.array(mag)
    mag2=np.array(mag2)
    print(mag)
#     print("inbuilt mag : ",mag2)
    return mag
def phase(matrix):
    phase=[]
    li=[]
    for i in range(0,matrix.shape[0]):
        li=[]
        for j in range(0,matrix.shape[1]):
#             li.append(math.atan(float(matrix[i][j].imag)/float(matrix[i][j].real)))
            li.append((cmath.phase(matrix[i][j])))
        phase.append(li)
    phase=np.array(phase)
    print("phase is : ",phase)
    return phase
fft_lena = find_dft(img)
fft_dog = find_dft(dog)

lena_mag_dog_phase = magnitude(fft_lena)*(np.exp(1j*phase(fft_dog)))
print("combined matrix with magnitude of lena and phase of dog:\n\n", lena_mag_dog_phase)
dog_mag_lena_phase = magnitude(fft_dog)*(np.exp(1j*phase(fft_lena)))
print("combined matrix with magnitude of dog and phase of lena: \n\n", dog_mag_lena_phase)



###############  Q6 and Q7

# IDFT
def idft(signal):
    F=[]
    M=len(signal)
    for u in range(0,M):
        temp=0
        for x in range(0,M):
            temp+=(signal[x]*(np.exp((2j)*np.pi*u*x/M)))
        F.append(temp)
    return(F)
img =  [(75+0j), (-28.500000000000025+33.774990747593094j), (-19.49999999999997-32.042939940024254j), (33-0j), (-19.49999999999997+32.042939940024254j), (-28.50000000000002-33.774990747593094j)]
def find_idft(img):
    idft_result = idft(img)
    idft_real=[]
    for item in idft_result:
        idft_real.append(item.real)
#     print("user defined idft: ",idft_real)
#     print()
    print("inbuilt",np.fft.ifft(img))
    return idft_real
# img =  [(75+0j), (-28.500000000000025+33.774990747593094j), (-19.49999999999997-32.042939940024254j), (33-0j), (-19.49999999999997+32.042939940024254j), (-28.50000000000002-33.774990747593094j)]
print("user def: ",find_idft(img))



from matplotlib import pyplot as plt
def original(img):
    row=[]
    col=[]
    m,n=img.shape
    result = np.array(img)
    for p in result:
        row.append(find_idft(p))
    result2 = np.array(row)
    for p in result2.T:
        col.append(find_idft(p))
    result3 = np.array(col)
    return (result3.T/(m*n))

# arr = np.array(arr)
# print(original(arr))
orig_img_dog = original(lena_mag_dog_phase)
orig_img_lena = original(dog_mag_lena_phase)
print("idft image:  ",(orig_img_lena))
plt.imshow(orig_img_dog,cmap='gray')
plt.show()
plt.imshow(orig_img_lena,cmap='gray')
plt.show()


