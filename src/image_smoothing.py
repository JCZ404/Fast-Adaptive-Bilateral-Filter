import time
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
from adaptiveBF import fastABF
from adaptiveBF import abf_bruteforce
from adaptiveBF import logClassifier

if __name__ == "__main__":
    # demo1: sharpening
    # 加载图片
    sharpening_file = "./images/peppers_degraded.tif"
    f = Image.open(sharpening_file)
    plt.figure("original image")
    plt.imshow(f,cmap=plt.cm.gray)
    f = np.array(f)

    
    rho = 5
    N = 5

    #! 统计计算zeta和sigma_r的用时
    t1 = time.time()
    [zeta,sigma_r] = logClassifier(f,rho,[23,33])
    t2 = time.time()
    print("Time of zeta and sigma_r: ",t2-t1)

    #! 统计暴力法的用时
    t1 = time.time()
    g = abf_bruteforce(f,rho,sigma_r,f+zeta)
    t2 = time.time()
    print("Total Time of brute force: ",t2-t1)
    plt.figure("brute force")
    plt.imshow(g,cmap=plt.cm.gray)

    #! 统计整个fastABF的用时
    t1 = time.time()
    g_hat = fastABF(f,rho,sigma_r,f+zeta,N)
    t2 = time.time()
    print("Total time of fastABF: ",t2-t1)
    g_hat[g_hat>255] = 255 
    g_hat[g_hat<0] = 0
    plt.figure("fastABF")
    plt.imshow(g_hat,cmap=plt.cm.gray)
    plt.show()