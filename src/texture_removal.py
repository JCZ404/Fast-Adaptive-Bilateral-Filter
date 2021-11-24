import numpy as np
import matplotlib.pyplot as plt
import cv2
from adaptiveBF import fastABF
from adaptiveBF import logClassifier
from adaptiveBF import get_gaussian_kernel,get_log_kernel,get_square_kernel,imfilter
from adaptiveBF import MinMaxFilter
from adaptiveBF import linearMap


# mrtv用来衡量texture removal的性能
# np.gradient

def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))

def mrtv(I,k):
  
    if len(I.shape) == 3:
        I = cv2.cvtColor(I,cv2.COLOR_RGB2GRAY)

    I = I.astype(np.float32)
    
    smooth_kernel = get_square_kernel(5,1./25)
    I = imfilter(I,smooth_kernel,"symmetric")
   
    [IMin, IMax] = MinMaxFilter(I,k)
    
    Delta = IMax - IMin
    gradI = np.gradient(I)
    gradI = norm(gradI)

    [_, num] = MinMaxFilter(gradI,k)
    sum_kernel = get_square_kernel(k,1.)
    den = imfilter(gradI,sum_kernel,"symmetric")

    M = Delta*num/(den + 1e-9)
    M[den==0] = 0
    M = M/np.max(M)

    return M

if __name__ == "__main__":
    # demo2: texture removal
    # 加载图片   
    f = cv2.imread('./images/fish.jpg')
    f = f[:,:,::-1]              # ndarray

    plt.figure("original image")
    plt.imshow(f)
    
    print("Image shape:", f.shape)
    # 参数
    rho_smooth = 2.    # Spatial kernel parameter for smoothing step
    rho_sharp = 4.     # Spatial kernel parameter for sharpening step

    # Set pixelwise sigma (range kernel parameters) for smoothing

    M = mrtv(f,5)

    sigma_smooth = linearMap(1-M,[0,1],[30,70])

  
    smooth_kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    sigma_smooth = cv2.dilate(sigma_smooth, smooth_kernel)  # Clean up the fine noise

    g = f.copy()
    g = g.astype(np.float32)

    for it in range(0,2):
        out = np.zeros(f.shape,dtype=np.float32)
        for k in range(0,f.shape[2]):
            
            out[:,:,k] = fastABF(g[:,:,k],rho_smooth,sigma_smooth,None,4)
        
        out[out>255] = 255. 
        out[out<0] = 0.
        g = out

        sigma_smooth = sigma_smooth*0.8


    g_gray = cv2.cvtColor(g.astype(np.uint8),cv2.COLOR_RGB2GRAY)
    g_gray = g_gray.astype(np.float32)

  
    [zeta,sigma_sharp] = logClassifier(g_gray,rho_sharp,[30,31])
    for it in range(0,1):     # Run more iterations for greater sharpening
        for k in range(0,f.shape[2]):
            g[:,:,k] = fastABF(g[:,:,k],rho_sharp,sigma_sharp,g[:,:,k]+zeta,5)
        g[g>255] = 255. 
        g[g<0] = 0.
     
     
    plt.figure("after texture removal")
    plt.imshow(g.astype(np.uint8))
    plt.show()