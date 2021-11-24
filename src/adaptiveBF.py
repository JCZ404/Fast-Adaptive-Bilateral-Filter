# 暴力法实现adaptive bilateral filtering
import numpy as np
import PIL.Image as Image
import cv2
import os
import sys
import matplotlib.pyplot as plt
import time
import numba as nb

# 参考：https://stackoverflow.com/questions/23471083/create-2d-log-kernel-in-opencv-like-fspecial-in-matlab
# 获得LoG滤波器的卷积核
def get_log_kernel(siz, std):
    assert siz%2 == 1, "kernel size must be odd!"
    siz = (siz-1)/2
    x = y = np.linspace(-siz, siz, 2*siz+1)
    x, y = np.meshgrid(x, y)
    arg = -(x**2 + y**2) / (2*std**2)
    h = np.exp(arg)
    h[h < sys.float_info.epsilon * h.max()] = 0
    h = h/h.sum() if h.sum() != 0 else h
    h1 = h*(x**2 + y**2 - 2*std**2) / (std**4)
    return h1 - h1.mean()

# 参考：https://blog.csdn.net/ckghostwj/article/details/12177273
# 获得Gauss滤波器的卷积核
def get_gaussian_kernel(siz, std):
    assert siz%2 == 1, "kernel size must be odd!"
    siz = (siz-1)/2
    x = y = np.linspace(-siz, siz, 2*siz+1)
    x, y = np.meshgrid(x, y)
    arg = -(x**2 + y**2) / (2*std**2)
    h = np.exp(arg)
    h1 = h/np.sum(h)
    return h1

# 获得正方形均值滤波器的卷积核
def get_square_kernel(siz, val):
    assert siz%2 == 1, "kernel size must be odd!"
    h = np.ones(int(siz))
    h1 = h*val
    return h1


def imfilter(img, kernel, mode):
  
    assert len(img.shape) == 2, "only support gray image!"
    h,w = img.shape
    k_size = kernel.shape[0]
    assert k_size%2 == 1, "kernel size must be add!"
    
    # padding
    # 边界对称填充
    # 参考：https://blog.csdn.net/it_flying625/article/details/104592233
    padding = k_size//2
    if mode == "constant":
        dst = np.pad(img,(padding, padding),"constant")
    elif mode == "symmetric":
        dst = np.pad(img,(padding, padding),"symmetric")
    
    # ps: need convert to float precesion
    dst = dst.astype(np.float32)
    # filtering
    for y in range(h):
        for x in range(w):
            dst[y, x] = np.sum(kernel * dst[y: y + k_size, x: x + k_size])

    # 剪切
    dst = dst[: h, :w]

    return dst
    

def linearMap( X,xrange,yrange):

    x1 = xrange[0]
    x2 = xrange[1]
    y1 = yrange[0]
    y2 = yrange[1]
    m = (y2-y1)/(x2-x1)
    c = (x2*y1-x1*y2)/(x2-x1)
    Y = m*X + c

    return Y

def logClassifier(f, rho, sigma_interval):
    h_log = get_log_kernel(9,9/6)
    f_log = imfilter(f,h_log,'symmetric')

    L = np.zeros(f.shape)
    L[f_log>60] = 60
    L[f_log<-60] = -60
    mask = (f_log>-60)& (f_log<60)
    L[mask] = f_log[mask]
    L = np.round(L*2)/2


    sigma_r = linearMap(np.abs(L),[0,np.max(np.abs(L))],[sigma_interval[1],sigma_interval[0]])

    h_square = get_square_kernel(6*rho+1,(1./(6*rho+1)**2))
    fbar = imfilter(f,h_square,'symmetric')
    zeta = f - fbar
    return zeta, sigma_r



def MinMaxFilter(F,w):

    """计算数组F中每个元素的k x k的window中的最大值和最小值

    Args:
        F ([numpy]): [the array that needs to calucate the local min and max element]
        w ([int]): [windows of each element in F to calculate the min and max]
    Return:
        Min ([numpy]): each element is the min element of w x w window in F
        Max ([numpy]): each element is the max element of w x w window in F
    """
    minput, ninput = F.shape

    sym = (w-1)/2
    sym = int(sym)
    
    rowpad = (minput/w + (1 if minput%w != 0 else 0))*w - minput
    columnpad = (ninput/w + (1 if ninput%w != 0 else 0))*w - ninput
    rowpad = int(rowpad)
    columnpad = int(columnpad)

    
    fmin = np.zeros((minput,ninput),dtype=np.float32)
    fmax = np.zeros((minput,ninput),dtype=np.float32)
    
    templateMax = np.pad(F,((0,rowpad),(0,columnpad)),'edge').astype(np.float32)
    templateMin = np.pad(F,((0,rowpad),(0,columnpad)),'edge').astype(np.float32)

    m = minput+rowpad
    n = ninput+columnpad
    
    rmax,lmax,rmin,lmin = None, None, None, None
    tminptr, tmaxptr = None, None

    #! Scan along rows 
    Lmax = np.zeros((n,))
    Lmin = np.zeros((n,))
    Rmax = np.zeros((n,))
    Rmin = np.zeros((n,))
    for ii in range(1,minput+1):
        # 取每一行进行操作
        tmaxptr = templateMax[ii-1]
        tminptr = templateMin[ii-1]
        Lmax[0] = tmaxptr[0]
        Lmin[0] = tminptr[0]
        
        Rmax[n-1] = tmaxptr[n-1]
        Rmin[n-1] = tminptr[n-1]

        for k in range(2,n+1):
            if (k-1)%w==0:
                Lmax[k-1] = tmaxptr[k-1]
                Rmax[n-k] = tmaxptr[n-k]
                Lmin[k-1] = tminptr[k-1]
                Rmin[n-k] = tminptr[n-k]
            else:
                Lmax[k-1] =  Lmax[k-2] if Lmax[k-2]>tmaxptr[k-1] else tmaxptr[k-1]
                Rmax[n-k] =  Rmax[n-k+1] if Rmax[n-k+1]>tmaxptr[n-k] else tmaxptr[n-k]
                Lmin[k-1] =  Lmin[k-2] if Lmin[k-2]<tminptr[k-1] else tminptr[k-1]
                Rmin[n-k] =  Rmin[n-k+1] if Rmin[n-k+1]<tminptr[n-k] else tminptr[n-k]

        for k in range(1,n+1):
            p = k - sym
            q = k + sym
            rmax = -1 if p<1 else Rmax[p-1]
            rmin = np.inf if p<1 else Rmin[p-1]
            lmax = -1 if q>n else Lmax[q-1]
            lmin = np.inf if q>n else Lmin[q-1]

            tmaxptr[k-1] = rmax if rmax>lmax else lmax
            tminptr[k-1] = rmin if rmin<lmin  else lmin
    

    #! Scan along columns 
    Lmax = np.zeros((m,))
    Lmin = np.zeros((m,))
    Rmax = np.zeros((m,))
    Rmin = np.zeros((m,))

    for jj in range(1,ninput+1):
        tmaxptr = templateMax[:,jj-1]
        tminptr = templateMin[:,jj-1]
        Lmax[0] = tmaxptr[0]
        Lmin[0] = tminptr[0]
        Rmax[m-1] = tmaxptr[m-1]
        Rmin[m-1] = tminptr[m-1]
        for k in range(2,m+1):
            if (k-1)%w==0:
                Lmax[k-1] = tmaxptr[k-1]
                Rmax[m-k] = tmaxptr[m-k]
                Lmin[k-1] = tminptr[k-1]
                Rmin[m-k] = tminptr[m-k]
            
            else:
                Lmax[k-1] = Lmax[k-2] if Lmax[k-2]>tmaxptr[k-1] else tmaxptr[k-1]
                Rmax[m-k] = Rmax[m-k+1] if Rmax[m-k+1]>tmaxptr[m-k] else tmaxptr[m-k]
                Lmin[k-1] = Lmin[k-2] if Lmin[k-2]<tminptr[k-1] else tminptr[k-1]
                Rmin[m-k] = Rmin[m-k+1] if Rmin[m-k+1]<tminptr[m-k] else tminptr[m-k]
            
        
        for k in range(1, m+1):
            p = k - sym
            q = k + sym
            rmax = -1 if p<1 else Rmax[p-1]
            rmin = np.inf if p<1 else Rmin[p-1]
            lmax = -1 if q>m else Lmax[q-1]
            lmin = np.inf if q>m else Lmin[q-1]
            if k<=minput:
                fmax[k-1,jj-1] = rmax if rmax>lmax else lmax
                fmin[k-1,jj-1] = rmin if rmin<lmin else lmin
    return fmin, fmax

import math
# 计算积分
from scipy import special
def compInt(N,lbda,t0 ):
    I = np.zeros((lbda.shape[0],N+1))
    zero = 0
    rootL = np.sqrt(lbda)
    Ulim = lbda*(1-t0)*(1-t0)

    expU = np.exp(-Ulim)

    # Compute I_0 and I_1 directly
    I[:,zero] = 0.5 * np.sqrt(np.pi/lbda) * (special.erf(rootL*(1-t0)) - special.erf(-rootL*t0))

    I[:,zero+1] = t0*I[:,zero] - (expU - np.exp(-lbda*t0*t0))/(2*lbda)

    # Use recurrence relation for k>1
    for k in range(2,N+1):   
        I[:,zero+k] = t0*I[:,zero+k-1] + ((k-1)*I[:,zero+k-2] - expU)/(2*lbda)
    
    return I


# 希尔伯特矩阵
def hilb(n):
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i][j] = 1/(i+j+1)
    return H

# 希尔伯特矩阵的逆矩阵
def invhilb(n):
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i][j] = 1/(i+j+1)
    return np.linalg.inv(H)


def fitPolynomial( f,rho,N,Alpha,Beta,filtertype ):
    if filtertype == 'gaussian':
        spker = get_gaussian_kernel(6*rho+1,rho)
    else:
        spker = get_square_kernel(2*rho+1,1./(2*rho+1)**2)


    # Find pixels where Alpha=Beta or Alpha=0
    mask = (Alpha!= Beta)
    num_pixels = np.count_nonzero(mask)
    Alpha = Alpha[mask]
    Beta = Beta[mask]
    Alpha0_mask = (Alpha==0)
    Alpha_non0 = Alpha[~Alpha0_mask]
    Beta_non0 = Beta[~Alpha0_mask]

    # Filter powers of f
    fpow_filt = np.zeros((np.count_nonzero(mask),N+1))
    fpow = np.ones(f.shape)
    zero = 0
    fpow_filt[:,zero] = 1
    for k in range(1,N+1):
        fpow = fpow*f
        fbar = imfilter(fpow,spker,'symmetric')
        
        fpow_filt[:,zero+k] = fbar[mask]
    # Compute moments of the shifted histograms (using numerically stable recursion)
    zero = 0
    M = np.zeros((num_pixels,N+1))
    M[:,zero] = 1;  # 0th moment is always 1
    multiplier = np.ones((np.count_nonzero(~Alpha0_mask),))
    Beta_k = np.ones((np.count_nonzero(Alpha0_mask),))
    for k in range(1,N+1):
        Beta_k = Beta_k*Beta[Alpha0_mask]
        multiplier = multiplier*(-Alpha_non0/(Beta_non0-Alpha_non0))  #! need to squeeze the multiplier to 1-dim
        prevTerm = multiplier
        non0_mom = prevTerm
        for r in range(1,k+1):
            temp1 = fpow_filt[:,zero+r]
            temp2 = fpow_filt[:,zero+r-1]
            nextTerm = ((r-k-1)/r) * temp1[~Alpha0_mask]/(Alpha_non0*temp2[~Alpha0_mask]) *prevTerm
            non0_mom = non0_mom + nextTerm
            prevTerm = nextTerm
    
        mom_k = np.zeros((num_pixels,))
        mom_k[~Alpha0_mask] = non0_mom #! need to unsqueeze the non0_mom at the last dim
        temp = fpow_filt[:,zero+k]
        
        mom_k[Alpha0_mask] = temp[Alpha0_mask]/(Beta_k)
        M[:,zero+k] = mom_k

    M = M.T

    # Compute polynomial coefficients
    Hinv = invhilb(N+1)
    C = np.dot(Hinv,M)
    C = C.T

    return C


# 快速Adaptive Bilater Filtering
def fastABF( f,rho,sigma_r,theta=None,N=None,filtertype=None ):
    if theta is None:
        theta = f
    if N is None:
        N = 5
    if filtertype is None:
        filtertype = "gaussian"

    [rr,cc] = f.shape
    f = f/255
    theta = theta/255
    sigma_r = sigma_r/255

    # Compute local histogram range
    if filtertype == 'gaussian':

        #! 统计MinMaxFilter的用时
        t1 = time.time()
        [Alpha,Beta] = MinMaxFilter(f,6*rho+1)
        t2 = time.time()
        print("Time of MinMaxFilter: ",t2-t1)

    elif filtertype == 'box':
        [Alpha,Beta] = MinMaxFilter(f,2*rho+1)
    else:
        raise('Invalid filter type')
    
    mask = (Alpha!=Beta);   # Mask for pixels with Alpha~=Beta
    a = np.zeros((rr,cc))
    a[mask] = 1./(Beta[mask]-Alpha[mask])

    # Compute polynomial coefficients at every pixel
    #! 统计fitPolynomial的用时
    t1 =time.time()
    C = fitPolynomial(f,rho,N,Alpha,Beta,filtertype)
    t2 = time.time()
    print("Time of FitPolynomial: ",t2-t1)

    # Pre-compute integrals at every pixel
    zero = 0
    t0 = (theta[mask]-Alpha[mask])/(Beta[mask]-Alpha[mask])
    lbda = 1/(2*sigma_r[mask]*sigma_r[mask]*a[mask]*a[mask])
    #! 统计compInt的用时
    t1 = time.time()
    I = compInt(N+1,lbda,t0)
    t2 = time.time()
    print("Time of compInt: ",t2-t1)

    # Compute numerator and denominator
    Num = np.zeros((np.count_nonzero(mask),))
    Den = Num.copy()
    for k in range(0,N+1):
        Ck = C[:,zero+k]
        Num = Num + Ck*I[:,zero+k+1]
        Den = Den + Ck*I[:,zero+k]
    

    # Undo shifting & scaling to get output (eq. 29 in paper)
    g_hat = np.zeros((rr,cc))
    g_hat[mask] = Alpha[mask] + (Beta[mask]-Alpha[mask])*Num/Den
    g_hat[~mask] = f[~mask]
    g_hat = 255*g_hat

    return g_hat

# 定义暴力法的Adaptive Bilateral filtering
def abf_bruteforce(f, rho, sigma_r, theta=None, filtertype=None):
    if theta is None:
        theta = f
    if filtertype is None:
        filtertype = "gaussian"

    [fr, fc] = f.shape
  
    if filtertype == 'gaussian':
        rad = 3*rho
    elif filtertype == 'box':
        rad = rho
    

    f = np.pad(f,(rad, rad),'symmetric')
    f = f.astype(np.float32)


    if filtertype == 'gaussian':
        omega = get_gaussian_kernel(2*rad+1,rho) 
    elif filtertype == 'box':
        omega = get_square_kernel(2*rad+1, 1)
    

    W = np.zeros((fr,fc))
    Z = np.zeros((fr,fc))
    for j1  in range(rad+1,rad+fr):
        for j2 in range( rad+1,rad+fc):
            nb = f[j1-rad:j1+rad+1,j2-rad:j2+rad+1]
            r_arg = (nb - theta[j1-rad,j2-rad])**2
            rker = np.exp(-0.5*r_arg/(sigma_r[j1-rad,j2-rad]**2))
            W[j1-rad,j2-rad] = np.sum(np.sum(omega * rker * nb))
            Z[j1-rad,j2-rad] = np.sum(np.sum(omega * rker))

    Bf = W / (Z+1e-6)

    return Bf
    