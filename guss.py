import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%matplotlib inline
#%matplotlib notebook
#%precision 3

THETA=0.5
SIGMA=0.1

def kernel_0(x0,x1):
    return np.exp(-(x0-x1)**2/2/SIGMA**2)

def kernel_1(x0,x1):
    return np.exp(-THETA*np.abs(x0-x1))

def make_kernel(xs,kernel):
    xs=xs.reshape(-1,1)        #(100,1)
    return kernel(xs,xs.T)

#-----------------------------
if __name__ == "__main__":
    N=100
    xs=np.linspace(0,1,N)
    
    k0=make_kernel(xs,kernel_0)
    k1=make_kernel(xs,kernel_1)
    mean=np.zeros(len(xs))


    fig = plt.figure()
    ax = Axes3D(fig)

    print("xs =",xs.shape)

    ys = xs
    X,Y = np.meshgrid(xs, ys)
    Z=k0

#    ax.plot_wireframe(X,Y,Z)
    ax.plot_surface(X,Y,Z)

    plt.show()

    plt.savefig("kernel.png")


#-----------------------------
    np.random.seed(1)
    ys=np.random.multivariate_normal(mean,k0,5)
    for y in ys:
       plt.plot(xs,y)

    plt.title("guss")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.savefig("guss.jpg")
    plt.show()
    
    ys=np.random.multivariate_normal(mean,k1,5)
    for y in ys:
       plt.plot(xs,y)

    plt.title("abs")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.savefig("abs.jpg")
    plt.show()
    
