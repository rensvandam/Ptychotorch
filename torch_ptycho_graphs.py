# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 11:28:48 2022

Some graphying functions for ptychography simulations in Pytorch

@author: Rens van Dam
         p.s.vandam@students.uu.nl
"""
#%% Import modules

import pickle
import matplotlib
from matplotlib import pyplot
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
import torch as to
device = to.device("cuda") if to.cuda.is_available() else to.device("cpu")
#to.set_default_tensor_type('torch.cuda.FloatTensor')


#%%

with open("diffpat", "rb") as fp:
    diffpat = pickle.load(fp)

pyplot.figure(figsize=(5,3.5))
pyplot.imshow(np.abs(diffpat), cmap='turbo')
pyplot.colorbar()
pyplot.show()

def bandlimit(f):

    Nx = 256
    Ny = 256
    pixelsize = 0.005
    dx = pixelsize
    dy = pixelsize
    λ = 6E-4
    z = 200
    def rect(x):
        return to.where(abs(x)<=0.3, 1, 0)

    f = to.nn.functional.pad(f, (int((Nx-f.shape[0])/2), int((Nx-f.shape[0])/2), int((Ny-f.shape[1])/2), int((Ny-f.shape[1])/2)), "constant", 0)

    fft_c = to.fft.fft2(f)
    c = to.fft.fftshift(fft_c)

    # transfer function

    u = to.fft.fftshift(to.fft.fftfreq(Nx, d = dx)).cpu()
    v = to.fft.fftshift(to.fft.fftfreq(Ny, d = dy)).cpu()
    u, v = to.meshgrid(u, v)

    H = to.exp(1j*(2*np.pi)*z/λ*(to.sqrt(1 - (λ*u) ** 2 - (λ*v) ** 2))).cpu()

    S_x = Nx*dx
    S_y = Ny*dy
    du = 1/(2*S_x)
    dv = 1/(2*S_y)

    ulim = 1/( to.sqrt( to.tensor( (2*z*du)**2 +1 ) )*λ ).cpu()
    vlim = 1/( to.sqrt( to.tensor( (2*z*dv)**2 +1 ) )*λ ).cpu()

    H_BL = H*rect(0.5*(u/ulim))*rect(0.5*(v/vlim)).cpu()

    # pyplot.figure(figsize=(5,3.5))
    # pyplot.imshow(to.abs(H_BL), cmap='turbo')
    # pyplot.colorbar()
    # pyplot.show()


    #g = to.fft.ifft2(to.fft.ifftshift(c * H))
    g = to.fft.ifft2(to.fft.ifftshift(c * H_BL))
    g = to.abs(g)**2

    return g


diffpat = to.from_numpy(diffpat).to(device)

def compute_regularization(image, C):




        Nx = 256
        Ny = 256
        pixelsize = 0.005
        dx = pixelsize
        dy = pixelsize
        λ = 6E-4
        z = 200
        def rect(x):
            return to.where(abs(x)<=0.2, 1, 0)

        # transfer function

        u = to.fft.fftshift(to.fft.fftfreq(Nx, d = dx))
        v = to.fft.fftshift(to.fft.fftfreq(Ny, d = dy))
        u, v = to.meshgrid(u, v)

        S_x = Nx*dx
        S_y = Ny*dy
        du = 1/(2*S_x)
        dv = 1/(2*S_y)

        ulim = 1/( to.sqrt( to.tensor( (2*z*du)**2 +1 ) )*λ )
        vlim = 1/( to.sqrt( to.tensor( (2*z*dv)**2 +1 ) )*λ )

        X = to.ones((256,256))

        square = X*rect(0.5*(u/ulim))*rect(0.5*(v/vlim))

        #square = to.trim(square, (216,216))
        a = 100
        square[0:a, 0:256] = 1
        square[0:256, 0:a] = 1
        square[256-a:256, 0:256] = 1
        square[0:256, 256-a:256] = 1
        # square[0:20][0:256] = 1
        # square[236:256][236:256] = 1
        # square[0:20][236:256] = 1



        image2 = to.fft.fftshift(to.fft.fft2(image))

        image2 = image2*square

        image2 = to.fft.ifft2(image2)

        image2 = C*image2

        def bandlimit(f):

            Nx = 256
            Ny = 256
            pixelsize = 0.005
            dx = pixelsize
            dy = pixelsize
            λ = 6E-4
            z = 200
            def rect(x):
                return to.where(abs(x)<=0.5, 1, 0)

            f = to.nn.functional.pad(f, (int((Nx-f.shape[0])/2), int((Nx-f.shape[0])/2), int((Ny-f.shape[1])/2), int((Ny-f.shape[1])/2)), "constant", 0)

            fft_c = to.fft.fft2(f)
            c = to.fft.fftshift(fft_c)

            # transfer function

            u = to.fft.fftshift(to.fft.fftfreq(Nx, d = dx))
            v = to.fft.fftshift(to.fft.fftfreq(Ny, d = dy))
            u, v = to.meshgrid(u, v)

            H = to.exp(1j*(2*np.pi)*z/λ*(to.sqrt(1 - (λ*u) ** 2 - (λ*v) ** 2)))

            S_x = Nx*dx
            S_y = Ny*dy
            du = 1/(2*S_x)
            dv = 1/(2*S_y)

            ulim = 1/( to.sqrt( to.tensor( (2*z*du)**2 +1 ) )*λ )
            vlim = 1/( to.sqrt( to.tensor( (2*z*dv)**2 +1 ) )*λ )

            H_BL = H*rect(0.5*(u/ulim))*rect(0.5*(v/vlim))

            # pyplot.figure(figsize=(5,3.5))
            # pyplot.imshow(to.abs(H_BL), cmap='turbo')
            # pyplot.colorbar()
            # pyplot.show()


            #g = to.fft.ifft2(to.fft.ifftshift(c * H))
            g = to.fft.ifft2(to.fft.ifftshift(c * H_BL))
            g = to.abs(g)**2

            return g

#        regul = C * bandlimit(image)

        regul = image2



        return regul


pyplot.figure()
pyplot.imshow(to.abs(compute_regularization(diffpat, C=1)).cpu().detach().numpy(), cmap='turbo')
pyplot.show()


# pyplot.figure(figsize=(5,3.5))
# pyplot.imshow(to.abs(compute_regularization(diffpat, C=10)[0]).cpu().detach().numpy(), cmap='turbo')
# pyplot.colorbar()
# pyplot.show()

#%% Loading in

with open("sample", "rb") as fp:
    sample = pickle.load(fp)



# with open("mixed001_cor", "rb") as fp:
#     mixed001_cor = pickle.load(fp)

# with open("mixed001_dc", "rb") as fp:
#     mixed001_dc = pickle.load(fp)

# with open("mixed001_recon", "rb") as fp:
#     mixed001_recon = pickle.load(fp)

with open("mixed01_cor", "rb") as fp:
    mixed01_cor = pickle.load(fp)

with open("mixed01_dc", "rb") as fp:
    mixed01_dc = pickle.load(fp)

with open("mixed01_recon", "rb") as fp:
    mixed01_recon = pickle.load(fp)

with open("mixed1_cor", "rb") as fp:
    mixed1_cor = pickle.load(fp)

with open("mixed1_dc", "rb") as fp:
    mixed1_dc = pickle.load(fp)

with open("mixed1_recon", "rb") as fp:
    mixed1_recon = pickle.load(fp)

with open("mixed10_cor", "rb") as fp:
    mixed10_cor = pickle.load(fp)

with open("mixed10_dc", "rb") as fp:
    mixed10_dc = pickle.load(fp)

with open("mixed10_recon", "rb") as fp:
    mixed10_recon = pickle.load(fp)




# with open("poisson_cor", "rb") as fp:
#     poisson_cor = pickle.load(fp)

# with open("poisson_dc", "rb") as fp:
#     poisson_dc = pickle.load(fp)

# with open("poisson_recon", "rb") as fp:
#     poisson_recon = pickle.load(fp)


# with open("poisson001_cor", "rb") as fp:
#     poisson001_cor = pickle.load(fp)

# with open("poisson001_dc", "rb") as fp:
#     poisson001_dc = pickle.load(fp)

# with open("poisson001_recon", "rb") as fp:
#     poisson001_recon = pickle.load(fp)


with open("poisson01_cor", "rb") as fp:
    poisson01_cor = pickle.load(fp)

with open("poisson01_dc", "rb") as fp:
    poisson01_dc = pickle.load(fp)

with open("poisson01_recon", "rb") as fp:
    poisson01_recon = pickle.load(fp)


with open("poisson1_cor", "rb") as fp:
    poisson1_cor = pickle.load(fp)

with open("poisson1_dc", "rb") as fp:
    poisson1_dc = pickle.load(fp)

with open("poisson1_recon", "rb") as fp:
    poisson1_recon = pickle.load(fp)


with open("poisson10_cor", "rb") as fp:
    poisson10_cor = pickle.load(fp)

with open("poisson10_dc", "rb") as fp:
    poisson10_dc = pickle.load(fp)

with open("poisson10_recon", "rb") as fp:
    poisson10_recon = pickle.load(fp)



# with open("sunf_poisson1_cor", "rb") as fp:
#     sunf_poisson1_cor = pickle.load(fp)

# with open("sunf_poisson1_dc", "rb") as fp:
#     sunf_poisson1_dc = pickle.load(fp)

# with open("sunf_poisson1_recon", "rb") as fp:
#     sunf_poisson1_recon = pickle.load(fp)

# with open("gaussian_cor", "rb") as fp:
#     gaussian_cor = pickle.load(fp)

# with open("gaussian_dc", "rb") as fp:
#     gaussian_dc = pickle.load(fp)

# with open("gaussian_recon", "rb") as fp:
#     gaussian_recon = pickle.load(fp)

iterations = np.arange(len(poisson1_cor))

#%% Sunflower vs grid

# pyplot.figure(figsize=(5, 3.5))
# pyplot.plot(iterations, abs(poisson1_cor), label = 'poisson, sigma=1, grid, corr:'+str(round(float(abs(poisson1_dc)),2)))
# pyplot.plot(iterations, abs(sunf_poisson1_cor), label = 'poisson, sigma=1, fermat, corr:'+str(round(float(abs(sunf_poisson1_dc)),2)))
# pyplot.legend(loc="lower right")
# pyplot.xlabel('epoch')
# pyplot.ylabel('correlation')
# pyplot.show()

#%%Plotting the correlations for the respective loss functions


pyplot.figure(figsize=(5, 3.5))
pyplot.title("Correlations for z=2000mm")
pyplot.plot(np.arange(len(mixed01_cor)), abs(mixed01_cor), label = 'mixed, sigma=0.1, corr:'+str(round(float(abs(mixed01_dc)),2)))
pyplot.plot(np.arange(len(mixed10_cor)), abs(mixed10_cor), label = 'mixed, sigma=10, corr:'+str(round(float(abs(mixed10_dc)),2)))
pyplot.plot(iterations, abs(mixed1_cor), label = 'mixed, sigma=1, corr:'+str(round(float(abs(mixed1_dc)),2)))
pyplot.plot(np.arange(len(poisson10_cor)), abs(poisson10_cor), label = 'poisson, sigma=10, corr:'+str(round(float(abs(poisson10_dc)),2)))
pyplot.plot(np.arange(len(poisson01_cor)), abs(poisson01_cor), label = 'poisson, sigma=0.1 corr:'+str(round(float(abs(poisson01_dc)),2)))
pyplot.plot(iterations, abs(poisson1_cor), label = 'poisson, sigma=1, corr:'+str(round(float(abs(poisson1_dc)),2)))
pyplot.legend(loc = 'lower right')
pyplot.xlabel('epoch')
pyplot.ylabel('correlation')
pyplot.show()

#%% Correlations for different sigma

pyplot.figure(figsize=(10, 7))
pyplot.suptitle("Correlations for z=2000mm")
pyplot.subplot(2, 2, 1)
# pyplot.plot(iterations, abs(mixed001_cor), label = 'mixed, sigma=0.1, corr:'+str(round(float(abs(mixed001_dc)),2)))
# pyplot.plot(iterations, abs(poisson001_cor), label = 'poisson, corr:'+str(round(float(abs(poisson001_dc)),2)))
# pyplot.legend()
# pyplot.title('sigma2 = 0.01')

pyplot.subplot(2, 2, 2)
pyplot.plot(np.arange(len(mixed01_cor)), abs(mixed01_cor), label = 'mixed, corr:'+str(round(float(abs(mixed01_dc)),2)))
pyplot.plot(np.arange(len(poisson01_cor)), abs(poisson01_cor), label = 'poisson, corr:'+str(round(float(abs(poisson01_dc)),2)))
pyplot.legend()
pyplot.title('sigma2 = 0.1')

pyplot.subplot(2, 2, 3)
pyplot.plot(iterations, abs(mixed1_cor), label = 'mixed, corr:'+str(round(float(abs(mixed1_dc)),2)))
pyplot.plot(iterations, abs(poisson1_cor), label = 'poisson, corr:'+str(round(float(abs(poisson1_dc)),2)))
pyplot.legend()
pyplot.title('sigma2 = 1')

pyplot.subplot(2, 2, 4)
pyplot.plot(np.arange(len(mixed10_cor)), abs(mixed10_cor), label = 'mixed, corr:'+str(round(float(abs(mixed10_dc)),2)))
pyplot.plot(np.arange(len(poisson10_cor)), abs(poisson10_cor), label = 'poisson, corr:'+str(round(float(abs(poisson10_dc)),2)))
pyplot.legend()
pyplot.title('sigma2 = 10')

# pyplot.show()



#%% Images


pyplot.figure(figsize=(15, 7))
pyplot.suptitle("Plots for z=2000mm")
pyplot.subplot(2, 3, 1)
pyplot.imshow(np.angle(poisson01_recon), cmap='gray')
pyplot.title('Poisson, sigma2 = 0.1, C:'+str(round(float(abs(poisson01_dc)),2)))
scalebar = ScaleBar(0.005, units="mm") # 1 pixel = 5 μm
pyplot.gca().add_artist(scalebar)
pyplot.colorbar()
pyplot.subplot(2, 3, 2)
pyplot.imshow(np.angle(poisson1_recon), cmap='gray')
pyplot.title('Poisson, sigma2 = 1, C:'+str(round(float(abs(poisson1_dc)),2)))
scalebar = ScaleBar(0.005, units="mm") # 1 pixel = 5 μm
pyplot.gca().add_artist(scalebar)
pyplot.colorbar()
pyplot.subplot(2, 3, 3)
pyplot.imshow(np.angle(poisson10_recon), cmap='gray')
pyplot.title('Poisson, sigma2 = 10, C:'+str(round(float(abs(poisson10_dc)),2)))
scalebar = ScaleBar(0.005, units="mm") # 1 pixel = 5 μm
pyplot.gca().add_artist(scalebar)
pyplot.colorbar()
pyplot.subplot(2, 3, 4)
pyplot.imshow(np.angle(mixed01_recon), cmap='gray')
pyplot.title('Mixed, sigma2 = 0.1, C:'+str(round(float(abs(mixed01_dc)),2)))
scalebar = ScaleBar(0.005, units="mm") # 1 pixel = 5 μm
pyplot.gca().add_artist(scalebar)
pyplot.colorbar()
pyplot.subplot(2, 3, 5)
pyplot.imshow(np.angle(mixed1_recon), cmap='gray')
pyplot.title('Mixed, sigma2 = 1, C:'+str(round(float(abs(mixed1_dc)),2)))
scalebar = ScaleBar(0.005, units="mm") # 1 pixel = 5 μm
pyplot.gca().add_artist(scalebar)
pyplot.colorbar()
pyplot.subplot(2, 3, 6)
pyplot.imshow(np.angle(mixed10_recon), cmap='gray')
pyplot.title('Mixed, sigma2 = 10, C:'+str(round(float(abs(mixed10_dc)),2)))
scalebar = ScaleBar(0.005, units="mm") # 1 pixel = 5 μm
pyplot.gca().add_artist(scalebar)
pyplot.colorbar()
pyplot.show()


# pyplot.figure(figsize=(10, 7))
# pyplot.subplot(2, 2, 1)
# # pyplot.imshow(np.angle(poisson001_recon), cmap='turbo', label="poisson001")
# pyplot.title('Poisson, sigma2 = 0.01')
# #pyplot.colorbar()
# pyplot.subplot(2, 2, 2)
# pyplot.imshow(np.angle(poisson01_recon), cmap='turbo')
# pyplot.title('Poisson, sigma2 = 0.1')
# pyplot.colorbar()
# pyplot.subplot(2, 2, 3)
# pyplot.imshow(np.angle(poisson1_recon), cmap='turbo')
# pyplot.title('Poisson, sigma2 = 1')
# pyplot.colorbar()
# pyplot.subplot(2, 2, 4)
# pyplot.imshow(np.angle(poisson10_recon), cmap='turbo')
# pyplot.title('Poisson, sigma2 = 10')
# pyplot.colorbar()
# pyplot.show()

#%% Differences pytorch vs tensorflow

# with open("tfmixed01_cor", "rb") as fp:
#     tfmixed01_cor = pickle.load(fp)

# with open("tfmixed01_dc", "rb") as fp:
#     tfmixed01_dc = pickle.load(fp)

# with open("tfmixed01_recon", "rb") as fp:
#     tfmixed01_recon = pickle.load(fp)

# with open("tfmixed1_cor", "rb") as fp:
#     tfmixed1_cor = pickle.load(fp)

# with open("tfmixed1_dc", "rb") as fp:
#     tfmixed1_dc = pickle.load(fp)

# with open("tfmixed1_recon", "rb") as fp:
#     tfmixed1_recon = pickle.load(fp)

# with open("tfpoisson01_cor", "rb") as fp:
#     tfpoisson01_cor = pickle.load(fp)

# with open("tfpoisson01_dc", "rb") as fp:
#     tfpoisson01_dc = pickle.load(fp)

# with open("tfpoisson01_recon", "rb") as fp:
#     tfpoisson01_recon = pickle.load(fp)

# with open("tfpoisson1_cor", "rb") as fp:
#     tfpoisson1_cor = pickle.load(fp)

# with open("tfpoisson1_dc", "rb") as fp:
#     tfpoisson1_dc = pickle.load(fp)

# with open("tfpoisson1_recon", "rb") as fp:
#     tfpoisson1_recon = pickle.load(fp)

# pyplot.figure(figsize=(10, 7))
# pyplot.subplot(2, 2, 1)
# pyplot.plot(iterations, abs(tfmixed01_cor), label = 'TF mixed, sigma=0.1, corr:'+str(round(float(abs(tfmixed01_dc)),2)))
# pyplot.plot(iterations, abs(mixed01_cor), label = 'TO mixed, sigma=0.1, corr:'+str(round(float(abs(mixed01_dc)),2)))
# pyplot.legend()
# pyplot.title('sigma2 = 0.1')

# pyplot.subplot(2, 2, 2)
# pyplot.plot(iterations, abs(tfmixed1_cor), label = 'TF mixed, sigma=1, corr:'+str(round(float(abs(tfmixed1_dc)),2)))
# pyplot.plot(iterations, abs(mixed1_cor), label = 'TO mixed, sigma=1, corr:'+str(round(float(abs(mixed1_dc)),2)))
# pyplot.legend()
# pyplot.title('sigma2 = 1')

# pyplot.subplot(2, 2, 3)
# pyplot.plot(iterations, abs(tfpoisson01_cor), label = 'TF poisson, sigma=0.1, corr:'+str(round(float(abs(tfpoisson01_dc)),2)))
# pyplot.plot(iterations, abs(poisson01_cor), label = 'TO poisson, sigma=0.1, corr:'+str(round(float(abs(poisson01_dc)),2)))
# pyplot.legend()
# pyplot.title('sigma2 = 0.1')

# pyplot.subplot(2, 2, 4)
# pyplot.plot(iterations, abs(tfpoisson1_cor), label = 'TF poisson, sigma=1, corr:'+str(round(float(abs(tfpoisson1_dc)),2)))
# pyplot.plot(iterations, abs(poisson1_cor), label = 'TO poisson, sigma=1, corr:'+str(round(float(abs(poisson01_dc)),2)))
# pyplot.legend()
# pyplot.title('sigma2 = 1')

# pyplot.show()



#%% Differences

# pyplot.figure(figsize=(10, 7))
# pyplot.subplot(2, 2, 1)
# pyplot.imshow(np.angle(sample) - mixed01_recon, cmap='turbo', label="mixed01")
# pyplot.title('Mixed, sigma2 = 0.1')
# pyplot.colorbar()
# pyplot.subplot(2, 2, 2)
# pyplot.imshow(np.angle(sample) - mixed1_recon, cmap='turbo')
# pyplot.title('Mixed, sigma2 = 1')
# pyplot.colorbar()
# pyplot.subplot(2, 2, 3)
# pyplot.imshow(np.angle(sample) - mixed10_recon, cmap='turbo')
# pyplot.title('Mixed, sigma2 = 10')
# pyplot.colorbar()
# pyplot.subplot(2, 2, 4)
# pyplot.imshow(np.angle(sample) - poisson_recon, cmap='turbo')
# pyplot.title('Poisson')
# pyplot.colorbar()
# pyplot.show()




# with open("recon_torch", "rb") as fp:   # Unpickling
#    recon_torch = pickle.load(fp)

# with open("recon_tf", "rb") as fp:   # Unpickling
#    recon_tf = pickle.load(fp)

# with open("correlations_torch", "rb") as fp:   # Unpickling
#    correlations_torch = pickle.load(fp)

# with open("correlations_tf", "rb") as fp:   # Unpickling
#    correlations_tf = pickle.load(fp)

# with open("iterations", "rb") as fp:   # Unpickling
#    iterations = pickle.load(fp)

# print(correlations_torch)
# print(correlations_tf)


# pyplot.figure(figsize=(5, 3.5))
# pyplot.plot(iterations, correlations_torch, label = 'torch')
# pyplot.plot(iterations, correlations_tf, label = 'tensorflow')
# pyplot.legend()
# pyplot.xlabel('iteration')
# pyplot.ylabel('correlation')
# pyplot.show()

# pyplot.figure(figsize=(5, 3.5))
# pyplot.imshow(recon_torch, cmap='twilight')
# pyplot.colorbar()
# pyplot.show()

# pyplot.figure(figsize=(5, 3.5))
# pyplot.imshow(recon_torch, cmap='twilight')
# pyplot.colorbar()
# pyplot.show()

# pyplot.figure(figsize=(5, 3.5))
# pyplot.title("Poisson loss, 40it, torch, correlation:"+str(correlations[-1]))
# pyplot.imshow(recon_torch - recon_tf, cmap='twilight')
# pyplot.colorbar()
# pyplot.show()


#%% Distance z vs correlation

with open("50P1_dc", "rb") as fp:
    a = pickle.load(fp)

with open("100P1_dc", "rb") as fp:
    b = pickle.load(fp)

with open("150P1_dc", "rb") as fp:
    c = pickle.load(fp)

with open("200P1_dc", "rb") as fp:
    d = pickle.load(fp)

with open("250P1_dc", "rb") as fp:
    e = pickle.load(fp)

with open("300P1_dc", "rb") as fp:
    f = pickle.load(fp)

with open("500P1_dc", "rb") as fp:
    g = pickle.load(fp)

with open("1000P1_dc", "rb") as fp:
    h = pickle.load(fp)

with open("1500P1_dc", "rb") as fp:
    i = pickle.load(fp)

with open("2000P1_dc", "rb") as fp:
    j = pickle.load(fp)

z_correlations = np.array([a,b,c,d,e,f,g,h,i,j])
z_correlations = abs(z_correlations)
z = [50,100,150,200,250,300,500,1000,1500,2000]

pyplot.figure(figsize=(5,3.5))
pyplot.title("Correlation vs distance between planes")
pyplot.plot(z, z_correlations)
pyplot.show()


#%% Bandwidth filter

with open("BFpoisson1_cor", "rb") as fp:
    BFpoisson1_cor = pickle.load(fp)

with open("BFpoisson1_dc", "rb") as fp:
    BFpoisson1_dc = pickle.load(fp)

with open("BFpoisson1_recon", "rb") as fp:
    BFpoisson1_recon = pickle.load(fp)

with open("BFmixed1_cor", "rb") as fp:
    BFmixed1_cor = pickle.load(fp)

with open("BFmixed1_dc", "rb") as fp:
    BFmixed1_dc = pickle.load(fp)

with open("BFmixed1_recon", "rb") as fp:
    BFmixed1_recon = pickle.load(fp)

pyplot.figure(figsize=(5, 3.5))
pyplot.title("Correlations for z=2000mm")
pyplot.plot(iterations, abs(mixed1_cor), label = 'mixed, sigma=1, corr:'+str(round(float(abs(mixed1_dc)),2)))
pyplot.plot(iterations, abs(BFmixed1_cor), label = 'mixed, sigma=1, BF, corr:'+str(round(float(abs(BFmixed1_dc)),2)))
pyplot.plot(iterations, abs(poisson1_cor), label = 'poisson, sigma=1, corr:'+str(round(float(abs(poisson1_dc)),2)))
pyplot.plot(iterations, abs(BFpoisson1_cor), label = 'poisson, sigma=1, BF, corr:'+str(round(float(abs(BFpoisson1_dc)),2)))
pyplot.legend(loc = 'lower right')
pyplot.xlabel('epoch')
pyplot.ylabel('correlation')
pyplot.show()


#%% AS vs Fraunhofer

with open("ASP2000_recon", "rb") as fp:
    ASP2000_recon = pickle.load(fp)

with open("ASP2000_cor", "rb") as fp:
    ASP2000_cor = pickle.load(fp)

with open("ASP2000_dc", "rb") as fp:
    ASP2000_dc = pickle.load(fp)

with open("ASMi2000_recon", "rb") as fp:
    ASMi2000_recon = pickle.load(fp)

with open("ASMi2000_cor", "rb") as fp:
    ASMi2000_cor = pickle.load(fp)

with open("ASMi2000_dc", "rb") as fp:
    ASMi2000_dc = pickle.load(fp)


with open("ASP200_recon", "rb") as fp:
    ASP200_recon = pickle.load(fp)

with open("ASP200_cor", "rb") as fp:
    ASP200_cor = pickle.load(fp)

with open("ASP200_dc", "rb") as fp:
    ASP200_dc = pickle.load(fp)

with open("ASMi200_recon", "rb") as fp:
    ASMi200_recon = pickle.load(fp)

with open("ASMi200_cor", "rb") as fp:
    ASMi200_cor = pickle.load(fp)

with open("ASMi200_dc", "rb") as fp:
    ASMi200_dc = pickle.load(fp)


with open("ASP50_recon", "rb") as fp:
    ASP50_recon = pickle.load(fp)

with open("ASP50_cor", "rb") as fp:
    ASP50_cor = pickle.load(fp)

with open("ASP50_dc", "rb") as fp:
    ASP50_dc = pickle.load(fp)

with open("ASMi50_recon", "rb") as fp:
    ASMi50_recon = pickle.load(fp)

with open("ASMi50_cor", "rb") as fp:
    ASMi50_cor = pickle.load(fp)

with open("ASMi50_dc", "rb") as fp:
    ASMi50_dc = pickle.load(fp)

with open("FraunhoferP_recon", "rb") as fp:
    FraunhoferP_recon = pickle.load(fp)

with open("FraunhoferP_cor", "rb") as fp:
    FraunhoferP_cor = pickle.load(fp)

with open("FraunhoferP_dc", "rb") as fp:
    FraunhoferP_dc = pickle.load(fp)


with open("FraunhoferMi_recon", "rb") as fp:
    FraunhoferMi_recon = pickle.load(fp)

with open("FraunhoferMi_cor", "rb") as fp:
    FraunhoferMi_cor = pickle.load(fp)

with open("FraunhoferMi_dc", "rb") as fp:
    FraunhoferMi_dc = pickle.load(fp)

it100 = np.arange(len(ASP50_cor))
it50 = np.arange(len(ASP200_cor))
pyplot.figure(figsize=(10, 3.5))
pyplot.suptitle("Correlations")
pyplot.plot(it100, abs(ASP50_cor), label = 'AS, Poisson, sigma=1, z = 50mm, corr:'+str(round(float(abs(ASP50_dc)),2)), color='r', linestyle = 'dotted')
pyplot.plot(it100, abs(ASMi50_cor), label = 'AS, mixed, sigma=1, z = 50mm, corr:'+str(round(float(abs(ASMi50_dc)),2)),color='b', linestyle = 'dotted')

pyplot.plot(it50, abs(ASP200_cor), label = 'AS, Poisson, sigma=1, z = 200mm, corr:'+str(round(float(abs(ASP200_dc)),2)),color='r', linestyle = 'dashdot')
pyplot.plot(it100, abs(ASMi200_cor), label = 'AS, mixed, sigma=1, z = 200mm, corr:'+str(round(float(abs(ASMi200_dc)),2)),color='b', linestyle = 'dashdot')

pyplot.plot(it50, abs(ASP2000_cor), label = 'AS, Poisson, sigma=1, z = 2000mm, corr:'+str(round(float(abs(ASP2000_dc)),2)), color='r')
pyplot.plot(it50, abs(ASMi2000_cor), label = 'AS, mixed, sigma=1, z = 2000mm, corr:'+str(round(float(abs(ASMi2000_dc)),2)), color='b')

pyplot.plot(it50, abs(FraunhoferP_cor), label = 'Fr., Poisson, sigma=1, corr:'+str(round(float(abs(FraunhoferP_dc)),2)), color='r', linestyle = 'dashed')
pyplot.plot(it50, abs(FraunhoferMi_cor), label = 'Fr., Mixed, sigma=1, corr:'+str(round(float(abs(FraunhoferMi_dc)),2)), color='b', linestyle = 'dashed')

pyplot.legend()
pyplot.xlabel('epoch')
pyplot.ylabel('correlation')
pyplot.show()

#%% Correlation Versus Sigma2

#Poisson loss function
with open("POI_CvS2_cor", "rb") as fp:
    POIC = pickle.load(fp)

with open("POI_CvS2_img", "rb") as fp:
    POIR = pickle.load(fp)

with open("POI_CvS2_par", "rb") as fp:
    POIP = pickle.load(fp)


with open("SGDPOI_CvS2_cor", "rb") as fp:
    POICsgd = pickle.load(fp)

with open("SGDPOI_CvS2_img", "rb") as fp:
    POIRsgd = pickle.load(fp)

with open("SGDPOI_CvS2_par", "rb") as fp:
    POIPsgd = pickle.load(fp)




#Mixed loss function
with open("MIX_CvS2_cor", "rb") as fp:
    MIXC = pickle.load(fp)

with open("MIX_CvS2_img", "rb") as fp:
    MIXR = pickle.load(fp)

with open("MIX_CvS2_par", "rb") as fp:
    MIXP = pickle.load(fp)


with open("SGDMIX_CvS2_cor", "rb") as fp:
    MIXCsgd = pickle.load(fp)

with open("SGDMIX_CvS2_img", "rb") as fp:
    MIXRsgd = pickle.load(fp)

with open("SGDMIX_CvS2_par", "rb") as fp:
    MIXPsgd = pickle.load(fp)





#mixed without negatives
with open("NNMIX_CvS2_cor", "rb") as fp:
    NNMIXC = pickle.load(fp)

with open("NNMIX_CvS2_img", "rb") as fp:
    NNMIXR = pickle.load(fp)





Nstat = 20

# errorbarMIX = np.std([i[0] for i in MIXC][-Nstat:], ddof = 1) / np.sqrt(len([i[0] for i in MIXC][-Nstat:]))
# errorbarPOI = np.std([i[0] for i in POIC][-Nstat:], ddof = 1) / np.sqrt(len([i[0] for i in POIC][-Nstat:]))


fig = pyplot.figure(figsize=(7, 4.5))
ax = fig.add_subplot(1,1,1)
pyplot.rcParams.update({'font.size': 15})
#pyplot.title("Recon quality for varying noise. Tot. probe intensity = {:E}  ".format(round(float(POIP['probe_totalintensity'].detach().numpy()), 0)))

ax.errorbar([i[2] for i in MIXC][0:-5], [i[0]  for i in MIXC][0:-5], yerr =  0, label="Mixed (ADAM)", color = 'midnightblue')
ax.errorbar([i[2] for i in MIXCsgd][0:-5], [i[0] for i in MIXCsgd][0:-5], yerr =  0, label="Mixed (SGD)", color = 'blue',  linestyle = '--')
ax.errorbar([i[2] for i in NNMIXC][0:-5], [i[0] for i in NNMIXC][0:-5], yerr =  0, label="Mixed (ADAM, NN)", color = 'magenta')
ax.errorbar([i[2] for i in POIC][0:-5], [i[0] for i in POIC][0:-5], yerr =  0, label="Poisson (ADAM)", color = 'darkred')
ax.errorbar([i[2] for i in POICsgd][0:-5], [i[0] for i in POICsgd][0:-5], yerr = 0, label="Poisson (SGD)", color = 'red',  linestyle = '--')






ax.set_xlabel(r"$\sigma_g^2$ (noise variance)")
ax.set_ylabel(r"$CC$ (complex correlation)")
ax.legend()
ax.set_xticks([i[2] for i in POIC][0:-5])
pyplot.show()
fig.tight_layout()




#--------- IMAGES ---------

# pyplot.figure(figsize=(10,3.5))
# pyplot.suptitle(r"Reconstructions with $\sigma^2$ (noise variance) = {}".format([i[2] for i in POIC][8]))
# pyplot.subplot(1, 3, 1)
# pyplot.imshow(np.abs(POIR[8]))
# pyplot.title("Poisson loss function")
# pyplot.subplot(1, 3, 2)
# pyplot.imshow(np.abs(MIXR[8]))
# pyplot.title("Mixed loss function")
# pyplot.subplot(1, 3, 3)
# pyplot.imshow(np.abs(NNMIXR[8]))
# pyplot.title("Mixed loss function no negatives")
# pyplot.show()



#%% Correlation versus probe intensity

#%%Corr v probe int

#Poisson loss function
with open("POI_CvPI_cor", "rb") as fp:
    POIC = pickle.load(fp)

with open("POI_CvPI_img", "rb") as fp:
    POIR = pickle.load(fp)

with open("POI_CvPI_par", "rb") as fp:
    POIP = pickle.load(fp)


with open("SGDPOI_CvPI_cor", "rb") as fp:
    POICsgd = pickle.load(fp)

with open("SGDPOI_CvPI_img", "rb") as fp:
    POIRsgd = pickle.load(fp)

with open("SGDPOI_CvPI_par", "rb") as fp:
    POIPsgd = pickle.load(fp)




#Mixed loss function
with open("MIX_CvPI_cor", "rb") as fp:
    MIXC = pickle.load(fp)

with open("MIX_CvPI_img", "rb") as fp:
    MIXR = pickle.load(fp)

with open("MIX_CvPI_par", "rb") as fp:
    MIXP = pickle.load(fp)


with open("SGDMIX_CvPI_cor", "rb") as fp:
    MIXCsgd = pickle.load(fp)

with open("SGDMIX_CvPI_img", "rb") as fp:
    MIXRsgd = pickle.load(fp)

with open("SGDMIX_CvPI_par", "rb") as fp:
    MIXPsgd = pickle.load(fp)


with open("SGDMIX_CvPI_cor", "rb") as fp:
    lMIXCsgd = pickle.load(fp)

with open("SGDMIX_CvPI_img", "rb") as fp:
    lMIXRsgd = pickle.load(fp)

with open("SGDMIX_CvPI_par", "rb") as fp:
    lMIXPsgd = pickle.load(fp)

# with open("NNMIX_CvPI_cor", "rb") as fp:
#     NNMIXC = pickle.load(fp)

# with open("NNMIX_CvPI_img", "rb") as fp:
#     NNMIXR = pickle.load(fp)

# with open("NNMIX_CvPI_par", "rb") as fp:
#     NNMIXP = pickle.load(fp)

# with open("NNSGDMIX_CvPI_cor", "rb") as fp:
#     NNMIXCsgd = pickle.load(fp)

# with open("NNSGDMIX_CvPI_img", "rb") as fp:
#     NNMIXRsgd = pickle.load(fp)

# with open("NNSGDMIX_CvPI_par", "rb") as fp:
#     NNMIXPsgd = pickle.load(fp)


# Nstat = MIXP["Ncor"]
# Nstat = 40

# errorbarMIX = np.std([i[0] for i in MIXC][-Nstat:-4], ddof = 1) / np.sqrt(len([i[0] for i in MIXC][-Nstat:-4]))
# errorbarPOI = np.std([i[0] for i in POIC][-Nstat:], ddof = 1) / np.sqrt(len([i[0] for i in POIC][-Nstat:]))

# print([i[0] for i in POIC])
# print([i[0] for i in MIXC])
# print(np.std([i[0] for i in POIC][-Nstat:], ddof = 1))
# print(np.std([i[0] for i in MIXC][-Nstat:], ddof = 1))
# errorbarMIXsgd = np.std([i[0] for i in MIXCsgd][-Nstat:-4], ddof = 1) / np.sqrt(len([i[0] for i in MIXCsgd][-Nstat:-4])) #-3 here to skip the last 3 values of SGD mixed, as they diverge and that would give a useless errorbar.
# errorbarPOIsgd = np.std([i[0] for i in POICsgd][-Nstat:], ddof = 1) / np.sqrt(len([i[0] for i in POICsgd][-Nstat:]))


#making x axis data ready for plotting
for i in range(len(MIXP["probe_totalintensity"])):
    MIXP["probe_totalintensity"][i] = float(MIXP["probe_totalintensity"][i].detach().numpy())


fig = pyplot.figure(figsize=(7, 4.5))
ax = fig.add_subplot(1,1,1)
pyplot.rcParams.update({'font.size': 15})
#pyplot.title(r"Recon quality for varying probe intensity. $\sigma^2$ = 1  ") #.format(POIP['var'])

ax.errorbar(MIXP["probe_totalintensity"], [i[0] for i in MIXC], yerr =  0, markersize = 0, lolims=False, capsize = 0, label="Mixed (ADAM)", color = 'midnightblue')
ax.errorbar(MIXPsgd["probe_totalintensity"], [i[0] for i in MIXCsgd], yerr =  0, markersize = 0,lolims=False, capsize = 0, label="Mixed (SGD)", color = 'blue', linestyle = '--')
ax.errorbar(POIP["probe_totalintensity"] , [i[0] for i in POIC], yerr =  0, markersize = 0,lolims=False, capsize = 0, label="Poisson (ADAM)", color = 'darkred')
ax.errorbar(POIPsgd["probe_totalintensity"] , [i[0] for i in POICsgd], yerr =  0, markersize = 0,lolims=False, capsize = 0, label="Poisson (SGD)", color = 'red', linestyle = '--')



# pyplot.plot(MIXP["probe_totalintensity"], [i[0] for i in NNMIXC], label="ADAM Mixed NN", color = 'green')
# pyplot.plot(MIXP["probe_totalintensity"], [i[0] for i in lMIXCsgd], label="SGD Mixed lr00000", color = 'darkgreen')


ax.set_xlabel(r" $I_{tot}$")
ax.set_ylabel(r"$CC$ (complex correlation)")
ax.set_ylim((0.985, 1.00))
ax.yaxis.set_major_locator(pyplot.MultipleLocator(0.05e-1))
ax.xaxis.set_major_locator(pyplot.MultipleLocator(1e6))
ax.legend()
fig.tight_layout()
pyplot.show()



#--------- IMAGES ---------




# pyplot.figure(figsize=(10,7))
# pyplot.title(r"Recon for varying probe intensity. $\sigma^2$ = 1  ") #.format(POIP['var'])
# pyplot.subplot(2, 2, 1)
# pyplot.imshow(np.abs(POIR[1]))
# pyplot.title("ADAM Poisson loss")
# pyplot.subplot(2, 2, 2)
# pyplot.imshow(np.abs(MIXR[1]))
# pyplot.title("ADAM Mixed loss")
# pyplot.subplot(2, 2, 3)
# pyplot.imshow(np.abs(POIRsgd[1]))
# pyplot.title("SGD Poisson loss")
# pyplot.subplot(2, 2, 4)
# pyplot.imshow(np.abs(MIXRsgd[1]))
# pyplot.title("SGD Mixed loss")
# pyplot.show()


#ADAM
pyplot.rc('font', size=19)
pyplot.rc('axes', labelsize=19)
fig,axs = pyplot.subplots(nrows=2, ncols=3)
pyplot.rc('font', size=19)
pyplot.rc('axes', labelsize=19) #fontsize of the x and y labels
fig.set_figwidth(12)
fig.set_figheight(8)
fig.tight_layout()

for i,ax in enumerate(axs.flat, start=97):
    ax.text(0.02,0.88,'('+chr(i)+')',fontsize=20, color='white', transform=ax.transAxes)

print(axs)

axs[0,0].imshow(np.abs(np.abs(MIXR[0])), cmap = 'gray')
axs[0,0].set_ylabel("Mixed")
axs[0,0].set_title("$I_{tot} = $"+"{:.3e}".format(float(MIXP["probe_totalintensity"][0])))

axs[0,1].imshow(np.abs(np.abs(MIXR[2])), cmap = 'gray')
axs[0,1].set_title("$I_{tot} = $"+"{:.3e}".format(float(MIXP["probe_totalintensity"][2])))

axs[0,2].imshow(np.abs(np.abs(MIXR[5])), cmap = 'gray')
axs[0,2].set_title("$I_{tot} = $"+"{:.3e}".format(float(MIXP["probe_totalintensity"][5])))

axs[1,0].imshow(np.abs(np.abs(POIR[0])), cmap = 'gray')
axs[1,0].set_ylabel("Poisson")

axs[1,1].imshow(np.abs(np.abs(POIR[2])), cmap = 'gray')

axs[1,2].imshow(np.abs(np.abs(POIR[5])), cmap = 'gray')

for i in [(0,1), (0,2), (1,1), (1,2)]:
    print(i)
    axs[i].axis('off')

axs[0,0].set_xticks([])
axs[0,0].set_yticks([])

axs[1,0].set_xticks([])
axs[1,0].set_yticks([])


#SGD

fig,axs = pyplot.subplots(nrows=2, ncols=3)
pyplot.rc('font', size=19)
pyplot.rc('axes', labelsize=19) #fontsize of the x and y labels
fig.set_figwidth(12)
fig.set_figheight(8)
fig.tight_layout()

for i,ax in enumerate(axs.flat, start=97):
    ax.text(0.02,0.88,'('+chr(i)+')',fontsize=20, color='white', transform=ax.transAxes)

print(axs)

axs[0,0].imshow(np.abs(np.abs(MIXRsgd[0])), cmap = 'gray')
axs[0,0].set_ylabel("Mixed")
axs[0,0].set_title("$I_{tot} = $"+"{:.3e}".format(float(MIXPsgd["probe_totalintensity"][0])))

axs[0,1].imshow(np.abs(np.abs(MIXRsgd[3])), cmap = 'gray')
axs[0,1].set_title("$I_{tot} = $"+"{:.3e}".format(float(MIXPsgd["probe_totalintensity"][3])))

axs[0,2].imshow(np.abs(np.abs(MIXRsgd[5])), cmap = 'gray')
axs[0,2].set_title("$I_{tot} = $"+"{:.3e}".format(float(MIXPsgd["probe_totalintensity"][5])))

axs[1,0].imshow(np.abs(np.abs(POIRsgd[0])), cmap = 'gray')
axs[1,0].set_ylabel("Poisson")

axs[1,1].imshow(np.abs(np.abs(POIRsgd[3])), cmap = 'gray')

axs[1,2].imshow(np.abs(np.abs(POIRsgd[5])), cmap = 'gray')

for i in [(0,1), (0,2), (1,1), (1,2)]:
    print(i)
    axs[i].axis('off')

axs[0,0].set_xticks([])
axs[0,0].set_yticks([])

axs[1,0].set_xticks([])
axs[1,0].set_yticks([])


#%% Fraunhofer vs Angular Spectrum (AS) propagation

with open("diffpat_fraunhofer", "rb") as fp:
    FH = pickle.load(fp)

with open("diffpat_angularspectrum", "rb") as fp:
    AS = pickle.load(fp)

print(np.mean(AS), np.mean(FH))
print(np.std(AS), np.std(FH))
print("AS SNR:", np.mean(AS)/np.std(AS))
print("FH SNR:", np.mean(FH)/np.std(FH))

# pyplot.figure(figsize=(10,10))
# pyplot.subplot(2, 2, 1)
# pyplot.plot(AS[128][0:256])
# pyplot.xlabel(r"$x$ (pixels)")
# pyplot.ylabel(r"$|\widehat{\psi(\theta})|^2$ (intensity)")
# pyplot.title("a)")

# pyplot.subplot(2, 2, 2)
# pyplot.plot(FH[128][0:256])
# pyplot.xlabel(r"$x$ (pixels)")
# pyplot.title("b)")

# pyplot.subplot(2, 2, 3)
# pyplot.imshow(AS)
# pyplot.xlabel(r"$x$ (pixels)")
# pyplot.ylabel(r"$y$ (pixels)")
# pyplot.colorbar()
# pyplot.title("c)")

# pyplot.subplot(2, 2, 4)
# pyplot.imshow(FH)
# pyplot.colorbar()
# pyplot.title("d)")

# pyplot.show()

with open("FRMIX_CvPI_cor", "rb") as fp:
    FRMIXC = pickle.load(fp)

with open("ASMIX_CvPI_cor", "rb") as fp:
    ASMIXC = pickle.load(fp)

print("Fraunhofer mean cc:", np.mean(np.abs([i[0] for i in FRMIXC])))
print("Fraunhofer diff cc:", np.max(np.abs([i[0] for i in FRMIXC])) - np.min(np.abs([i[0] for i in FRMIXC])))
print("Fraunhofer std cc:", np.std(np.abs([i[0] for i in FRMIXC])))

print("AS mean cc:", np.mean(np.abs([i[0] for i in ASMIXC])))

print("AS max cc:", np.max(np.abs([i[0] for i in ASMIXC])) - np.min(np.abs([i[0] for i in ASMIXC])))
print("AS std cc:", np.std(np.abs([i[0] for i in ASMIXC])))



fig,axs = pyplot.subplots(2,2)
pyplot.rc('font', size=19)
pyplot.rc('axes', labelsize=25) 

fig.set_figwidth(10)
fig.set_figheight(10)

axs[1,0].plot(AS[128][0:256])
axs[1,0].set_xlabel(r"$x$ (pixels)")
axs[1,0].set_ylabel(r"$|\widehat{\psi(\theta})|^2$ (intensity)")

axs[1,1].plot(FH[128][0:256])
axs[1,1].set_xlabel(r"$x$ (pixels)")

axs[0,0].imshow(AS)
axs[0,0].set_xlabel(r"$x$ (pixels)")
axs[0,0].set_ylabel(r"$y$ (pixels)")

axs[0,1].imshow(FH)
axs[0,1].set_xlabel(r"$x$ (pixels)")



fig.colorbar(axs[0,1].imshow(FH), ax = axs[0,1])
fig.colorbar(axs[0,0].imshow(AS), ax = axs[0,0])


for i,ax in enumerate(axs.flat, start=97):
  ax.text(0.05,0.9,'('+chr(i)+')', fontsize=19, transform=ax.transAxes)

#%% Tensorflow vs Pytorch


with open("Tensorflow_corr", "rb") as fp:
    TFC = pickle.load(fp)

with open("Tensorflow_loss", "rb") as fp:
    TFL = pickle.load(fp)

with open("Tensorflow_image", "rb") as fp:
    TFI = pickle.load(fp)


with open("2Tensorflow_corr", "rb") as fp:
    TFC2 = pickle.load(fp)

with open("2Tensorflow_loss", "rb") as fp:
    TFL2 = pickle.load(fp)

with open("2Tensorflow_image", "rb") as fp:
    TFI2 = pickle.load(fp)


with open("3Tensorflow_corr", "rb") as fp:
    TFC3 = pickle.load(fp)

with open("3Tensorflow_loss", "rb") as fp:
    TFL3 = pickle.load(fp)

with open("3Tensorflow_image", "rb") as fp:
    TFI3 = pickle.load(fp)


with open("4Tensorflow_corr", "rb") as fp:
    TFC4 = pickle.load(fp)

with open("4Tensorflow_loss", "rb") as fp:
    TFL4 = pickle.load(fp)

with open("4Tensorflow_image", "rb") as fp:
    TFI4 = pickle.load(fp)


# with open("5Tensorflow_corr", "rb") as fp:
#     TFC5 = pickle.load(fp)

# with open("5Tensorflow_loss", "rb") as fp:
#     TFL5 = pickle.load(fp)

# with open("5Tensorflow_image", "rb") as fp:
#     TFI5 = pickle.load(fp)


with open("Pytorch_corr", "rb") as fp:
    PFC = pickle.load(fp)

with open("Pytorch_loss", "rb") as fp:
    PFL = pickle.load(fp)

with open("Pytorch_image", "rb") as fp:
    PFI = pickle.load(fp)

with open("2Pytorch_corr", "rb") as fp:
    PFC2 = pickle.load(fp)

with open("2Pytorch_loss", "rb") as fp:
    PFL2 = pickle.load(fp)

with open("2Pytorch_image", "rb") as fp:
    PFI2 = pickle.load(fp)



with open("3Pytorch_corr", "rb") as fp:
    PFC3 = pickle.load(fp)

with open("3Pytorch_loss", "rb") as fp:
    PFL3 = pickle.load(fp)

with open("3Pytorch_image", "rb") as fp:
    PFI3 = pickle.load(fp)


with open("4Pytorch_corr", "rb") as fp:
    PFC4 = pickle.load(fp)

with open("4Pytorch_loss", "rb") as fp:
    PFL4 = pickle.load(fp)

with open("4Pytorch_image", "rb") as fp:
    PFI4 = pickle.load(fp)


# with open("5Pytorch_corr", "rb") as fp:
#     PFC5 = pickle.load(fp)

# with open("5Pytorch_loss", "rb") as fp:
#     PFL5 = pickle.load(fp)

# with open("5Pytorch_image", "rb") as fp:
#     PFI5 = pickle.load(fp)





with open("TFtestcorr", "rb") as fp:
    TFCt = pickle.load(fp)

with open("TFtestloss", "rb") as fp:
    TFLt = pickle.load(fp)

with open("TFtestimage", "rb") as fp:
    TFIt = pickle.load(fp)

with open("PYtestcorr", "rb") as fp:
    PFCt = pickle.load(fp)

with open("PYtestloss", "rb") as fp:
    PFLt = pickle.load(fp)

with open("PYtestimage", "rb") as fp:
    PFIt = pickle.load(fp)





with open("sample", "rb") as fp:
    GT = pickle.load(fp)

pyplot.rc('font', size=19)
pyplot.rc('axes', labelsize=19)
fig,axs = pyplot.subplots(2,2)
pyplot.rc('font', size=19)
pyplot.rc('axes', labelsize=19) #fontsize of the x and y labels
fig.set_figwidth(12)
fig.set_figheight(12)

# axs[0,0].plot(range(len(TFCt)), abs(TFCt), label="Tensorflow")
# axs[0,0].plot(range(len(PFCt)), abs(PFCt), label= "Pytorch")


axs[0,0].plot(range(len(TFC)), abs(TFC), label="Tensorflow", color='red')
axs[0,0].plot(range(len(TFC2)), abs(TFC2), color='red', alpha = 0.8)
axs[0,0].plot(range(len(TFC3)), abs(TFC3), color='red', alpha = 0.7)
axs[0,0].plot(range(len(TFC4)), abs(TFC4), color='red', alpha=0.5)

axs[0,0].plot(range(len(PFC)), abs(PFC), label= "Pytorch", color='blue')
axs[0,0].plot(range(len(PFC2)), abs(PFC2),  color='blue', alpha = 0.8)
axs[0,0].plot(range(len(PFC3)), abs(PFC3),  color='blue', alpha = 0.7)
axs[0,0].plot(range(len(PFC4)), abs(PFC4), color='blue', alpha = 0.5)

# axs[0,0].set_xlabel(r"Epoch")
# axs[0,0].set_ylabel(r"$C$ (complex correlation)")
# axs[0,0].ticklabel_format(axis='y', style='sci')
# axs[0,0].legend(loc='lower right')


# axs[0,1].plot(range(len(TFLt)), abs(TFLt), label="Tensorflow")
# axs[0,1].plot(range(len(PFLt)), abs(PFLt), label= "Pytorch")


axs[0,1].plot(range(len(TFL)), abs(TFL), label="Tensorflow", color='red')
axs[0,1].plot(range(len(TFL2)), abs(TFL2),  color='red')
axs[0,1].plot(range(len(TFL3)), abs(TFL3),color='red')
axs[0,1].plot(range(len(TFL4)), abs(TFL4),  color='red')

axs[0,1].plot(range(len(PFL)), abs(PFL), label= "Pytorch", color='blue')
axs[0,1].plot(range(len(PFL2)), abs(PFL2), color='blue')
axs[0,1].plot(range(len(PFL3)), abs(PFL3), color='blue')
axs[0,1].plot(range(len(PFL4)), abs(PFL4), color='blue')

# axs[0,1].set_xlabel(r"Epoch")
# axs[0,1].set_ylabel(r"Loss")
# axs[0,1].ticklabel_format(axis='both', style='sci')
# axs[0,1].legend()

for i,ax in enumerate(axs.flat, start=97):
    ax.text(0.05,0.9,'('+chr(i)+')',fontsize=20, transform=ax.transAxes)

axs[1,0].imshow(np.abs(TFI), cmap = 'gray')
axs[1,0].set_title("Tensorflow")
axs[1,0].set_xlabel(r"$x$ (pixels)")
axs[1,0].set_ylabel(r"$y$ (pixels)")

axs[1,1].imshow(np.abs(PFI), cmap = 'gray')
axs[1,1].set_title("Pytorch")
axs[1,1].set_xlabel(r"$x$ (pixels)")

pyplot.tight_layout()

print(abs(TFC[-1]),abs(TFC2[-1]),abs(TFC3[-1]),abs(TFC4[-1]) )
print(abs(PFC[-1]),abs(PFC2[-1]),abs(PFC3[-1]),abs(PFC4[-1]) )



fig,axs = pyplot.subplots(1,3)
pyplot.rc('font', size=15)
pyplot.rc('axes', labelsize=15) #fontsize of the x and y labels
fig.set_figwidth(8)
fig.set_figheight(8)

for i,ax in enumerate(axs.flat, start=97):
    ax.text(0.02,0.88,'('+chr(i)+')',fontsize=20, color='white', transform=ax.transAxes)

for i in [0,1,2]:
    print(i)
    axs[i].axis('off')

axs[0].imshow(np.abs(TFI2)[16:-9, 16:-9]**2, cmap = 'gray')
axs[0].set_ylabel("Tensorflow")

axs[1].imshow(np.abs(PFI2)**2, cmap = 'gray')

axs[2].imshow(np.abs(GT)[181:331, 180:331]**2, cmap = 'gray')


#axs[0,2].imshow(np.abs(TFI3)[16:-9, 16:-9], cmap = 'gray')



#axs[0,3].imshow(np.abs(TFI4)[16:-9, 16:-9], cmap = 'gray')



#axs[1,0].imshow(np.abs(PFI), cmap = 'gray')
#axs[0,0].set_ylabel("Pytorch")

#axs[1,1].imshow(np.abs(PFI2), cmap = 'gray')



#axs[1,2].imshow(np.abs(PFI3), cmap = 'gray')



#axs[1,3].imshow(np.abs(PFI4), cmap = 'gray')



# axs[0,2].imshow(np.abs(GT)[181:331, 180:331], cmap = 'gray')
# axs[0,2].set_title("Ground truth")
# axs[0,2].set_xlabel(r"$x$ (pixels)")

pyplot.tight_layout()


#%% sunflower vs grid

with open("gridMIX_CvPI_cor", "rb") as fp:
    GRMIXC = pickle.load(fp)

with open("sunflowerMIX_CvPI_cor", "rb") as fp:
    SFMIXC = pickle.load(fp)



with open("gridMIX_CvPI_img", "rb") as fp:
    GRMIXR = pickle.load(fp)

with open("sunflowerMIX_CvPI_img", "rb") as fp:
    SFMIXR = pickle.load(fp)


print("grid mean cc:", np.mean(np.abs([i[0] for i in GRMIXC])))
print("grid diff cc:", np.max(np.abs([i[0] for i in GRMIXC])) - np.min(np.abs([i[0] for i in GRMIXC])))
print("grid std cc:", np.std(np.abs([i[0] for i in GRMIXC])))

print("sunflower mean cc:", np.mean(np.abs([i[0] for i in SFMIXC])))

print("sunflower max cc:", np.max(np.abs([i[0] for i in SFMIXC])) - np.min(np.abs([i[0] for i in SFMIXC])))
print("sunflower std cc:", np.std(np.abs([i[0] for i in SFMIXC])))



fig,axs = pyplot.subplots(1,2)
pyplot.rc('font', size=15)
pyplot.rc('axes', labelsize=15) #fontsize of the x and y labels
fig.set_figwidth(8)
fig.set_figheight(8)

for i,ax in enumerate(axs.flat, start=97):
    ax.text(0.02,0.88,'('+chr(i)+')',fontsize=20, color='white', transform=ax.transAxes)

for i in [0,1]:
    print(i)
    axs[i].axis('off')

axs[0].imshow(np.abs(GRMIXR[0]), cmap = 'gray')

axs[1].imshow(np.abs(SFMIXR[0]), cmap = 'gray')


#%% Decreasing LR globally

with open("dlMIX_CvPI_cor", "rb") as fp:
    dlMIXC = pickle.load(fp)

with open("ndlMIX_CvPI_cor", "rb") as fp:
    ndlMIXC = pickle.load(fp)

with open("dlMIX_CvPI_img", "rb") as fp:
    dlMIXR = pickle.load(fp)

with open("ndlMIX_CvPI_img", "rb") as fp:
    ndlMIXR = pickle.load(fp)


print("dec lr mean cc:", np.mean(np.abs([i[0] for i in dlMIXC])))
print("dec lr diff cc:", np.max(np.abs([i[0] for i in dlMIXC])) - np.min(np.abs([i[0] for i in dlMIXC])))
print("dec lr std cc:", np.std(np.abs([i[0] for i in dlMIXC])))

print("ndl mean cc:", np.mean(np.abs([i[0] for i in ndlMIXC])))

print("ndl max cc:", np.max(np.abs([i[0] for i in ndlMIXC])) - np.min(np.abs([i[0] for i in ndlMIXC])))
print("ndl std cc:", np.std(np.abs([i[0] for i in ndlMIXC])))



# fig,axs = pyplot.subplots(1,2)
# pyplot.rc('font', size=15)
# pyplot.rc('axes', labelsize=15) #fontsize of the x and y labels
# fig.set_figwidth(8)
# fig.set_figheight(8)

# for i,ax in enumerate(axs.flat, start=97):
#     ax.text(0.02,0.88,'('+chr(i)+')',fontsize=20, color='white', transform=ax.transAxes)

# for i in [0,1]:
#     print(i)
#     axs[i].axis('off')

# axs[0].imshow(np.abs(dlMIXR[0]), cmap = 'gray')

# axs[1].imshow(np.abs(ndlMIXR[0]), cmap = 'gray')




# fig,axs = pyplot.subplots(1,2)
# pyplot.rc('font', size=15)
# pyplot.rc('axes', labelsize=15) #fontsize of the x and y labels
# fig.set_figwidth(8)
# fig.set_figheight(8)

# for i,ax in enumerate(axs.flat, start=97):
#     ax.text(0.02,0.88,'('+chr(i)+')',fontsize=20, color='white', transform=ax.transAxes)

for i in [0,1]:
    print(i)
    axs[i].axis('off')

# axs[0].plot(np.abs(dlMIXC[0]), cmap = 'gray')

# axs[1].imshow(np.abs(ndlMIXR[0]), cmap = 'gray')


with open("noLRDcorr", "rb") as fp:
    noLRDcorr = pickle.load(fp)

with open("LRDcorr", "rb") as fp:
    LRDcorr = pickle.load(fp)


fig,axs = pyplot.subplots(1,2)
pyplot.rc('font', size=20)
pyplot.rc('axes', labelsize=20) #fontsize of the x and y labels
fig.set_figwidth(8)
fig.set_figheight(4)

for i,ax in enumerate(axs.flat, start=97):
    ax.text(-0.15,0.95,'('+chr(i)+')',fontsize=20, color='black', transform=ax.transAxes)



axs[0].plot(np.abs(LRDcorr))
axs[0].set_ylabel(r"$C$ (complex correlation)")
axs[0].set_xlabel(r"Epoch")
axs[1].plot(np.abs(noLRDcorr))
axs[1].set_xlabel(r"Epoch")

pyplot.tight_layout()





fig,axs = pyplot.subplots(1,2)
pyplot.rc('font', size=20)
pyplot.rc('axes', labelsize=20) #fontsize of the x and y labels
fig.set_figwidth(8)
fig.set_figheight(3)

for i,ax in enumerate(axs.flat, start=99):
    ax.text(0.02,0.88,'('+chr(i)+')',fontsize=20, color='black', transform=ax.transAxes)



axs[0].plot(np.abs(LRDcorr))


axs[0].set_xlim((20,100))
axs[0].set_ylim((0.9874,0.989))
axs[0].set_xticks([])

axs[1].plot(np.abs(noLRDcorr))

axs[1].set_xlim((20,100))
axs[1].set_ylim((0.9874,0.989))
axs[1].set_xticks([])


pyplot.tight_layout()

#%% Bandwidht limiting


# with open("BLMIX_CvPI_cor", "rb") as fp:
#     blMIXC = pickle.load(fp)

# with open("noBLMIX_CvPI_cor", "rb") as fp:
#     nblMIXC = pickle.load(fp)

# with open("BLMIX_CvPI_img", "rb") as fp:
#     blMIXR = pickle.load(fp)

# with open("noBLMIX_CvPI_img", "rb") as fp:
#     nblMIXR = pickle.load(fp)


# print("BL mean cc:", np.mean(np.abs([i[0] for i in blMIXC])))
# print("BL diff cc:", np.max(np.abs([i[0] for i in blMIXC])) - np.min(np.abs([i[0] for i in blMIXC])))
# print("BL std cc:", np.std(np.abs([i[0] for i in blMIXC])))

# print("no BL mean cc:", np.mean(np.abs([i[0] for i in nblMIXC])))

# print("no BL max cc:", np.max(np.abs([i[0] for i in nblMIXC])) - np.min(np.abs([i[0] for i in nblMIXC])))
# print("no BL std cc:", np.std(np.abs([i[0] for i in nblMIXC])))

# fig,axs = pyplot.subplots(1,2)
# pyplot.rc('font', size=15)
# pyplot.rc('axes', labelsize=15) #fontsize of the x and y labels
# fig.set_figwidth(8)
# fig.set_figheight(8)

# for i,ax in enumerate(axs.flat, start=97):
#     ax.text(0.02,0.88,'('+chr(i)+')',fontsize=20, color='white', transform=ax.transAxes)

# for i in [0,1]:
#     print(i)
#     axs[i].axis('off')

# axs[0].imshow(np.abs(blMIXR[5]), cmap = 'gray')

# axs[1].imshow(np.abs(nblMIXR[5]), cmap = 'gray')




with open("diffpat_noBL", "rb") as fp:
    noBL = pickle.load(fp)


with open("diffpat_BL", "rb") as fp:
    BL = pickle.load(fp)

fig,axs = pyplot.subplots(2,2)
pyplot.rc('font', size=15)

fig.set_figwidth(10)
fig.set_figheight(10)

axs[1,0].plot(noBL[128][0:256])
axs[1,0].set_xlabel(r"$x$ (pixels)")
axs[1,0].set_ylabel(r"$|\widehat{\psi(\theta})|^2$ (intensity)")

axs[1,1].plot(BL[128][0:256])
axs[1,1].set_xlabel(r"$x$ (pixels)")

axs[0,0].imshow(noBL)
axs[0,0].set_xlabel(r"$x$ (pixels)")
axs[0,0].set_ylabel(r"$y$ (pixels)")

axs[0,1].imshow(BL)
axs[0,1].set_xlabel(r"$x$ (pixels)")



fig.colorbar(axs[0,1].imshow(BL), ax = axs[0,1])
fig.colorbar(axs[0,0].imshow(noBL), ax = axs[0,0])


for i,ax in enumerate(axs.flat[0:2], start=97):
  ax.text(0.05,0.9,'('+chr(i)+')', color = 'white', fontsize=19, transform=ax.transAxes)


for i,ax in enumerate(axs.flat[2:4], start=99):
  ax.text(0.05,0.9,'('+chr(i)+')', fontsize=19, transform=ax.transAxes)

#%% Regularization

with open("regMIX_CvPI_cor", "rb") as fp:
    dlMIXC = pickle.load(fp)

with open("noBLMIX_CvPI_cor", "rb") as fp:
    ndlMIXC = pickle.load(fp)

with open("regMIX_CvPI_img", "rb") as fp:
    dlMIXR = pickle.load(fp)

with open("noBLMIX_CvPI_img", "rb") as fp:
    ndlMIXR = pickle.load(fp)


print("dec lr mean cc:", np.mean(np.abs([i[0] for i in dlMIXC])))
print("dec lr diff cc:", np.max(np.abs([i[0] for i in dlMIXC])) - np.min(np.abs([i[0] for i in dlMIXC])))
print("dec lr std cc:", np.std(np.abs([i[0] for i in dlMIXC])))

print("ndl mean cc:", np.mean(np.abs([i[0] for i in ndlMIXC])))

print("ndl max cc:", np.max(np.abs([i[0] for i in ndlMIXC])) - np.min(np.abs([i[0] for i in ndlMIXC])))
print("ndl std cc:", np.std(np.abs([i[0] for i in ndlMIXC])))

fig,axs = pyplot.subplots(1,2)
pyplot.rc('font', size=15)
pyplot.rc('axes', labelsize=15) #fontsize of the x and y labels
fig.set_figwidth(8)
fig.set_figheight(8)

for i,ax in enumerate(axs.flat, start=97):
    ax.text(0.02,0.88,'('+chr(i)+')',fontsize=20, color='white', transform=ax.transAxes)

for i in [0,1]:
    print(i)
    axs[i].axis('off')

axs[0].imshow(np.abs(dlMIXR[5]), cmap = 'gray')

axs[1].imshow(np.abs(ndlMIXR[5]), cmap = 'gray')


#%% Sample and probe

def image(data, Nplots, title="title", color='turbo', colorbar=True):
    if Nplots == 1:
        pyplot.figure(figsize=(5, 3.5))
        pyplot.title(str(title))
        pyplot.imshow(data[0], cmap=color)
        if colorbar == True:
            pyplot.colorbar()
        pyplot.show()
    if Nplots == 2:
        pyplot.figure(figsize=(10, 3.5))
        pyplot.suptitle(str(title))
        pyplot.subplot(1, 2, 1)
        pyplot.imshow(data[0], cmap=color)
        if colorbar == True:
            pyplot.colorbar()
        pyplot.subplot(1, 2, 2)
        pyplot.imshow(data[1], cmap=color)
        if colorbar == True:
            pyplot.colorbar()
        pyplot.show()

def masked(image):
    #Method 1: just create a circular mask

    # mask =  np.zeros(shape=[512, 512], dtype=np.uint8)
    # cv2.circle(mask, center=(256,256), radius=150, color=(255,255,255), thickness= -1)
    # mask = np.array(mask)/255
    # image = np.multiply(image,mask)

    #Method 2: create a fading mask

    x_axis = np.linspace(-1, 1, 400)
    x_axis = np.concatenate((-np.ones(56), x_axis))
    x_axis = np.concatenate((x_axis, np.ones(56)))
    y_axis = np.linspace(-1, 1, 400)
    y_axis = np.concatenate((-np.ones(56), y_axis))
    y_axis = np.concatenate((y_axis, np.ones(56)))
    xx, yy = np.meshgrid(x_axis, y_axis)
    arr = np.sqrt(xx ** 2 + yy ** 2)

    arr[np.sqrt((yy) ** 2 + (xx) ** 2) > 1]=1

    inner = np.array([0, 0, 0])[None, None, :]
    outer = np.array([1, 1, 1])[None, None, :]

    arr /= arr.max()
    arr = arr[:, :, None]
    arr = arr * outer + (1 - arr) * inner
    arr = 1 - arr

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    arr = rgb2gray(arr)
    image = np.multiply(image, arr)

    # pyplot.figure(figsize=(5, 3.5))
    # pyplot.imshow(arr, cmap='gray')
    # pyplot.colorbar()
    # pyplot.show()
    return image


def generate_sample(mask, plot=False):

    amp = np.matrix(np.float_(matplotlib.image.imread('images/bellpepper.jpg')))
    amp = amp - np.amin(amp)
    amp = amp / np.amax(amp)

    pha = np.matrix(np.float_(matplotlib.image.imread('images/boat.png')))
    pha = pha - np.amin(pha)
    pha = pha / np.amax(pha)

    sample = np.multiply(
        (0.25 + 0.75 * amp),
        np.exp(1j * np.pi * (0.1 + 0.9 * pha)))
    #sample = normalized(sample)
    #sample /= np.max(sample) #make sure the real part of the sample (the intensity) does not exceed 1
    sample = to.from_numpy(sample)

    if mask == True:
        amp = masked(amp)
        pha = masked(pha)
        sample = np.multiply(
            (0.25 + 0.75 * amp),
            np.exp(1j * np.pi * (0.1 + 0.9 * pha)))
        sample = to.from_numpy(sample)
        #sample = masked(sample)

    if plot == True:
        image([np.abs(sample), np.angle(sample)], 2, "grid", color="gray")

    return sample

sample = generate_sample(mask=True, plot=False).to(device)

def generate_probe(probetype, probesize, pixelsize, probe_amp, plot=False):
    #x,y width
    wy = 16
    wx = 16

    #x,y pixel number
    ny = probesize[0]
    nx = probesize[1]

    #x,y pixel size
    dy = pixelsize[0]
    dx = pixelsize[1]

    # x y vector
    y = np.arange(-ny / 2, ny / 2) * dy
    x = np.arange(-nx / 2, nx / 2) * dx

    xx, yy = np.meshgrid(x,y)

    if probetype == 'gaussian':
        probe = probe_amp*np.exp(-(((yy / wy) ** 2 + (xx / wx) ** 2) / (2 * 1 ** 2)))
        probe = probe + 0*1j
        probe = to.from_numpy(probe)
        probeint = to.sum(to.abs(probe)**2)
    if plot == True:
        image([np.abs(probe), np.angle(probe)], 2, title="Probe", color='jet')

    return probe

probe = generate_probe('gaussian', (256,266), (1,1), 20, plot=False).to(device)


# fig,axs = pyplot.subplots(1,2)
# pyplot.rc('font', size=15)
# pyplot.rc('axes', labelsize=15) #fontsize of the x and y labels
# fig.set_figwidth(8)
# fig.set_figheight(8)

# print(axs)
# for i,ax in enumerate(axs.flat, start=97):
#     ax.text(0.02,0.88,'('+chr(i)+')',fontsize=20, color='white', transform=ax.transAxes)

# for i in [0,1]:
#     print(i)
#     axs[i].axis('off')

# axs[0].imshow(np.abs(sample), cmap = 'gray')
# # pyplot.colorbar(axs[0].imshow(np.abs(sample), cmap = 'gray'), shrink = 0.5)

# axs[1].imshow(np.angle(sample), cmap = 'gray')

# pyplot.colorbar(axs[1].imshow(np.angle(sample), cmap = 'gray'), shrink = 0.5)



import matplotlib.pyplot as plt

# Create the figure and the subplots
fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12,4), nrows=1, ncols=3)

# Plot the first data on the first subplot
im1 = ax1.imshow(np.abs(sample), cmap = 'gray')
ax1.axis("off")
# Plot the second data on the second subplot
im2 = ax2.imshow(np.angle(sample), cmap = 'gray')
ax2.axis("off")


im3 = ax3.imshow(np.abs(probe), cmap = 'gray')
ax3.axis("off")
# Add a colorbar to each subplot
fig.colorbar(im1, ax=ax1, shrink = 0.7)
fig.colorbar(im2, ax=ax2, shrink = 0.7)
fig.colorbar(im3, ax=ax3, shrink = 0.7)

ax1.text(0.1, 0.9, "(a)", ha="center", va="center", color = 'white', transform=ax1.transAxes)
ax2.text(0.1, 0.9, "(b)", ha="center", va="center", color = 'white', transform=ax2.transAxes)
ax3.text(0.1, 0.9, "(c)", ha="center", va="center", color = 'white', transform=ax3.transAxes)


# Show the plot
plt.show()


