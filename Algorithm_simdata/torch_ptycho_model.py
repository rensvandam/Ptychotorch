# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 11:58:17 2022

collection of classes and functions used for Pytorch ptychography simulations

@author: Rens van Dam
         p.s.vandam@students.uu.nl
"""

# %% Import modules

import torch as to
import matplotlib
from matplotlib import pyplot
import numpy as np
import pickle

# %% ScanLayer


class ScanLayer(to.nn.Module):

    def __init__(self, sample, **kwargs):  # **kwargs is a dict of all other input

        # be sure to call this function at the end
        super(ScanLayer, self).__init__(**kwargs)

        # real part
        sample_real = to.real(sample)  # convert precision
        self.sample_real = to.nn.Parameter(
                sample_real,  # take real part
                requires_grad=True) #computes gradient

        # imag part
        sample_imag = to.Tensor.to(to.imag(sample)) # convert precision
        self.sample_imag = to.nn.Parameter(
                sample_imag,  # take imag part
                requires_grad=True) # set trainable

    def forward(self, scan_pos):
        sample = to.complex(self.sample_real, self.sample_imag)  # combine the complex-valued sample



        shift = (int(-scan_pos[0][0]), int(-scan_pos[0][1]))

        # roll: pixelated shift
        sample = to.nn.functional.pad(sample, (300, 300, 300, 300), "constant", 1)

        #scan_pos = scan_pos.astype(int)
        #print(scan_pos)
        #shifted_sample = to.empty((0,0), dtype =to.complex64)
       # shifted_sample = to.roll(sample, shifts=tuple(scan_pos[0]), dims=[0, 1])
        shifted_sample = to.roll(sample, shifts=shift, dims=[0, 1])
        #print(shifted_sample.shape)
        #print(shifted_sample)# I should still delete this extra one.
        for pos in scan_pos:
            pos = tuple(pos)
            to.stack((shifted_sample, to.roll(sample, shifts=tuple(shift), dims=[0, 1])), 0)  # 0:vertical & # 1:horizontal




        return shifted_sample


    # %% InteractLayer


class InteractLayer(to.nn.Module):

    def __init__(self, probe, **kwargs):

        # be sure to call this function at the end
        super(InteractLayer, self).__init__(**kwargs)

        # real part
        probe_real = to.Tensor.to(to.real(probe))  # convert precision
        self.probe_real = to.nn.Parameter(probe_real,   # take real part
                                  requires_grad=False) #computes gradient
        # imag part
        probe_imag = to.Tensor.to(to.imag(probe))  # convert precision
        self.probe_imag = to.nn.Parameter(probe_imag,   # take imag part
                                  requires_grad=False) #computes gradient


    def forward(self, sample):
        probe = to.complex(self.probe_real, self.probe_imag)  # combine the complex-valued probe

        exit_field = to.zeros([256,256])
        sample = [sample]
        for elem in sample:

            exit_field = self.sliced_product(elem, probe)


        return exit_field


    def sliced_product(self, a, b):

        begin_vert = int((list(a.shape)[0] - list(b.shape)[0])/2)
        begin_hori = int((list(a.shape)[0] - list(b.shape)[1])/2)
        sliced_vert = to.narrow(a, 1, begin_vert, int(list(b.shape)[0]))
        c = to.narrow(sliced_vert, 0, begin_hori, int(list(b.shape)[1])) * b

        return c


# %% prop layer


class PropLayer(to.nn.Module):

    def __init__(self, **kwargs):
        # be sure to call this function at the end
        super(PropLayer, self).__init__(**kwargs)

    def forward(self, exit_field):

        diff_pat = to.zeros([256,256])

        exit_field = [exit_field]
        for elem in exit_field:
            diff_pat = to.abs(self.Angular_Spectrum(elem, Bandlimit=None))**2
            #diff_pat = to.abs(self.fft2(elem))**2

            #diff_pat = to.abs(self.fourier_propagator(elem))**2# 0:vertical & # 1:horizontal


        # pyplot.figure(figsize=(10, 3.5))
        # pyplot.subplot(1, 2, 1)
        # pyplot.imshow(diff_pat.cpu().detach().numpy(), cmap='turbo')
        # pyplot.colorbar()
        # pyplot.subplot(1, 2, 2)
        # #pyplot.imshow(diff_pat_list[ind].cpu().detach().numpy(), cmap='turbo')
        # pyplot.show()

        return diff_pat

    def fft2(self, f):

        g = to.fft.fftshift(to.fft.fft2(to.fft.fftshift(f))) / \
            to.sqrt(to.tensor(int((list(f.size())[1]))*int((list(f.size())[0]))))
        return g

    def Angular_Spectrum(self, f, Bandlimit):

        def rect(x):
            return to.where(abs(x)<=0.5, 1, 0)

        z = 200
        z = 200 #mm
        λ = 5.61E-4 #mm
        Nx = 256
        Ny = 256
        pixelsize = 0.00345 #mm

        dx = pixelsize
        dy = pixelsize

        # compute angular spectrum

        #zero padding
        f = to.nn.functional.pad(f, (int((Nx-f.shape[0])/2), int((Nx-f.shape[0])/2), int((Ny-f.shape[1])/2), int((Ny-f.shape[1])/2)), "constant", 0)

        # f = np.pad(f, [(int((Nx-f.shape[0])/2),int((Ny-f.shape[1])/2)), (int((Nx-f.shape[0])/2),int((Ny-f.shape[1])/2))])
        fft_c = to.fft.fft2(f)
        c = to.fft.fftshift(fft_c)

        # transfer function

        u = to.fft.fftshift(to.fft.fftfreq(Nx, d = dx))
        v = to.fft.fftshift(to.fft.fftfreq(Ny, d = dy))
        u, v = to.meshgrid(u, v)


        H = to.exp(1j*(2*np.pi)*z/λ*(to.sqrt(1 - (λ*u) ** 2 - (λ*v) ** 2)))

        #getting rid of evanescent waves
        #mask = np.sqrt((((λ*f_x) ** 2 + (λ*f_y) ** 2))) < 1
        #H = np.where(mask, H,  0)
        #H = to.from_numpy(H)

        #bandlimit the transfer function:

        # dfx = 2*Nx*dx
        # dfy = 2*Ny*dy
        # fx_lim = np.sqrt((2*z/dfx)**2 + 1)*λ
        # fy_lim = np.sqrt((2*z/dfy)**2 + 1)*λ
        # H_BL = H*rect(0.5*fx_lim/f_x)*rect(0.5*fy_lim/f_y)

        S_x = Nx*dx
        S_y = Ny*dy
        du = 1/(2*S_x)
        dv = 1/(2*S_y)

        ulim = 1/(np.sqrt((2*z*du)**2 + 1)*λ)
        vlim = 1/(np.sqrt((2*z*dv)**2 + 1)*λ)

        H_BL = H*rect(0.5*(u/ulim))*rect(0.5*(v/vlim))

        if Bandlimit == "Selfmade":
            H_BL = H*rect(0.5*(u/ulim))*rect(0.5*(v/vlim))
            g = to.fft.ifft2(to.fft.ifftshift(c * H_BL))
        if Bandlimit == None:
            g = to.fft.ifft2(to.fft.ifftshift(c * H))

        return g

# %% model

class AssembleModel(to.nn.Module):
    def __init__(self, sample, probe):

        super(AssembleModel, self).__init__()
        self.scan = ScanLayer(sample)
        self.interact = InteractLayer(probe)
        self.prop = PropLayer()

    def forward(self, scan_pos):

        shifted_sample = self.scan(scan_pos)
       # print(shifted_sample)
        exit_field = self.interact(shifted_sample)

        # with open("exit_field", "wb") as fp:
        #     pickle.dump(exit_field, fp)

        #print(exit_field)
        diff_pat = self.prop(exit_field)
        #print(diff_pat)


        return diff_pat





# %% loss functions

def compute_regularization(image, pixelsize, Nx, Ny, z, C, lam, regularization):

    if regularization == True:

        λ = lam
        dx = pixelsize
        dy = pixelsize
        S_x = Nx*dx
        S_y = Ny*dy
        du = 1/(2*S_x)
        dv = 1/(2*S_y)
        ulim = 1/( to.sqrt( to.tensor( (2*z*du)**2 +1 ) )*λ )
        vlim = 1/( to.sqrt( to.tensor( (2*z*dv)**2 +1 ) )*λ )
        u = to.fft.fftshift(to.fft.fftfreq(Nx, d = dx))
        v = to.fft.fftshift(to.fft.fftfreq(Ny, d = dy))
        u, v = to.meshgrid(u, v)
        mask = to.where((abs(0.5*(u/ulim))<=0.5) & (abs(0.5*(v/vlim)) <= 0.5), 0, 1)
        image_fft = to.fft.fftshift(to.fft.fft2(image))
        regul = to.abs(to.fft.ifft2(image_fft  * mask ))

    else:
        regul = to.zeros((int(Nx),int(Ny)))

    regul = C*to.sum(regul**2)


    return regul


def compute_regularization2(image, C):


        Nx = 256
        Ny = 256
        pixelsize = 0.005
        dx = pixelsize
        dy = pixelsize
        λ = 6E-4
        z = 200
        def rect(x):
            return to.where(abs(x)<=0.5, 1, 0)

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

        regul = image - image2



        return regul


def loss_func_poisson(meas, pred, sigma2, regul):
   # meas += to.abs(to.min(meas))
   # pred += to.abs(to.min(pred))
#    meas = to.from_numpy(np.array(meas))
  #  print(to.sqrt(meas) - to.sqrt(pred))
    #print(to.sq)
    #print((to.sqrt(meas)-to.sqrt(pred))**2)

    #normal
    loss = to.sum((to.sqrt(meas)-to.sqrt(pred))**2 + regul)
    #sigma2 = 1000
    #with distributions for pred
    #loss = to.sum((to.sqrt(meas)-to.sqrt(pred))**2 + 0.5*(pred**2/sigma2))
   # loss = to.sum((to.sqrt(meas)-to.sqrt(pred))**2 + to.log(to.lgamma(pred+1)))
    #loss = to.sum((to.sqrt(meas)-to.sqrt(pred))**2 - (1/100)*(to.log(pred) - (to.log(pred)**2)/(2*sigma2)))



    #loss = to.sum((pred - meas*to.log(pred)))
    return loss

def loss_func_gauss(meas, pred):

    loss = to.sum((meas - pred)**2)

    return loss

def loss_func_sum(meas, pred):

    loss = 0.5 * ( to.sum((to.sqrt(meas)-to.sqrt(pred))**2) ) + 0.5 * ( to.sum((meas - pred)**2) )

    return loss

def loss_func_mixed(meas, pred, sigma2, regul):


    #loss = to.sum( ((meas - pred)**2) / (a*pred + b + sigma2) + to.log(a*pred + b + sigma2) )
#    loss = to.sum( ((meas - pred)**2) / (2*(pred + sigma2)) + 0.5*to.log(pred + sigma2))
    loss = to.sum( ((meas - pred)**2) / ((pred + sigma2)) + to.log(pred + sigma2)) + regul
    #loss = regul
    #loss = to.sum( ((meas - pred)**2) / ((pred + sigma2)) + to.log(pred + 1))

    return loss
# %% Noise function

def noisy(noise_type,image, sigma2, loss):

     if noise_type == "gauss":
         row,col= image.shape
         #print(row,col)
         mean = 0
         var = sigma2 #1
         sigma = var**0.5
         gauss = np.random.normal(mean,sigma,(row,col))
         gauss = gauss.reshape(row,col)
         noisy = image + gauss


         if loss == 'poisson':
             noisy = noisy.ravel()
             for i in range(len(noisy.ravel())):
                          if noisy[i] < 0:
                              noisy[i] = 0
             noisy = noisy.reshape(image.shape[0],image.shape[0])

         if loss == 'mixed':
             noisy = noisy.ravel()
             for i in range(len(noisy.ravel())):
                          if noisy[i] == None:
                              noisy[i] = 0
             noisy = noisy.reshape(image.shape[0],image.shape[0])


         return noisy
     elif noise_type == "poisson":
          noisy = np.random.poisson(image)
          return noisy


#%% Parameters class

