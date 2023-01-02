# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 11:58:17 2022

@author: rens van dam
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
        # roll: pixelated shift
        #scan_pos = (scan_pos + 300).astype(int)
        #print(scan_pos)
        #shifted_sample = to.empty((0,0), dtype =to.complex64)

        shift = tuple((-int(scan_pos[0][0]), -int(scan_pos[0][1])))

        # pyplot.figure(figsize=(5, 3.5))
        # pyplot.imshow(np.abs(sample.detach().numpy()), cmap='gray')
        # pyplot.colorbar()
        # pyplot.show()

        sample = to.nn.functional.pad(sample, (300, 300, 300, 300), "constant", 1)
        shifted_sample = to.roll(sample, shifts=shift, dims=[0, 1])
        # pyplot.figure(figsize=(5, 3.5))
        # pyplot.imshow(np.abs(shifted_sample.detach().numpy()), cmap='gray')
        # pyplot.colorbar()
        # pyplot.show()

        #shifted_sample = shifted_sample[300:end+300, 300:end+300]

        # print(shifted_sample.shape)
        #print(shifted_sample.shape)
        #print(shifted_sample)# I should still delete this extra one.
        # for pos in scan_pos:
        #     pos = tuple(-pos)
        #     to.stack((shifted_sample, to.roll(sample, shifts=pos, dims=[0, 1])), 0)



        # 0:vertical & # 1:horizontal
        #print(shifted_sample.shape)
        #print(shifted_sample.shape)
        # pyplot.figure(figsize=(5, 3.5))
        # pyplot.imshow(np.abs(shifted_sample.detach().numpy()), cmap='gray')
        # pyplot.colorbar()
        # pyplot.show()


        return shifted_sample


    # %% InteractLayer


class InteractLayer(to.nn.Module):

    def __init__(self, probe, **kwargs):

        # be sure to call this function at the end
        super(InteractLayer, self).__init__(**kwargs)

        # real part
        probe_real = to.Tensor.to(to.real(probe))  # convert precision
        self.probe_real = to.nn.Parameter(probe_real,   # take real part
                                  requires_grad=True) #computes gradient
        # imag part
        probe_imag = to.Tensor.to(to.imag(probe))  # convert precision
        self.probe_imag = to.nn.Parameter(probe_imag,   # take imag part
                                  requires_grad=True) #computes gradient


    def forward(self, sample, pixelsize, Nx, Ny, C, z, lam):
        probe = to.complex(self.probe_real, self.probe_imag)  # combine the complex-valued probe

        # interact (multiplicative assumption)

        sample = [sample]
        for elem in sample:

            exit_field = self.sliced_product(elem, probe, pixelsize, Nx, Ny, C, z, lam)

        return exit_field


    def sliced_product(self, a, b, pixelsize, Nx, Ny, C, z, lam):


        begin_vert = int((list(a.shape)[0] - list(b.shape)[0])/2)
        begin_hori = int((list(a.shape)[0] - list(b.shape)[1])/2)
        sliced_vert = to.narrow(a, 1, begin_vert, int(list(b.shape)[0]))

        q = to.narrow(sliced_vert, 0, begin_hori, int(list(b.shape)[1]))
        c = q * b

        return c


# %% prop layer


class PropLayer(to.nn.Module):

    def __init__(self, **kwargs):
        # be sure to call this function at the end
        super(PropLayer, self).__init__(**kwargs)

    def forward(self, exit_field, probesize, Nx, Ny, pixelsize, z, lam, BL):

        diff_pat = to.zeros(probesize)

        exit_field = [exit_field]
        for elem in exit_field:

            #diff_pat = to.abs(self.Angular_Spectrum(elem, Nx, Ny, pixelsize, z, lam, Bandlimit="Selfmade"))**2

            diff_pat = self.Angular_Spectrum(elem, Nx, Ny, pixelsize, z, lam, BL=BL)


            #diff_pat = to.abs(self.fft2(elem))**2

            #diff_pat = to.abs(self.fourier_propagator(elem))**2# 0:vertical & # 1:horizontal

        return diff_pat

    def fft2(self, f):

        g = to.fft.fftshift(to.fft.fft2(to.fft.fftshift(f))) / \
            to.sqrt(to.tensor(int((list(f.size())[1]))*int((list(f.size())[0]))))

        return g

    def Angular_Spectrum(self, f, Nx, Ny, pixelsize, z, lam, BL):

        def rect(x):
            return to.where(abs(x)<=0.5, 1, 0)

        λ = lam #mm

        dx = pixelsize
        dy = pixelsize

        # compute angular spectrum

        #zero padding
        # f = to.nn.functional.pad(f, (int((Nx-f.shape[0])/2), int((Nx-f.shape[0])/2), int((Ny-f.shape[1])/2), int((Ny-f.shape[1])/2)), "constant", 0)

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

        S_x = Nx*dx
        S_y = Ny*dy
        du = 1/(2*S_x)
        dv = 1/(2*S_y)

        ulim = 1/( to.sqrt( to.tensor( (2*z*du)**2 +1 ) )*λ )
        vlim = 1/( to.sqrt( to.tensor( (2*z*dv)**2 +1 ) )*λ )

        H_BL = H*rect(0.5*(u/ulim))*rect(0.5*(v/vlim))


        if BL == "Selfmade":
            H_BL = H*rect(0.5*(u/ulim))*rect(0.5*(v/vlim))
            g = to.fft.ifft2(to.fft.ifftshift(c * H_BL))
        if BL == None:
            g = to.fft.ifft2(to.fft.ifftshift(c * H))

        return g

# %% model

class AssembleModel(to.nn.Module):
    def __init__(self, sample, probe):

        super(AssembleModel, self).__init__()
        self.scan = ScanLayer(sample)
        self.interact = InteractLayer(probe)
        self.prop = PropLayer()

    def forward(self, scan_pos, probesize, Nx, Ny, pixelsize, z, lam, C, BL):

        shifted_sample = self.scan(scan_pos)
       # print(shifted_sample)
        exit_field = self.interact(shifted_sample, pixelsize, Nx, Ny, C, z, lam)


        # with open("exit_field", "wb") as fp:
        #     pickle.dump(exit_field, fp)


        #print(exit_field)
        diff_pat = self.prop(exit_field, probesize, Nx, Ny, pixelsize, z, lam, BL)
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


        # print(to.sum(to.abs(mask)))
        # pyplot.figure(figsize=(5, 3.5))
        # pyplot.imshow(to.abs(mask).cpu().detach().numpy(), cmap='gray')
        # pyplot.colorbar()
        # pyplot.show()

        regul = C*to.abs(to.fft.ifftshift(image_fft * mask))


        # pyplot.figure(figsize=(5, 3.5))
        # pyplot.imshow(to.abs(image).cpu().detach().numpy(), cmap='gray')
        # pyplot.colorbar()
        # pyplot.show()


    else:
        regul = to.zeros((Nx,Ny))


    return regul



def loss_func_poisson(meas, pred, device, regul, sigma2):
   # meas += to.abs(to.min(meas))
   # pred += to.abs(to.min(pred))
#    meas = to.from_numpy(np.array(meas))
  #  print(to.sqrt(meas) - to.sqrt(pred))
    #print(to.sq)
    #print((to.sqrt(meas)-to.sqrt(pred))**2)
    loss = to.sum((to.sqrt(meas)-to.sqrt(pred))**2 + regul)

    #loss = to.sum((pred - meas*to.log(pred)))
    return loss

def loss_func_gauss(meas, pred):

    loss = to.sum((meas - pred)**2)

    return loss

def loss_func_sum(meas, pred):

    loss = 0.5 * ( to.sum((to.sqrt(meas)-to.sqrt(pred))**2) ) + 0.5 * ( to.sum((meas - pred)**2) )

    return loss

def loss_func_mixed(meas, pred, device, regul, sigma2):

    loss = to.sum( ((meas - pred)**2) / ((pred + sigma2)) + to.log(pred + sigma2)) + to.sum(to.abs(regul))

    #loss = to.sum( ((to.sqrt(meas) - to.sqrt(pred))**2) / ((pred + sigma2)) + to.log(pred + sigma2)) # DEBUG WEIRD LOSS TEST

    #loss = to.sum( ((meas**0.7 - pred**0.7)**2) / ((pred + sigma2)) + to.log(pred + sigma2))



    return loss
# %% Noise function

def noisy(noise_type,image):
     if noise_type == "gauss":
         row,col= image.shape
         #print(row,col)
         mean = 0
         var = 1 #1
         sigma = var**0.5
         gauss = np.random.normal(mean,sigma,(row,col))
         gauss = gauss.reshape(row,col)
         noisy = image + gauss
         noisy = noisy.ravel()
         noisy[noisy<0] = 0
         noisy = noisy.reshape(image.shape[0],image.shape[0])
         return noisy
     elif noise_type == "poisson":
          noisy = np.random.poisson(image)
#          noisy = image + noise_mask
          return noisy
     elif noise_type =="speckle":
          row,col = image.shape
          gauss = np.random.randn(row,col)
          gauss = gauss.reshape(row,col)
          noisy = image+  image * gauss
          return noisy