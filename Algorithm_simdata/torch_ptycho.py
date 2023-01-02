# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 11:52:42 2022

Main file for ptychography simulations using Pytorch


@author: Rens van Dam
         p.s.vandam@students.uu.nl
"""


# %% import modules
import pickle
import torch as to
import numpy as np
import math
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot
from matplotlib.patches import Circle
#from matplotlib_scalebar import Scalebar
from matplotlib_scalebar.scalebar import ScaleBar

from IPython import get_ipython
from torch_ptycho_model import loss_func_poisson #best lr: 0.01
from torch_ptycho_model import loss_func_gauss   # best lr: 0.01
from torch_ptycho_model import loss_func_sum   # best lr: 0.005
from torch_ptycho_model import loss_func_mixed # best lr: 0.006
from torch_ptycho_model import compute_regularization


device = to.device("cuda") if to.cuda.is_available() else to.device("cpu")
to.set_default_tensor_type('torch.cuda.FloatTensor')

#get_ipython().run_line_magic('matplotlib', 'qt') #inline for inline or qt for seperate window


# %% Parameters

#Image details

sample_pixel_num = (512, 512)
sample_pixel_size = (1, 1)
probe_pixel_num = (256, 256)
probe_pixel_size = (1, 1)
#probe_amplitude = [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90]
#probe_amplitude = [20,20,20,20,20,20,20,20,20,20] #20 is normal
probe_amplitude = [20]
pixelsize= 0.00345
Nx = 256
Ny = 256

var =  [1]
#var = [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]

#optimization details
optimization = 'ADAM'   # choose 'ADAM' or 'SGD'
loss = 'mixed'     # choose 'mixed' or 'poisson'
iteration = 50    # amount of iterations (epochs)
lr = 0.04            #
regularization = False
C = 1000000000000000 #1E15
z = 200
lam = 6E-4
driftcorrection = 'off'

#statistics
Ncor = 20 #= the last Ncor correlation values to calculate stdev of mean on for every noise sigma2

parameters = {}
parameters['sample_pixel_num'] = sample_pixel_num
parameters['sample_pixel_size'] = sample_pixel_size
parameters['probe_pixel_num'] = probe_pixel_num
parameters['probe_pixel_size'] = probe_pixel_size
parameters['probe_amplitude'] = probe_amplitude
parameters['variance'] = var
parameters['Nx'] = Nx
parameters['Ny'] = Ny

parameters['optimization'] = optimization
parameters['loss'] = loss
parameters['iteration'] = iteration
parameters['lr'] = lr
parameters['regularization'] = regularization
parameters['driftcorrection'] = driftcorrection
parameters['C'] = C
parameters['z'] = z
parameters['lam'] = lam
parameters['Ncor'] = Ncor




#%% The work

if len(probe_amplitude) > 1:
    indepvariable = probe_amplitude

if len(var) > 1:
    indepvariable = var

if len(probe_amplitude) == 1 and len(var) == 1:
    indepvariable = [1]

parameters['probe_totalintensity'] = [0]*len(indepvariable)

complexcorrelation = [[0,0,0]]*len(indepvariable)
recon = [0]*len(indepvariable)
print(recon)
print(indepvariable)

for i in range(len(indepvariable)):
    print("I am starting with iteration {} of {}, please wait a little longer!".format(i+1, len(indepvariable)))
    if indepvariable == var:
        sigma2 = var[i]
        probe_amp = probe_amplitude[0]

    if indepvariable == probe_amplitude:
        probe_amp = probe_amplitude[i]
        sigma2 = var[0]

    if indepvariable == [1]:
        probe_amp = probe_amplitude[0]
        sigma2 = var[0]


    if loss == "mixed":
        loss_func = loss_func_mixed
    if loss == "poisson":
        loss_func = loss_func_poisson

# %% generate sample

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
            np.exp(1j * math.pi * (0.1 + 0.9 * pha)))
        #sample = normalized(sample)
        #sample /= np.max(sample) #make sure the real part of the sample (the intensity) does not exceed 1
        sample = to.from_numpy(sample)

        if mask == True:
            amp = masked(amp)
            pha = masked(pha)
            sample = np.multiply(
                (0.25 + 0.75 * amp),
                np.exp(1j * math.pi * (0.1 + 0.9 * pha)))
            sample = to.from_numpy(sample)
            #sample = masked(sample)

        if plot == True:
            image([np.abs(sample), np.angle(sample)], 2, "grid", color="gray")

        return sample

    sample = generate_sample(mask=True, plot=False).to(device)

    # with open("sample", "wb") as fp:
    #     pickle.dump(sample, fp)



# %% generate probe

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
            parameters['probe_totalintensity'][i] = probeint
        if plot == True:
            image([np.abs(probe), np.angle(probe)], 2, title="Probe", color='jet')

        return probe

    probe = generate_probe('gaussian', probe_pixel_num, probe_pixel_size, probe_amp, plot=False).to(device)


# %% generate scan_pos

# v = np.arange(-100.0, 120.0, 20)
# u = np.arange(-100.0, 120.0, 20)
# vv, uu = np.meshgrid(v, u)

    def generate_scanpos(sample, method, plot, shuffle):
        if method == "grid":
            v = np.arange(-100.0, 120.0, 20)
            u = np.arange(-100.0, 120.0, 20)
            vv, uu = np.meshgrid(v, u)
            scan_pos_list = np.transpose([vv.flatten(), uu.flatten()])
            scan_pos_list = np.reshape(scan_pos_list, newshape=(len(scan_pos_list), 2))

        if method == "sunflower":

            num_pts = 118
            indices = np.arange(0, num_pts, dtype=float) + 0.5
            r = np.sqrt(indices/num_pts)
            theta = np.pi * (1 + 5**0.5) * indices
            scan_pos_list = [0]*num_pts
            for i in range(len(scan_pos_list)):
                scan_pos_list[i] = [115*r[i]*np.cos(theta[i]), 115*r[i]*np.sin(theta[i])]
            scan_pos_list = np.reshape(scan_pos_list, newshape=(len(scan_pos_list), 2))

        if shuffle == True:

            def shuffle(list1):
                indices = np.arange(list1.shape[0])
                np.random.shuffle(indices)
                list1 = list1[indices]
                return list1

            shuffled = shuffle(scan_pos_list)
            scan_pos_list = shuffled

        if plot == True:
            print(scan_pos_list[1,0])
            pyplot.figure(figsize=(6, 5))
            pyplot.rc('font', size=15)
            pyplot.rc('axes', labelsize=15) #fontsize of the x and y labels
            pyplot.plot(scan_pos_list[:, 1], scan_pos_list[:, 0], 'o')
            for x,y in scan_pos_list:
                circ = Circle((y,x),30, alpha=0.05)
                pyplot.gca().add_patch(circ)
            pyplot.xlabel(r"$x$ [pixels]")
            pyplot.ylabel(r"$y$ [pixels]")
            pyplot.tight_layout()
            pyplot.show()

        return scan_pos_list

    scan_pos_list = generate_scanpos(sample, method="sunflower", plot=False, shuffle = False)
    scan_pos_list = to.from_numpy(scan_pos_list).to(device)

    # fig,axs = pyplot.subplots(1,2)
    # pyplot.rc('font', size=19)
    # pyplot.rc('axes', labelsize=19) #fontsize of the x and y labels
    # fig.set_figwidth(10)
    # fig.set_figheight(5)

    # for i,ax in enumerate(axs.flat, start=97):
    #     ax.text(0.02,0.88,'('+chr(i)+')',fontsize=20, color='black', transform=ax.transAxes)

    # scan_pos_list = generate_scanpos(sample, method="grid", plot=False, shuffle = False)

    # axs[0].plot(scan_pos_list[:, 1], scan_pos_list[:, 0], 'o')
    # for x,y in scan_pos_list:
    #     circ = Circle((y,x),30, alpha=0.05)
    #     axs[0].add_patch(circ)
    # axs[0].set_xlabel(r"$x$ [pixels]")
    # axs[0].set_ylabel(r"$y$ [pixels]")

    # scan_pos_list = generate_scanpos(sample, method="sunflower", plot=False, shuffle = True)

    # axs[1].plot(scan_pos_list[:, 1], scan_pos_list[:, 0], 'o')
    # for x,y in scan_pos_list:
    #     circ = Circle((y,x),30, alpha=0.05)
    #     axs[1].add_patch(circ)
    # axs[1].set_xlabel(r"$x$ [pixels]")
    # pyplot.tight_layout()


# %% simulate diffraction patterns

    from torch_ptycho_model import AssembleModel
    model = AssembleModel(sample, probe)
    print(model)

    def simulate_diff_pat(scanpositions, sigma2, mask=False, noise=False, plot=False, bandlimit=False):
        diff_pat_list = []
        for ind in range(len(scan_pos_list)):
            diff_pat_list.append(model(scan_pos_list[[ind]]).cpu().detach().numpy())

        if mask==True:

            def create_circular_mask(h, w, center=None, radius=None):
                if center is None: # use the middle of the image
                    center = (int(w/2), int(h/2))
                if radius is None: # use the smallest distance between the center and image walls
                    radius = min(center[0], center[1], w-center[0], h-center[1])
                Y, X = np.ogrid[:h, :w]
                dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
                mask = dist_from_center <= radius
                return mask
            mask = create_circular_mask(256,256, radius=20)
            mask = 1-mask
            for i in range(len(diff_pat_list)):
                #diff_pat_list[i] = np.abs(np.fft.fftshift(np.fft.fft2(diff_pat_list[i])))*mask
                #diff_pat_list[i] = np.abs(np.fft.ifftshift(np.fft.ifft2(diff_pat_list[i])))
                diff_pat_list[i] = mask*diff_pat_list[i]
            #image([mask], 1, "mask")
            #image([1-mask], 1, "1-mask")
        diffpatsize = diff_pat_list[0].shape[0]
        diff_pat_list = np.array(diff_pat_list)
        diff_pat_list = diff_pat_list.reshape(len(scan_pos_list),diffpatsize,diffpatsize)


        if bandlimit == True:
            def rect(x):
                return to.where(abs(x)<=0.5, 1, 0)

            z = 200  #mm
            λ = 6E-4 #mm
            Nx = 256
            Ny = 256
            pixelsize = 0.00345 #mm

            dx = pixelsize
            dy = pixelsize

            u = to.fft.fftshift(to.fft.fftfreq(Nx, d = dx))
            v = to.fft.fftshift(to.fft.fftfreq(Ny, d = dy))
            u, v = to.meshgrid(u, v)

            S_x = Nx*dx
            S_y = Ny*dy
            du = 1/(2*S_x)
            dv = 1/(2*S_y)

            ulim = 1/(np.sqrt((2*z*du)**2 + 1)*λ)
            vlim = 1/(np.sqrt((2*z*dv)**2 + 1)*λ)

            bandlimiter = rect(0.1*(u/ulim))*rect(0.1*(v/vlim))

            for i in range(len(diff_pat_list)):
                diff_pat_list[i] = bandlimiter.cpu()*diff_pat_list[i]

        if noise==True:

            from torch_ptycho_model import noisy
            for i in range(len(diff_pat_list)):
                diff_pat_list[i] = noisy("poisson", diff_pat_list[i], sigma2, parameters["loss"])
                diff_pat_list[i] = noisy("gauss", diff_pat_list[i], sigma2, parameters["loss"])
                diff_pat_list[i] = diff_pat_list[i]

        if plot==True:


            #totalintensity = np.abs(np.sum(diff_pat_list[4]))
            image([diff_pat_list[4]], 1, "diffraction pattern #1 Pytorch")

            pyplot.figure(figsize=(8,5))
            pyplot.plot(diff_pat_list[4][128][0:256])
            pyplot.xlabel(r"$x$ (pixels)")
            pyplot.ylabel(r"$|\widehat{\psi(\theta})|^2$ (intensity)")
            pyplot.show()


            with open("diffpat_noBL", "wb") as fp:
                pickle.dump(diff_pat_list[4], fp)

        return diff_pat_list

    diff_pat_list = simulate_diff_pat(scan_pos_list, sigma2, mask=False, noise=False, plot=False, bandlimit = False)


    snr = [0]*118
    for j in range(len(snr)):
        snr[j] = np.mean(diff_pat_list[j])/np.std(diff_pat_list[j])
    print("SNR:", np.mean(snr))

    diff_pat_list = to.from_numpy(diff_pat_list).to(device)
#back = np.fft.ifftshift(np.fft.ifft2(diff_pat_list[4]))



#image([np.abs(back)], 1, "backpropagated diffpat #4")

# %% sample guess

    def create_sample_guess(sample, plot=False):
        sample = sample.cpu().detach().numpy()
        #sample_init = np.ones_like(sample)
        sample_init = np.ones((2*sample.shape[0], 2*sample.shape[1])) #DEBUG
        #sample_init = masked(sample_init)
        sample_init = to.from_numpy(sample_init)
        sample_init = sample_init.to(to.complex128)

        if plot == True:
            image([np.abs(sample_init), np.angle(sample_init)], 2, "sample initializer", color="gray")

        return sample_init

    sample_init = create_sample_guess(sample, plot=False).to(device)

# %% probe guess

    def create_probe_guess(probe, probesize, pixelsize, plot=False):
        #x,y width
        wy = 16
        wx = 16
        #x,y pixel number
        ny = probesize[0]
        nx = probesize[1]
        #x,y pixel size
        dy = pixelsize[0]
        dx = pixelsize[1]
        y = np.arange(-ny / 2, ny / 2) * dy
        x = np.arange(-nx / 2, nx / 2) * dx
        xx, yy = np.meshgrid(x,y)

        probe = probe.cpu().detach().numpy()
        probe_init = np.ones_like(probe)
        probe_init[np.sqrt((yy / wy / 2) ** 2 + (xx / wx / 2) ** 2) > 1] = 0
        probe_init = to.from_numpy(probe_init)

        if plot == True:
            image([np.abs(probe_init), np.angle(probe_init)], 2, "probe initializer", color="gray")


        return probe_init

    #probe_init = create_probe_guess(probe, probe_pixel_num, probe_pixel_size, plot=False).to(device)
    probe_init = probe

# %% Initializing model

    model = AssembleModel(sample_init, probe_init)

    model(scan_pos_list[[0]])

    with open("exit_field", "rb") as fp:
        exit_field = pickle.load(fp)

    regul=compute_regularization(exit_field, pixelsize, Nx, Ny, z, C, lam, regularization)

    # pyplot.figure()
    # pyplot.imshow(regul.cpu().detach().numpy())
    # pyplot.show()



    loss = loss_func(diff_pat_list[0], model(scan_pos_list[[0]]), sigma2, regul)
    print('loss function used:' + str(loss_func))
    print('loss: ' + str(loss.cpu().detach().numpy()))

    if optimization == 'ADAM':
        print("Using ADAM optimization.")
        optimizer = to.optim.Adam(model.parameters(), lr=lr)
        #scheduler = to.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)

    if optimization == 'SGD':
        print("Using SGD optimization.")
        parameters["lr"] = 0.0001
        optimizer = to.optim.SGD(model.parameters(), lr=parameters["lr"])
        scheduler = to.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)

#%% Some functions for plotting and correlation

    #PLOTTING ---------------------------
    if len(indepvariable) == 1:

        pyplot.ion()

        fig = pyplot.figure(figsize=(10,10))
        X = [ (0, 3,2,1), (1, 3,2,2), (2, 3,2,3), (3, 3,2,4), (4, 3,2,(5,6))]
        axs = [0]*5
        for plot, nrows, ncols, plot_number in X:
            axs[plot] = pyplot.subplot(nrows, ncols, plot_number, adjustable='box')

        #axs =  = pyplot.subplot(2, 3, figsize=(10, 7), gridspec_kw=gs_kw)
        # sample = to.complex(
        #     model.scan.sample_real,
        #     model.scan.sample_imag)
            #model.get_layer('ScanLayer').sample_real,
            #model.get_layer('ScanLayer').sample_imag)
        axs[0].imshow(np.abs(sample.cpu().detach().numpy()), cmap='turbo')
        axs[1].imshow(np.angle(sample.cpu().detach().numpy()), cmap='twilight')
        probe = to.complex(
            model.interact.probe_real,
            model.interact.probe_imag)

            #model.get_layer('InteractLayer').probe_real,
            #model.get_layer('InteractLayer').probe_imag)
        axs[2].imshow(np.abs(probe.cpu().detach().numpy()), cmap='turbo')
        axs[3].imshow(np.angle(probe.cpu().detach().numpy()), cmap='twilight')


        axs[4].set_xlabel("Epoch")
        axs[4].set_ylabel("Loss")
        fig.suptitle('loss: ' + str(loss.cpu().detach().numpy()))
        pyplot.pause(0.01)

    #-------------------------------------

    losses = np.zeros(iteration)
    regularizations = np.zeros(iteration)
    correlations = np.zeros(iteration, dtype='complex')

    def sliced(a, b):
        #print(a.shape)
        #print(b.shape)
        begin_vert = int((list(a.shape)[0] - list(b.shape)[0])/2)
        begin_hori = int((list(a.shape)[0] - list(b.shape)[1])/2)
        sliced_vert = to.narrow(a, 1, begin_vert, int(list(b.shape)[0]))
        c = to.narrow(sliced_vert, 0, begin_hori, int(list(b.shape)[1]))
    #    sliced_vert = to.narrow(a, 1, 115, 256)
    #    c = to.narrow(sliced_vert, 0, 115, 256)
        #interaction (multiplicative assumption)
        #print(c)
        return c

    def slice_for_correlation(a, correlation_size):

        begin_point = int(0.5*(list(a.shape)[0] - correlation_size))
        sliced_vert = to.narrow(a,1, begin_point , correlation_size)
        c = to.narrow(sliced_vert, 0, begin_point, correlation_size)
        return c

    def calc_correlation(a,b):
        #a /= np.max(a) ABS
        #b /= np.max(b)
        a = to.from_numpy(a)
        b = to.from_numpy(b)

        corr = to.sum(to.conj(a)*b)/\
            (to.sqrt(to.sum(to.abs(b)**2))*to.sqrt(to.sum(to.abs(a)**2)))

        # pyplot.figure(figsize=(10, 3.5))
        # pyplot.subplot(1, 2, 1)
        # pyplot.imshow(np.angle(a.detach().numpy()), cmap='turbo')
        # pyplot.colorbar()
        # pyplot.subplot(1, 2, 2)
        # pyplot.imshow(np.angle(b.detach().numpy()), cmap='turbo')
        # pyplot.title(str(abs(corr)))
        # pyplot.show()

        return corr

    #sample = generate_sample(mask=True)

    # with open("sample", "wb") as fp:
    #     pickle.dump(sliced(sample, probe), fp)

#%% Optimizing
    for epoch in range(iteration):

        avgterm = 1/(len(scan_pos_list))
        regulepoch= [0]*len(scan_pos_list)
        lossepoch = [0]*len(scan_pos_list)

        for ind in range(len(scan_pos_list)):
            ###################### the only important part ######################

            with open("exit_field", "rb") as fp:
                exit_field = pickle.load(fp)

            reg = compute_regularization(exit_field, pixelsize, Nx, Ny, z, C, lam, regularization)

            loss = loss_func(diff_pat_list[ind], model(scan_pos_list[[ind]]), sigma2, reg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ###################### the only important part ######################

            if epoch in [3,7,12, 16]:
                if ind == 4:

                    sample_recon = to.complex(
                        model.scan.sample_real,
                        model.scan.sample_imag)

                    pyplot.figure(figsize=(10, 3.5))
                    pyplot.subplot(1, 2, 1)
                    pyplot.imshow(np.abs(sample_recon.cpu().detach().numpy()[412:612 , 412:612] ), cmap='gray')
                    pyplot.colorbar()
                    pyplot.subplot(1, 2, 2)
                    pyplot.imshow(np.angle(sample_recon.cpu().detach().numpy()[412:612 , 412:612] ), cmap='gray')
                    pyplot.colorbar()
                    pyplot.show()


            lossepoch[ind] = loss.cpu().detach().numpy()
            #print(lossepoch[ind])
            regulepoch[ind] = reg.cpu().detach().numpy()

        # if epoch == range(iteration)[-1]:
        #     if ind == range(len(scan_pos_list))[-1]:
        #         with open("diffpat_recon", "wb") as fp:
        #             pickle.dump(model(scan_pos_list[[-1]]), fp)


        losses[epoch] = avgterm*sum(lossepoch)
        regularizations[epoch] = avgterm*np.sum(regulepoch)


        # Calculating correlation for this epoch
        sample_recon = to.complex(
            model.scan.sample_real,
            model.scan.sample_imag)

        sample = slice_for_correlation(sample, 150)
        sample_recon = slice_for_correlation(sample_recon, 150)

        sample = sample.cpu().detach().numpy()
        sample_recon = sample_recon.cpu().detach().numpy()

        # pyplot.figure(figsize=(10, 3.5))
        # pyplot.subplot(1, 2, 1)
        # pyplot.imshow(np.abs(sample), cmap='gray')
        # pyplot.colorbar()
        # pyplot.subplot(1, 2, 2)
        # pyplot.imshow(np.abs(sample_recon), cmap='gray')
        # pyplot.show()



        correlations[epoch] = calc_correlation(sample, sample_recon).cpu().detach().numpy()
        sample = to.from_numpy(sample)

        # PLOTTING -----------------------------------------------------------

        # scalebar = ScaleBar(0.005, units="mm") # 1 pixel = 5 μm
        print(
            'epoch (' + str(epoch + 1) + '/' + str(iteration) + ') ' +
            'loss: ' + str(losses[epoch]- regularizations[epoch]) +
            ' correlation: ' + str(abs(correlations[epoch])))
        sample_recon = to.complex(
            model.scan.sample_real,
            model.scan.sample_imag)
        sample_recon_cropped = sample_recon[412:612 , 412:612]   #[165:340, 165:340]

        probe_recon = to.complex(
            model.interact.probe_real,
            model.interact.probe_imag)

        if len(indepvariable) == 1:

            axs[0].clear()
            axs[1].clear()
            axs[2].clear()
            axs[3].clear()
            axs[4].clear()

            axs[0].imshow(np.abs(sample_recon_cropped.cpu().detach().numpy()), cmap='gray')
            scalebar = ScaleBar(pixelsize, units="um")
            axs[0].add_artist(scalebar)

            #axs[1].imshow(np.angle(sample_recon[150:350, 150:350].cpu().detach().numpy()), cmap='gray')
            axs[1].imshow(np.angle(sample_recon_cropped.cpu().detach().numpy()), cmap='gray')
            scalebar = ScaleBar(pixelsize, units="um")
            axs[1].add_artist(scalebar)


            axs[2].imshow(np.abs(probe_recon.cpu().detach().numpy()), cmap='gray')
            scalebar = ScaleBar(pixelsize, units="um")
            axs[2].add_artist(scalebar)

            axs[3].imshow(np.angle(probe_recon.cpu().detach().numpy()), cmap='gray')
            scalebar = ScaleBar(pixelsize, units="um")
            axs[3].add_artist(scalebar)


            axs[4].plot(range(len(losses[0:epoch])), losses[0:epoch], label="Total")
            axs[4].plot( range(len(regularizations[0:epoch])), regularizations[0:epoch], label="Regularization")
            axs[4].plot(range(len(losses[0:epoch])), losses[0:epoch] - regularizations[0:epoch], label = "Loss")
            # axs[4].plot(range(len(losses[0:epoch+1])), losses[0:epoch+1])
            # axs[4].plot(range(len(regularizationterms[0:epoch+1])), regularizationterms[0:epoch+1])
            axs[4].legend()
            axs[4].set_xlim([0, len(losses)])
            axs[4].set_ylim([0, max(losses)])
            axs[4].set_xlabel("Epoch")
            axs[4].set_ylabel("Loss")

            fig.suptitle(
                'epoch (' + str(epoch + 1) + '/' + str(iteration) + ') ' +
                'loss: ' + str(losses[epoch]))

            pyplot.pause(0.01)
        #---------------------------------------------------------------------

#%% Plotting conclusion

    sample = slice_for_correlation(sample, 150).cpu().detach().numpy()
    sample_recon = slice_for_correlation(sample_recon, 150).cpu().detach().numpy()



    if parameters["driftcorrection"]  == "on":
        from postprocessing import drift_correction
        if parameters["optimization"] == "SGD":
            print("drift correcting")
            if probe_amp < 65: #this bit of code ensures that the reconstruction continues when it diverges for
                sample, sample_recon = drift_correction(sample, sample_recon)

        correlation_driftcor = calc_correlation(sample, sample_recon)

        sample = to.from_numpy(sample)
        sample_recon = to.from_numpy(sample_recon)

        reconstruction = sample_recon.cpu().detach().numpy()
        errorbar = np.std(correlations[-Ncor:], ddof = 1) / np.sqrt(len(correlations[-Ncor:]))

        recon[i] = reconstruction
        complexcorrelation[i] = [abs(correlation_driftcor), errorbar, indepvariable[i]]


    if parameters["driftcorrection"] == "off":


        correlation = calc_correlation(sample, sample_recon)

        sample = to.from_numpy(sample)
        sample_recon = to.from_numpy(sample_recon)

        reconstruction = sample_recon.cpu().detach().numpy()
        errorbar = np.std(correlations[-Ncor:], ddof = 1) / np.sqrt(len(correlations[-Ncor:]))
        recon[i] = reconstruction
        complexcorrelation[i] = [abs(correlation), errorbar, indepvariable[i]]

    print("Maximum correlation is: ", str(max((list(abs(correlations))))), ", at epoch:", list(abs(correlations)).index(max((list(abs(correlations))))) + 1)
    # print(correlations)
    # print(losses)

    # pyplot.figure(figsize=(10, 3.5))
    # pyplot.subplot(1, 2, 1)
    # pyplot.plot(np.arange(iteration), losses)
    # pyplot.xlabel('iteration')
    # pyplot.ylabel('loss')
    # pyplot.subplot(1, 2, 2)
    # pyplot.plot(np.arange(iteration), abs(correlations))
    # pyplot.xlabel('iteration')
    # pyplot.ylabel('correlation')
    # pyplot.show()

    # with open("MI_cor", "wb") as fp:
    #     pickle.dump(correlations, fp)

    # with open("MI_dc", "wb") as fp:
    #     pickle.dump(correlation_driftcor, fp)

    # with open("MI_recon", "wb") as fp:
    #     pickle.dump(reconstruction, fp)


if len(var) > 1:
    print("Saving correlation vs noise level data...")
    pyplot.figure(figsize=(8, 3.5))

    pyplot.plot(var, [i[0] for i in complexcorrelation], label="mixed loss function")
    pyplot.errorbar(var, [i[0] for i in complexcorrelation], yerr =  [i[1] for i in complexcorrelation])
    pyplot.xlabel('sigma2')
    pyplot.ylabel('complex correlation')
    pyplot.xticks(var)
    pyplot.show()

    pyplot.figure(figsize=(10,3.5))
    pyplot.suptitle("Reconstructions")
    pyplot.subplot(1, 2, 1)
    pyplot.imshow(np.abs(recon[0]))
    pyplot.title("Noise sigma2 = {}".format(var[0]))
    pyplot.subplot(1, 2, 2)
    pyplot.imshow(np.abs(recon[-1]))
    pyplot.title("Noise sigma2 = {}".format(var[-1]))
    pyplot.show()


        #Saving data
    if parameters["loss"] == 'mixed':
        if parameters["optimization"] == 'ADAM':
            print("Saving MIX-ADAM data...")

            with open("MIX_CvS2_cor", "wb") as fp:
                pickle.dump(complexcorrelation, fp)

            with open("MIX_CvS2_img", "wb") as fp:
                pickle.dump(recon, fp)

            with open("MIX_CvS2_par", "wb") as fp:
                pickle.dump(parameters, fp)

        if parameters["optimization"] == 'SGD':
            print("Saving MIX-SGD data...")

            with open("SGDMIX_CvS2_cor", "wb") as fp:
                pickle.dump(complexcorrelation, fp)

            with open("SGDMIX_CvS2_img", "wb") as fp:
                pickle.dump(recon, fp)

            with open("SGDMIX_CvS2_par", "wb") as fp:
                pickle.dump(parameters, fp)


    if parameters["loss"] == 'poisson':
        if parameters["optimization"] == 'ADAM':
            print("Saving POI-ADAM data...")

            with open("POI_CvS2_cor", "wb") as fp:
                pickle.dump(complexcorrelation, fp)

            with open("POI_CvS2_img", "wb") as fp:
                pickle.dump(recon, fp)

            with open("POI_CvS2_par", "wb") as fp:
                pickle.dump(parameters, fp)

        if parameters["optimization"] == 'SGD':
            print("Saving POI-SGD data...")

            with open("SGDPOI_CvS2_cor", "wb") as fp:
                pickle.dump(complexcorrelation, fp)

            with open("SGDPOI_CvS2_img", "wb") as fp:
                pickle.dump(recon, fp)

            with open("SGDPOI_CvS2_par", "wb") as fp:
                pickle.dump(parameters, fp)




if len(probe_amplitude) > 1:
    print("Saving correlation vs probe intensity data...")
    pyplot.figure(figsize=(8, 3.5))

    # pyplot.plot(probe_amplitude, [i[0] for i in complexcorrelation], label="mixed loss function")
    pyplot.errorbar(probe_amplitude, [i[0] for i in complexcorrelation], yerr =  [i[1] for i in complexcorrelation])
    pyplot.xlabel('sigma2')
    pyplot.ylabel('complex correlation')
    pyplot.xticks(probe_amplitude)
    pyplot.show()


        #Saving data
    print(parameters["loss"])
    if parameters["loss"] == 'mixed':
        if parameters["optimization"] == 'ADAM':
            print("Saving MIX-ADAM data...")

            with open("regMIX_CvPI_cor", "wb") as fp:
                pickle.dump(complexcorrelation, fp)

            with open("regMIX_CvPI_img", "wb") as fp:
                pickle.dump(recon, fp)

            with open("regMIX_CvPI_par", "wb") as fp:
                pickle.dump(parameters, fp)

        if parameters["optimization"] == 'SGD':
            print("Saving MIX-SGD data...")

            with open("SGDMIX_CvPI_cor", "wb") as fp:
                pickle.dump(complexcorrelation, fp)

            with open("SGDMIX_CvPI_img", "wb") as fp:
                pickle.dump(recon, fp)

            with open("SGDMIX_CvPI_par", "wb") as fp:
                pickle.dump(parameters, fp)



    if parameters["loss"] == 'poisson':
        if parameters["optimization"] == 'ADAM':
            print("Saving POI-ADAM data...")

            with open("regPOI_CvPI_cor", "wb") as fp:
                pickle.dump(complexcorrelation, fp)

            with open("regPOI_CvPI_img", "wb") as fp:
                pickle.dump(recon, fp)

            with open("regPOI_CvPI_par", "wb") as fp:
                pickle.dump(parameters, fp)

        if parameters["optimization"] == 'SGD':
            print("Saving POI-SGD data...")

            with open("SGDPOI_CvPI_cor", "wb") as fp:
                pickle.dump(complexcorrelation, fp)

            with open("SGDPOI_CvPI_img", "wb") as fp:
                pickle.dump(recon, fp)

            with open("SGDPOI_CvPI_par", "wb") as fp:
                pickle.dump(parameters, fp)



if len(var) == 1 and len(probe_amplitude) == 1:

    pyplot.figure(figsize=(10, 3.5))
    pyplot.subplot(1, 2, 1)
    pyplot.plot(np.arange(iteration), losses)
    pyplot.xlabel('iteration')
    pyplot.ylabel('loss')
    pyplot.subplot(1, 2, 2)
    pyplot.plot(np.arange(iteration), abs(correlations))
    pyplot.xlabel('iteration')
    pyplot.ylabel('correlation')
    pyplot.show()


    # error = (np.abs(sample) - np.abs(sample_recon)).detach().numpy()

    # histogram, bin_edges = np.histogram(error, bins=256, range=(-1, 1))

    # # configure and draw the histogram figure
    # pyplot.figure()
    # pyplot.title(r"{} recon intensity hist, var={}".format(parameters["loss"], np.var(error)))
    # pyplot.xlabel("grayscale value")
    # pyplot.ylabel("pixel count")
    # pyplot.xlim([-1.0, 1.0])  # <- named arguments do not work here

    # pyplot.plot(bin_edges[0:-1], histogram)  # <- or here


with open("noLRDcorr", "wb") as fp:
    pickle.dump(correlations, fp)

with open("noLRDloss", "wb") as fp:
    pickle.dump(losses, fp)

with open("noLRDimage", "wb") as fp:
    pickle.dump(reconstruction, fp)
