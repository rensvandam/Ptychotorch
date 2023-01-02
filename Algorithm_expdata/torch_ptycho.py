# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 11:52:42 2022

@author: rens van dam
"""


# %% import modules
import pickle
import torch as to
import numpy as np
import math
import matplotlib
import h5py
from matplotlib import pyplot
from matplotlib.patches import Circle
from matplotlib_scalebar.scalebar import ScaleBar
matplotlib.use('Qt5Agg')

#%%
device = to.device("cuda") if to.cuda.is_available() else to.device("cpu")
to.set_default_tensor_type('torch.cuda.FloatTensor')

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

#%% Importing Real data

def get_ptychodata(size, pixelsize_um, plot_scanpos=False, plot_diffpat=False, shuffle=False):
    if size == 'large_shortexposure':
        data = "3__2022-03-23-11-03-25_ptychoMeasurement.hdf5"
    if size == 'large_longexposure':
        data = "calib__2022-03-23-10-53-48_ptychoMeasurement.hdf5"
    if size == 'small':
        data = "19__2022-03-23-11-04-12_ptychoMeasurement.hdf5"

    data = h5py.File(data, "r")

    background = np.array(data.get('background'))
    scan_pos_list = np.array(data.get('encoder'))*1000*1000/pixelsize_um #from meter to millimeter to micrometer to pixels
    diff_pat_list = np.array(data.get('ptychogram'))

    for i in range(len(diff_pat_list)):
        diff_pat_list[i] = diff_pat_list[i] - background
        diff_pat_list[i][diff_pat_list[i] < 0] = 0
        diff_pat_list[i] = diff_pat_list[i] #increasing diffpat intensity improves recon it seems

    if plot_scanpos == True:
        pyplot.figure(figsize=(5, 5))
        pyplot.title("Scan positions in pixels")
        pyplot.plot(scan_pos_list[:, 1], scan_pos_list[:, 0], 'o')
        pyplot.show()

    if plot_diffpat == True:
        image([diff_pat_list[4]], 1, "example diff pat")

    if shuffle == True:

        def shuffle(list1, list2):
            indices = np.arange(list1.shape[0])
            np.random.shuffle(indices)
            list1 = list1[indices]
            list2 = list2[indices]
            return [list1, list2]

        shuffled = shuffle(scan_pos_list, diff_pat_list)
        diff_pat_list = shuffled[1]
        scan_pos_list = shuffled[0]

    return [background, pixelsize_um, scan_pos_list, diff_pat_list]

data = get_ptychodata('small', 3.5, plot_scanpos=False, plot_diffpat=False, shuffle=False)
scan_pos_list = to.from_numpy(data[2]).to(device)
diff_pat_list = to.from_numpy(data[3]).to(device)
pixelsize = data[1]

#%% Importing images for simulated data



# %% generate sample

def generatedata(pixelsize, z, lam, c, BL, simulationdata):

    if simulationdata == False:

        def get_ptychodata(size, pixelsize_um, plot_scanpos=False, plot_diffpat=False, shuffle=False):
            if size == 'large_shortexposure':
                data = "3__2022-03-23-11-03-25_ptychoMeasurement.hdf5"
            if size == 'large_longexposure':
                data = "calib__2022-03-23-10-53-48_ptychoMeasurement.hdf5"
            if size == 'small':
                data = "19__2022-03-23-11-04-12_ptychoMeasurement.hdf5"

            data = h5py.File(data, "r")

            background = np.array(data.get('background'))
            scan_pos_list = np.array(data.get('encoder'))*1000*1000/pixelsize_um #from meter to millimeter to micrometer to pixels
            diff_pat_list = np.array(data.get('ptychogram'))

            for i in range(len(diff_pat_list)):
                diff_pat_list[i] = diff_pat_list[i] - background
                diff_pat_list[i][diff_pat_list[i] < 0] = 0
                diff_pat_list[i] = diff_pat_list[i] #increasing diffpat intensity improves recon it seems

            if plot_scanpos == True:
                pyplot.figure(figsize=(5, 5))
                pyplot.title("Scan positions in pixels")
                pyplot.plot(scan_pos_list[:, 1], scan_pos_list[:, 0], 'o')
                pyplot.show()

            if plot_diffpat == True:
                image([diff_pat_list[4]], 1, "example diffraction pattern")

            if shuffle == True:

                def shuffle(list1, list2):
                    indices = np.arange(list1.shape[0])
                    np.random.shuffle(indices)
                    list1 = list1[indices]
                    list2 = list2[indices]
                    return [list1, list2]

                shuffled = shuffle(scan_pos_list, diff_pat_list)
                diff_pat_list = shuffled[1]
                scan_pos_list = shuffled[0]

            return [background, pixelsize_um, scan_pos_list, diff_pat_list]

        data = get_ptychodata('small', 3.5, plot_scanpos=False, plot_diffpat=False, shuffle=False)
        scan_pos_list = to.from_numpy(data[2]).to(device)
        diff_pat_list = to.from_numpy(data[3]).to(device)
        pixelsize = data[1]

    if simulationdata == True:

        #example images amplitude and phase
        amp = np.matrix(np.float_(matplotlib.image.imread('images/boat.png')))
        pha = np.matrix(np.float_(matplotlib.image.imread('images/lenna.jpg')))

        npx = amp.shape[0]
        npy = amp.shape[1]

        #SAMPLE

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

        def generate_sample(amp, pha, mask, plot=False):

            amp = amp - np.amin(amp)
            amp = amp / np.amax(amp)

            pha = pha - np.amin(pha)
            pha = pha / np.amax(pha)

            sample = np.multiply(
                (0.25 + 0.75 * amp),
                np.exp(1j * math.pi * (0.1 + 0.9 * pha)))
            sample = to.from_numpy(sample)
            #sample = to.nn.functional.pad(sample, (256,256,256,256), "constant", 0)


            if mask == True:
                sample = masked(sample)

            if plot == True:
                image([np.abs(sample), np.angle(sample)], 2, "Sample", color="gray")

            return sample

        sample = generate_sample(amp, pha, mask=True, plot=False)
        sample = sample.to(device)

        #PROBE
        def generate_probe(probetype, probesize, pixelsize, plot=False):
            #x,y width

            wy = 16
            wx = 16

            #x,y pixel number
            ny = probesize[0]
            nx = probesize[1]

            #x,y pixel size
            dy = pixelsize
            dx = pixelsize

            # x y vector
            y = np.arange(-ny / 2, ny / 2) * dy
            x = np.arange(-nx / 2, nx / 2) * dx

            xx, yy = np.meshgrid(x,y)

            if probetype == 'gaussian':
                probe = 20*np.exp(-(((yy / wy) ** 2 + (xx / wx) ** 2) / (2 * 1 ** 2)))
                probe = probe + 0*1j
                probe = to.from_numpy(probe)

            if plot == True:
                image([np.abs(probe), np.angle(probe)], 2, title="Probe", color='jet')

            return probe


        probesize = (int(npx/2), int(npy/2))

        probe = generate_probe('gaussian', probesize, pixelsize, plot=False)
        probe = probe.to(device)

        def generate_scanpos(sample, method, plot):
            if method == "grid":
                v = np.arange(-100.0, 120.0, 20)
                u = np.arange(-100.0, 120.0, 20)
                vv, uu = np.meshgrid(v, u)
                scan_pos_list = np.transpose([vv.flatten(), uu.flatten()])
                scan_pos_list = np.reshape(scan_pos_list, newshape=(len(scan_pos_list), 2))

                if plot == True:
                    pyplot.figure(figsize=(5, 5))
                    pyplot.plot(scan_pos_list[:, 1], scan_pos_list[:, 0], 'o')

                    for x,y in scan_pos_list:
                        circ = Circle((x,y),60, alpha=0.05)
                        pyplot.gca().add_patch(circ)
                    pyplot.show()

                return scan_pos_list

            if method == "sunflower":

                num_pts = 121
                indices = np.arange(0, num_pts, dtype=float) + 0.5
                r = np.sqrt(indices/num_pts)
                theta = np.pi * (1 + 5**0.5) * indices
                scan_pos_list = [0]*num_pts
                for i in range(len(scan_pos_list)):
                    scan_pos_list[i] = [115*r[i]*np.cos(theta[i]), 115*r[i]*np.sin(theta[i])]
                scan_pos_list = np.reshape(scan_pos_list, newshape=(len(scan_pos_list), 2))

                if plot == True:
                    pyplot.figure(figsize=(5, 5))
                    pyplot.title("Scanning positions for {} points".format(num_pts))
                    pyplot.plot(scan_pos_list[:, 1], scan_pos_list[:, 0], 'o')
                    for x,y in scan_pos_list:
                        circ = Circle((x,y),60, alpha=0.05)
                        pyplot.gca().add_patch(circ)
                    ax=pyplot.gca()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    pyplot.show()

                return scan_pos_list

        scan_pos_list = generate_scanpos(sample, method="sunflower", plot=False)
        scan_pos_list = to.from_numpy(scan_pos_list).to(device)

        from torch_ptycho_model import AssembleModel
        model = AssembleModel(sample, probe)
        print(model)

        def simulate_diff_pat(scanpositions, probesize, Nx, Ny, pixelsize, z, lam, C, BL, mask=False, noise=False, plot=False):
            diff_pat_list = []
            for ind in range(len(scan_pos_list)):
                diff_pat_list.append(model(scan_pos_list[[ind]], probesize, Nx, Ny, pixelsize, z, lam, C, BL).cpu().detach().numpy())

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
                mask = create_circular_mask(512,512, radius=20)
                mask = 1-mask
                #for i in range(len(diff_pat_list)):
                    #diff_pat_list[i] = np.abs(np.fft.fftshift(np.fft.fft2(diff_pat_list[i])))*mask
                    #diff_pat_list[i] = np.abs(np.fft.ifftshift(np.fft.ifft2(diff_pat_list[i])))
                    #diff_pat_list[i] = mask*diff_pat_list[i]
                #image([mask], 1, "mask")
                #image([1-mask], 1, "1-mask")
            diffpatsize = diff_pat_list[0].shape[0]
            diff_pat_list = np.array(diff_pat_list)
            diff_pat_list = diff_pat_list.reshape(len(scan_pos_list),diffpatsize,diffpatsize)
            diff_pat_list = np.abs(diff_pat_list)**2


            if noise==True:

                from torch_ptycho_model import noisy
                for i in range(len(diff_pat_list)):
                    diff_pat_list[i] = noisy("poisson", diff_pat_list[i])
                    diff_pat_list[i] = noisy("gauss", diff_pat_list[i])

            if plot==True:
                image([diff_pat_list[4]], 1, "diffraction pattern #4")


            diff_pat_list = to.from_numpy(diff_pat_list)

            return diff_pat_list



        z = 200 #um
        lam = 6E-4 #um
        C = 0.1 #regularization term for antialiasing
        BL = None
        diff_pat_list = simulate_diff_pat(scan_pos_list, probesize, probesize[0], probesize[1], pixelsize, z, lam, C, BL, mask=False, noise=True, plot=True)


    #SAMPLE INIT AND PROBE INIT

    def create_sample_guess(samplesize, plot=False):
        sample_init = np.ones(samplesize)

        if simulationdata == True:
            sample_init = masked(sample_init)

        sample_init = to.from_numpy(sample_init)
        if plot == True:
            image([np.abs(sample_init), np.angle(sample_init)], 2, "sample initializer", color="gray")

        return sample_init

    def create_probe_guess(probesize, pixelsize, plot=False):
        #x,y width
        wy = 16
        wx = 16
        #x,y pixel number
        ny = probesize[0]
        nx = probesize[1]
        #x,y pixel size
        dy = pixelsize
        dx = pixelsize
        y = np.arange(-ny / 2, ny / 2) * dy
        x = np.arange(-nx / 2, nx / 2) * dx
        xx, yy = np.meshgrid(x,y)

        probe_init = np.ones(probesize)
        probe_init[np.sqrt((yy / wy / 2) ** 2 + (xx / wx / 2) ** 2) > 1] = 0
        probe_init = to.from_numpy(probe_init)

        if plot == True:
            image([np.abs(probe_init), np.angle(probe_init)], 2, "probe initializer", color="gray")

        return probe_init


    if simulationdata == False:

        npx = diff_pat_list[1].shape[0]
        npy = diff_pat_list[1].shape[1]
        #SAMPLE
        size = (int(npx*2),int(npy*2))
        sample_init = create_sample_guess(size, plot=False)
        sample_init = sample_init.to(to.complex128)

        #PROBE (REAL PROBE)
        calib = np.load("calibration.npz")
        probe = calib['probe']
        probe = np.reshape(probe, (npx,npy))
        probe = to.from_numpy(probe)
        probe_init = probe #*np.sqrt(0.00043) #only if i need lighter probe for shorter exposure
        probesize = (npx, npy)
        print(to.sum(to.abs(probe)))
    #image([abs(probe_init)], 1)

    if simulationdata == True:

        size = (npx, npy)
        sample_init = create_sample_guess(size, plot=False)
        sample_init = sample_init.to(to.complex128)

        probe_init = create_probe_guess(probesize, pixelsize, plot=False)
        probe_init = probe_init.to(to.complex128)


    return [diff_pat_list.to(device), scan_pos_list, sample_init.to(device), probe_init.to(device), pixelsize, probesize]


simulationdata = False

if simulationdata == False:
    sigma2 = np.load("background_measurement.npz")['background_variances'][-1,:,:] # use here 3 instead of -1 for short exposure data
    sigma2 = to.from_numpy(sigma2).to(device)


    # pyplot.figure()
    # pyplot.imshow(sigma2.cpu().detach().numpy(), cmap='turbo')
    # pyplot.show()

    print(to.sum(to.abs(sigma2))/(1024*1024))

    pixelsize = 3.5 #um
    z = 65461 #um
    lam = 0.561 #um
    C = 1 #regularization term for antialiasing
    BL = None

if simulationdata == True:
    sigma2 = 1
    pixelsize = 1 #um
    z = 20 #um
    lam = 6E-4 #um
    C = 1000 #regularization term for antialiasing
    BL = None


stuff = generatedata(pixelsize, z, lam, C, BL, simulationdata)

diff_pat_list = stuff[0]
scan_pos_list = stuff[1]
sample_init = stuff[2]
probe_init = stuff[3]
pixelsize = stuff[4]
probesize = stuff[5]


# %% Initializing model
from torch_ptycho_model import AssembleModel
from torch_ptycho_model import loss_func_poisson #best lr: 0.01
from torch_ptycho_model import loss_func_gauss   # best lr: 0.01
from torch_ptycho_model import loss_func_sum   # best lr: 0.005
from torch_ptycho_model import loss_func_mixed # best lr: 0.006
from torch_ptycho_model import compute_regularization


model = AssembleModel(sample_init.to(device), probe_init.to(device))

loss_func = loss_func_mixed
Nx = probesize[0]
Ny = probesize[1]
iteration = 50
lr = 0.01
regularization = False
regul = compute_regularization


model(scan_pos_list[[0]], probesize, Nx, Ny, pixelsize, z, lam, C, BL=BL)
print("model", model(scan_pos_list[[0]], probesize, Nx, Ny, pixelsize, z, lam, C, BL=BL).shape)
with open("exit_field_test", "rb") as fp:
    exit_field = pickle.load(fp)

loss = loss_func(diff_pat_list[0], to.abs(model(scan_pos_list[[0]], probesize, Nx, Ny, pixelsize, z, lam, C, BL=BL))**2, device=device, regul=regul(exit_field, pixelsize, Nx, Ny, z, C, lam, regularization), sigma2=sigma2)
print('loss function used:' + str(loss_func))
print('loss: ' + str(loss.cpu().detach().numpy()))
optimizer = to.optim.Adam(model.parameters(), lr=lr)


#%% Some functions for plotting and correlation

pyplot.ion()


fig = pyplot.figure(figsize=(10,10))
X = [ (0, 3,2,1), (1, 3,2,2), (2, 3,2,3), (3, 3,2,4), (4, 3,2,(5,6))]
axs = [0]*5
for plot, nrows, ncols, plot_number in X:
    axs[plot] = pyplot.subplot(nrows, ncols, plot_number, adjustable='box')

#axs =  = pyplot.subplot(2, 3, figsize=(10, 7), gridspec_kw=gs_kw)
sample = to.complex(
    model.scan.sample_real,
    model.scan.sample_imag)
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

losses = np.zeros(iteration)
regularizationterms = np.zeros(iteration)
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

# sample = generate_sample(mask=True)

# with open("sample", "wb") as fp:
#     pickle.dump(sliced(sample, probe), fp)

# sample = groundtruth

#%% Optimizing
exit_field_test = to.zeros((1024,1024))
# with open("exit_field", "wb") as fp:
#     pickle.dump(exit_field, fp)


for epoch in np.arange(iteration):
    avgterm = 1/(len(scan_pos_list))
    regulepoch= [0]*len(scan_pos_list)
    lossepoch = [0]*len(scan_pos_list)
    for ind in range(len(scan_pos_list)):
        ###################### the only important part ######################

        with open("exit_field", "rb") as fp:
            exit_field = pickle.load(fp)

        reg = regul(exit_field, pixelsize, Nx, Ny, z, C, lam, regularization)

        loss = loss_func(diff_pat_list[ind], to.abs(model(scan_pos_list[[ind]], probesize, Nx, Ny, pixelsize, z, lam, C, BL=BL))**2, device=device, regul=reg, sigma2=sigma2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # if ind == 3 and epoch in [2,3,4,5]:
        #     pyplot.figure(figsize=(5, 3.5))
        #     pyplot.imshow(np.abs(exit_field.cpu().detach().numpy()), cmap='gray')
        #     pyplot.colorbar()
        #     pyplot.show()




        # if ind == 3 and epoch in [3,4,5,6,10,26]:

        #     sample_recon = to.complex(
        #         model.scan.sample_real,
        #         model.scan.sample_imag)

        #     pyplot.figure(figsize=(10, 7))
        #     pyplot.subplot(2, 2, 1)
        #     pyplot.imshow(np.abs(model(scan_pos_list[[ind]] , probesize, Nx, Ny, pixelsize, z, lam, C, BL).cpu().detach().numpy())**2, cmap='turbo')
        #     pyplot.colorbar()
        #     pyplot.subplot(2, 2, 2)
        #     pyplot.imshow(np.abs(diff_pat_list[3].cpu().detach().numpy()), cmap='turbo')
        #     pyplot.colorbar()
        #     pyplot.subplot(2, 2, 3)
        #     pyplot.imshow(np.angle(model(scan_pos_list[[ind]] , probesize, Nx, Ny, pixelsize, z, lam, C, BL).cpu().detach().numpy()), cmap='turbo')
        #     pyplot.colorbar()
        #     pyplot.subplot(2, 2, 4)
        #     pyplot.imshow(np.angle(diff_pat_list[3].cpu().detach().numpy()), cmap='turbo')
        #     pyplot.colorbar()
        #     pyplot.show()

            # pyplot.figure(figsize=(5, 3.5))
            # pyplot.imshow(np.abs(sample_recon.cpu().detach().numpy()), cmap='turbo')
            # pyplot.colorbar()
            # pyplot.show()


        lossepoch[ind] = loss.cpu().detach().numpy()
        regulepoch[ind] = reg.cpu().detach().numpy()#add loss of a scanpos

        ###################### the only important part ######################

    #losses[epoch] = loss.cpu().detach().numpy()
    losses[epoch] = avgterm*sum(lossepoch)
    print(losses[epoch])
    regularizationterms[epoch] = avgterm*np.sum(regulepoch)
    print(regularizationterms[epoch])



    if simulationdata == True:

        #Calculating correlation for this epoch
        sample_recon = to.complex(
            model.scan.sample_real,
            model.scan.sample_imag)

        sample = slice_for_correlation(sample, 150)
        sample_recon = slice_for_correlation(sample_recon, 150)

        sample = sample.cpu().detach().numpy()
        sample_recon = sample_recon.cpu().detach().numpy()

        correlations[epoch] = calc_correlation(sample, sample_recon).detach().numpy()
        sample = to.from_numpy(sample)

    # Plotting

    scalebar = ScaleBar(3.5, units="um") # 1 pixel = 3.5 μm
    print(
        'epoch (' + str(epoch + 1) + '/' + str(iteration) + ') ' +
        'loss: ' + str(losses[epoch]) +
        ' correlation: ' + str(abs(correlations[epoch])))
    axs[0].clear()
    axs[1].clear()
    axs[2].clear()
    axs[3].clear()
    axs[4].clear()
    sample_recon = to.complex(
        model.scan.sample_real,
        model.scan.sample_imag)
    axs[0].imshow(np.abs(sample_recon[750:1250, 750:1250].cpu().detach().numpy()), cmap='gray')
    scalebar = ScaleBar(pixelsize, units="um")
    axs[0].add_artist(scalebar)

    axs[1].imshow(np.angle(sample_recon[750:1250, 750:1250].cpu().detach().numpy()), cmap='gray')
    scalebar = ScaleBar(pixelsize, units="um")
    axs[1].add_artist(scalebar)

    probe_recon = to.complex(
        model.interact.probe_real,
        model.interact.probe_imag)
    axs[2].imshow(np.abs(probe_recon.cpu().detach().numpy()), cmap='gray')
    scalebar = ScaleBar(pixelsize, units="um")
    axs[2].add_artist(scalebar)

    axs[3].imshow(np.angle(probe_recon.cpu().detach().numpy()), cmap='gray')
    scalebar = ScaleBar(pixelsize, units="um")
    axs[3].add_artist(scalebar)


    axs[4].plot(range(len(losses[0:epoch])), losses[0:epoch], label="Total")
    axs[4].plot( range(len(regularizationterms[0:epoch])), regularizationterms[0:epoch], label="Regularization")
    axs[4].plot(range(len(losses[0:epoch])), losses[0:epoch] - regularizationterms[0:epoch], label = "Loss")
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







#%% Plotting conclusion

sample = slice_for_correlation(sample, 150).cpu().detach().numpy()
sample_recon = slice_for_correlation(sample_recon, 150).cpu().detach().numpy()

from torch_ptycho_preproc import drift_correction
sample, sample_recon = drift_correction(sample, sample_recon)
correlation_driftcor = calc_correlation(sample, sample_recon)
sample = to.from_numpy(sample)
sample_recon = to.from_numpy(sample_recon)
print( ' correlation after DC: ' + str(abs(correlation_driftcor)))




reconstruction = sample_recon.cpu().detach().numpy()

print(correlations)
print(losses)

pyplot.figure(figsize=(5, 3.5))
pyplot.plot(range(iteration), abs(correlations))
pyplot.xlabel('iteration')
pyplot.ylabel('correlation')
pyplot.show()

# with open("4_ASP_cor", "wb") as fp:
#     pickle.dump(correlations, fp)

# with open("4_ASP_dc", "wb") as fp:
#     pickle.dump(correlation_driftcor, fp)

# with open("longexp_ASMI_recon", "wb") as fp:
#     pickle.dump(reconstruction, fp)

# with open("longexp_ASMI_losses", "wb") as fp:
#     pickle.dump(losses, fp)


# with open("longexp_ASMIBL_recon", "wb") as fp:
#     pickle.dump(reconstruction, fp)

# with open("longexp_ASMIBL_losses", "wb") as fp:
#     pickle.dump(losses, fp)