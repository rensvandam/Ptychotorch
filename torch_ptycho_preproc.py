#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21.02.2020

Here, a set of useful postprocessing functions
(for ptychography) will be collected.

@author: Jacob Seifert  - j.seifert@uu.nl
"""

#%% Imports

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


#%% Functions

def subpixel_shift(image, dx, dy):
    ''' Performs a shift of 'image' by float number distances 'dx' and 'dy'.
            image - a 2D numpy array
            dx    - shift downwards (in pixel values)
            dy    - shift towards the right (in pixel values)
    '''
    from scipy.ndimage import fourier_shift
    image = np.fft.fft2(image)
    image = fourier_shift(image, shift=(dx, dy))
    image = np.fft.ifft2(image)
    return image


def get_center_of_2Dgaussian(image):
    import scipy.optimize as opt
    import numpy as np
    import matplotlib.pyplot as plt


    def twoD_Gaussian(x_y, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        x, y = x_y
        xo = float(xo)
        yo = float(yo)
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                                + c*((y-yo)**2)))
        return g.ravel()

    # Preparations

    image_size = image.shape[0]  # only square images allowed for now

    # Create x and y indices
    x = np.linspace(0, image_size, image_size)
    y = np.linspace(0, image_size, image_size)
    x, y = np.meshgrid(x, y)

    #create data
    #data = twoD_Gaussian((x, y), 10, 300, 312, 300, 300, 0, 10)
    data = image.ravel()

    # # plot twoD_Gaussian data generated above
    # plt.figure()
    # plt.imshow(data.reshape(image_size, image_size))
    # plt.colorbar()

    # Fitting

    # parameter structure: Amplitude, X, Y, sigma X, sigma Y, 0, offset
    initial_guess = (1, np.int_(image_size/2), np.int_(image_size/2),
                     np.int_(image_size/4), np.int_(image_size/4), 0, 10)

    # data_noisy = data + 0.2*np.random.normal(size=data.shape)

    popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data, p0=initial_guess)

    # Plotting fitting result

    data_fitted = twoD_Gaussian((x, y), *popt)
    data_fitted = data_fitted.reshape(image_size, image_size)
    data= data.reshape(image_size, image_size)

    # plt.imshow(data)
    # plt.contour(data_fitted, cmap=plt.cm.Reds, linewidths = 0.5)
    # plt.colorbar()
    # plt.title('Gaussian fit (contours)')
    # plt.show()
    # print('Fit parameters:\nAmplitude = {}\nX = {}, Y = {}\
    #       \nSigmaX = {}, SigmaY = {}\
    #       \nOffset = {}'.format(popt[0], popt[2], popt[1], popt[3],
    #                             popt[4], popt[6]))

    x_fit = popt[2]
    y_fit = popt[1]

    return x_fit, y_fit


def complex_correlation(a, b):
    ''' Returns the correlation magnitude of complex input 2D-arrays. '''
    field_correlation = np.dot(np.conjugate(a.flatten()), b.flatten()) / \
                               (np.linalg.norm(a)*np.linalg.norm(b))
    return np.abs(field_correlation)


def FRC(img1, img2, dl):
    ''' Calculate the Fourier Ring Correlation of two independent images.

    inputs:
        - image 1 and image 2: must be squared and of identical size
        - dl: pixel pitch in micrometer
        -
    outputs:
        -
    source:
        https://www.osapublishing.org/optica/abstract.cfm?uri=optica-5-1-32
    '''
    # Hamming window filtering to suppress spurious correlations
    hamming_window = signal.hamming(3)**1   # Super-Hamming window is
                                             # smoother
    window_2D = np.sqrt(np.outer(hamming_window, hamming_window))
    img1 = signal.convolve2d(img1, window_2D, mode='same')
    img2 = signal.convolve2d(img2, window_2D, mode='same')

    # Fourier transforming images
    FT_img1 = np.fft.fftshift(np.fft.fft2(img1))
    FT_img2 = np.fft.fftshift(np.fft.fft2(img2))

    # Frequency calculations
    f_max = 1/(2 * dl)            # maximum frequency [1/um]
    N = img1.shape[0]             # pixel count along edge
    delta_f = 1/(dl * N)          # frequency steps in Fourier domain
    f = np.linspace(-f_max, f_max-1/(N*dl), N)  # frequency vector
    fX, fY = np.meshgrid(f, f)
    f_map = np.sqrt(fX**2+fY**2)  # map that contains all frequencies that
                                  # correspond to the pixel location in the
                                  # Fourier domain

    ring_frequencies = np.linspace(0, f_max, int(f_max/delta_f + 1))

    # Perform FRC
    N_rings = len(ring_frequencies)
    n_q = np.zeros(N_rings)       # number of pixels contained
                                  # in Fourier ring of radius q
    FRC = np.zeros(N_rings)       # Fourier Ring Correlation
    for q in range(N_rings - 1):
        # Find rings masks based on f_map and ring_frequencies
        mask_lower = f_map >= ring_frequencies[q]
        mask_upper = f_map <= ring_frequencies[q+1]
        mask = np.logical_and(mask_lower, mask_upper)
        n_q[q+1] = np.sum(mask)

        # Use the mask to select the values within the ring and correlate
        selected_frequencies1 = FT_img1[mask]
        selected_frequencies2 = FT_img2[mask]
        FRC[q] = complex_correlation(selected_frequencies1,
                                     selected_frequencies2)

    # 1/2-bit resolution threshold
    # (source: https://linkinghub.elsevier.com/retrieve/pii/S1047847705001292
    n_q[n_q==0] = 1e-16   # avoid division by zero
    T_half_bit = (0.2071 + 1.9102 * 1/np.sqrt(n_q)) / \
                 (1.2071 + 0.9102 * 1/np.sqrt(n_q))

    ### Plots
    # smooth the 1/2-bit curve
    w_length = 5
    window = np.ones(w_length)
    T_half_bit_smoothed = np.convolve(window/np.sum(window),
                                      T_half_bit, mode='same')
    T_half_bit_smoothed[0:w_length] = T_half_bit[0:w_length]
    T_half_bit_smoothed[-w_length:] = T_half_bit[-w_length:]

    # Plot parameters
    lw = 1.8      # linewidth


    # if i == 0:



    #     plt.plot(ring_frequencies/f_max, np.abs(FRC), '-', linewidth=lw,
    #      label='drift corrected')

    # if i == 1:

    #     plt.plot(ring_frequencies/f_max, np.abs(FRC), '-', linewidth=lw,
    #      label='drift of 0.5 pixels', alpha=0.8)


    # if i == 2:

    #     plt.plot(ring_frequencies/f_max, np.abs(FRC), '-', linewidth=lw,
    #      label='drift of 1.5 pixels', alpha=0.8)

    #     plt.plot(ring_frequencies/f_max, T_half_bit_smoothed, '--', color='0.2',
    #              label='1/2-bit threshold', linewidth=lw)


    # # plt.grid()
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('FRC')
    # plt.xlabel('Nyquist frequency')
    # plt.legend(frameon=True, shadow=True)
    # plt.show()


def drift_correction(img1, img2):
    ''' Performs drift correction by finding the highest cross-correlation
        between the two images img1 and img2.
    '''

    # Fourier transforming images
    FT_img1 = np.fft.fftshift(np.fft.fft2(img1))
    FT_img2 = np.fft.fftshift(np.fft.fft2(img2))
    # print(FT_img1.shape)
    # print(FT_img2.shape)

    cross_correlation = np.fft.ifftshift(np.fft.ifft2(FT_img1 *
                                                      np.conjugate(FT_img2)))
    cross_correlation = np.abs(cross_correlation)   # only magnitudes are
                                                    # relevant
    print(cross_correlation.shape)
    # Maximum and Center of mass
    from scipy.ndimage.measurements import center_of_mass
    from scipy.ndimage import fourier_shift

    com_ROI = 8   # ROI around the maximum
    maximum = np.where(cross_correlation == np.amax(cross_correlation))
    # print(np.amax(cross_correlation))
    # print(maximum)
    # maximum = max(tuple)
    # print(maximum)
    crop = cross_correlation[int(maximum[0]-com_ROI):
                             int(maximum[0]+com_ROI+1),
                             int(maximum[1]-com_ROI):
                             int(maximum[1]+com_ROI+1)]
    crop -= np.min(crop)
    crop /= np.max(crop)

    # fitting to 2D Gaussian and correcting half-pixel fitting offset
    fit_results = get_center_of_2Dgaussian(crop)
    x_correction = fit_results[0] - 0.5 - com_ROI
    y_correction = fit_results[1] - 0.5 - com_ROI

    # estimating total drift offsets (of img2)
    image_size = img1.shape[0]
    x_drift = -(maximum[0] + x_correction - image_size/2)
    y_drift = -(maximum[1] + y_correction - image_size/2)

    # correcting drift
    img2 = subpixel_shift(img2, -x_drift, -y_drift)

    return img1, img2



#%% TESTING

# Generate images
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mathtext as mathtext
# parser = mathtext.MathTextParser("Bitmap")
# rgba1, depth1 = parser.to_rgba(
#     r'$sss$', color='black', fontsize=120, dpi=200)

# edge_length = 1024
# inputfield=np.zeros([edge_length, edge_length])



# textmask=rgba1[:,:,3]
# xwidth=textmask.shape[0]
# ywidth=textmask.shape[1]
# xpos=(edge_length-xwidth)//2
# ypos=(edge_length-ywidth)//2
# inputfield[xpos:xpos+xwidth,ypos:ypos+ywidth]=textmask/3

# img1 = np.random.poisson(inputfield) + np.random.normal(loc=1, scale=3,
#                                                         size=inputfield.shape)
# img2 = np.random.poisson(inputfield) + np.random.normal(loc=1, scale=4,
#                                                         size=inputfield.shape)

#%% generate sample
import matplotlib
from matplotlib import pyplot
import math
import torch as to
import pickle

# amp = np.matrix(np.float_(matplotlib.image.imread('images/boat.png')))
# amp = amp - np.amin(amp)
# amp = amp / np.amax(amp)

# pha = np.matrix(np.float_(matplotlib.image.imread('images/lenna.jpg')))
# pha = pha - np.amin(pha)
# pha = pha / np.amax(pha)


# def normalized(a):
#     a_oo = a - a.real.min() - 1j*a.imag.min() # origin offsetted
#     return a_oo/np.abs(a_oo).max()


# sample = np.multiply(
#     (0.25 + 0.75 * amp),
#     np.exp(1j * math.pi * (0.1 + 0.9 * pha)))
# sample = to.from_numpy(sample)

# img1 = sample

# with open("plaatje", "rb") as fp:   # Unpickling
#     img2 = pickle.load(fp)

# pyplot.figure(figsize=(10, 3.5))
# pyplot.subplot(1, 2, 1)
# pyplot.imshow(np.angle(img1.detach().numpy()), cmap='turbo')
# pyplot.colorbar()
# pyplot.subplot(1, 2, 2)
# pyplot.imshow(np.angle(img2.detach().numpy()), cmap='turbo')
# pyplot.colorbar()
# pyplot.show()


# def sliced(a, b):
#     #print(a.shape)
#     #print(b.shape)
#     begin_vert = int((list(a.shape)[0] - list(b.shape)[0])/2)
#     begin_hori = int((list(a.shape)[0] - list(b.shape)[1])/2)
#     print(int(list(b.shape)[1]))
#     sliced_vert = to.narrow(a, 1, 128, 256)
#     c = to.narrow(sliced_vert, 0, 128, 256)
#     #sliced_vert = to.narrow(a, 1, #interaction (multiplicative assumption)
#     #print(c)
#     return c

# # generate probe

# probe_pixel_num = (256, 256)
# probe_pixel_size = (1, 1)

# precision = 'float32'

# # x y width
# wy = 16
# wx = 16

# # x y pixel number
# ny = probe_pixel_num[0]
# nx = probe_pixel_num[1]

# # x y pixel size
# dy = probe_pixel_size[0]
# dx = probe_pixel_size[1]

# # x y vector
# y = np.arange(-ny / 2, ny / 2) * dy
# x = np.arange(-nx / 2, nx / 2) * dx

# # x y meshgrid
# xx, yy = np.meshgrid(x, y)

# probe = np.exp(-(((yy / wy) ** 2 + (xx / wx) ** 2) / (2 * 1 ** 2)))
# probe = probe + 0*1j
# probe = to.from_numpy(probe)

# print(img1.shape)
# #img2 = sliced(img2, probe)
# img1 = sliced(img1, probe)
# print(img1.shape)

# pyplot.figure(figsize=(10, 3.5))
# pyplot.subplot(1, 2, 1)
# pyplot.imshow(np.angle(img1.detach().numpy()), cmap='turbo')
# pyplot.colorbar()
# pyplot.subplot(1, 2, 2)
# pyplot.imshow(np.angle(img2.detach().numpy()), cmap='turbo')
# pyplot.colorbar()
# pyplot.show()


# img1 = img1.detach().numpy()
# img2 = img2.detach().numpy()




# # Parameters

# dl = 10  # pixel pitch in mirometer

# i = 0

# img3 = subpixel_shift(img2, -100, 0)

# plt.figure()
# plt.imshow(np.angle(img1))
# plt.figure()
# plt.imshow(np.angle(img3))

# img1, img3 = drift_correction(img1, img3)

# plt.figure()
# plt.imshow(np.angle(img1))
# plt.figure()
# plt.imshow(np.angle(img3))

# plt.figure()

# FRC(img1, img3, dl)
# i+=1

# img3 = subpixel_shift(img2, -0.5, 0)
# FRC(img1, img3, dl)
# i+=1

# img3 = subpixel_shift(img2, -1.5, 0)
# FRC(img1, img3, dl)


# plt.figure()
# plt.grid()
# plt.tight_layout()

# plt.savefig('drift_correction.png', dpi=600)

























# EOF