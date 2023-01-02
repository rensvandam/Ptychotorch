# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 11:28:48 2022

@author: rensvandam
"""
import pickle
import matplotlib
from matplotlib import pyplot
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
import torch as to


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

print(abs(ASMi50_cor))

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

# %% Real data results

def crop(image, cropsize):

    cropped = image[int((image.shape[0] - cropsize)/2) : int(image.shape[0] - (image.shape[0] - cropsize)/2), int((image.shape[1] - cropsize)/2) :  int(image.shape[1] - (image.shape[1] - cropsize)/2 )]

    return cropped



#%% Longexp Bandlimit Mixed

R1title = 'Mixed, no BL'
with open("longexp_ASMI_recon", "rb") as fp:
    R1 = pickle.load(fp)

R2title = 'Mixed, BL'
with open("longexp_ASMIBL_recon", "rb") as fp:
    R2 = pickle.load(fp)

L1title = 'Losses no BL'
with open("longexp_ASMI_losses", "rb") as fp:
    L1 = pickle.load(fp)

L2title = 'Losses BL'
with open("longexp_ASMIBL_losses", "rb") as fp:
    L2 = pickle.load(fp)

#%% Longexp Bandlimit Poisson

R1title = 'Poisson, no BL'
with open("longexp_ASP_recon", "rb") as fp:
    R1 = pickle.load(fp)

R2title = 'Poisson, BL'
with open("longexp_ASPBL_recon", "rb") as fp:
    R2 = pickle.load(fp)

L1title = 'Losses no BL'
with open("longexp_ASP_losses", "rb") as fp:
    L1 = pickle.load(fp)

L2title = 'Losses BL'
with open("longexp_ASPBL_losses", "rb") as fp:
    L2 = pickle.load(fp)


#%% Short exposure

R1title = 'Mixed recon short exposure'
with open("shortexp_ASMI_recon", "rb") as fp:
    R1 = pickle.load(fp)

R2title = 'Poisson recon short exposure'
with open("shortexp_ASP_recon", "rb") as fp:
    R2 = pickle.load(fp)

L1title = 'Losses mixed'
with open("shortexp_ASMI_losses", "rb") as fp:
    L1 = pickle.load(fp)

L2title = 'Losses poisson'
with open("shortexp_ASP_losses", "rb") as fp:
    L2 = pickle.load(fp)

#%% Short exposure BL

R1title = 'Mixed recon BL short exposure'
with open("shortexp_ASMIBL_recon", "rb") as fp:
    R1 = pickle.load(fp)

R2title = 'Poisson recon BL short exposure'
with open("shortexp_ASPBL_recon", "rb") as fp:
    R2 = pickle.load(fp)

L1title = 'Losses mixed'
with open("shortexp_ASMIBL_losses", "rb") as fp:
    L1 = pickle.load(fp)

L2title = 'Losses poisson'
with open("shortexp_ASPBL_losses", "rb") as fp:
    L2 = pickle.load(fp)


#%%

pyplot.figure(figsize=(10, 7))
pyplot.subplot(2, 2, 1)
pyplot.imshow(crop(np.abs(R1), 500), cmap='gray')
pyplot.title(R1title)
pyplot.colorbar()
pyplot.subplot(2, 2, 2)
pyplot.imshow(crop(np.abs(R2), 500), cmap='gray')
pyplot.title(R2title)
pyplot.colorbar()
pyplot.subplot(2, 2, 3)
pyplot.plot(np.arange(len(L1)), L1, color = 'orange')
pyplot.xlabel('epoch')
pyplot.ylabel('Losses')
pyplot.title(L1title)
pyplot.subplot(2, 2, 4)
pyplot.plot(np.arange(len(L2)), L2, color = 'b')
pyplot.xlabel('epoch')
pyplot.ylabel('Losses')
pyplot.title(L2title)
pyplot.show()


#%% making fake groundtruth for correlation

# pyplot.figure(figsize=(5, 3.5))
# pyplot.imshow(crop(np.abs(R1), 500), cmap='gray')
# pyplot.title(R1title)

# with open("4_fakeGT", "wb") as fp:
#     pickle.dump(crop(np.abs(R1), 500), fp)


#%% Calculating correlation with better recon

with open("4_fakeGT", "rb") as fp:
    GT = pickle.load(fp)

def crop(image, cropsize):

    cropped = image[int((image.shape[0] - cropsize)/2) : int(image.shape[0] - (image.shape[0] - cropsize)/2), int((image.shape[1] - cropsize)/2) :  int(image.shape[1] - (image.shape[1] - cropsize)/2 )]

    return cropped

def calc_correlation(a,b):
    a = to.from_numpy(a)
    b = to.from_numpy(b)
    corr = to.sum(to.conj(a)*b)/\
        (to.sqrt(to.sum(to.abs(b)**2))*to.sqrt(to.sum(to.abs(a)**2)))
    return corr

from postprocessing import drift_correction
R1, R2 = drift_correction(crop(np.abs(R1), 500), crop(np.abs(R2), 500))


# print("correlation mixed short exposure:", float(np.abs(calc_correlation(crop(np.abs(R1), 500), GT))))
# print("correlation poisson short exposure:", float(np.abs(calc_correlation(crop(np.abs(R2), 500), GT))))


pyplot.figure(figsize=(10, 7))
pyplot.subplot(2, 2, 1)
pyplot.imshow(crop(np.abs(R1), 500), cmap='gray')
pyplot.title(R1title+',  corr='+str(round(float(np.abs(calc_correlation(crop(np.abs(R1), 500), GT))), 3)))
pyplot.colorbar()
pyplot.subplot(2, 2, 2)
pyplot.imshow(crop(np.abs(R2), 500), cmap='gray')
pyplot.title(R2title+',  corr='+str(round(float(np.abs(calc_correlation(crop(np.abs(R2), 500), GT))), 3)))
pyplot.colorbar()
pyplot.subplot(2, 2, 3)
pyplot.plot(np.arange(len(L1)), L1, color = 'orange')
pyplot.xlabel('epoch')
pyplot.ylabel('Losses')
pyplot.title(L1title)
pyplot.subplot(2, 2, 4)
pyplot.plot(np.arange(len(L2)), L2, color = 'b')
pyplot.xlabel('epoch')
pyplot.ylabel('Losses')
pyplot.title(L2title)
pyplot.show()

