import argparse
import glob
import os
from os import listdir
from os.path import join

import numpy as np
from imageio import imread, imwrite
from scipy.optimize import minimize
from skimage.metrics import peak_signal_noise_ratio
from tqdm import trange

from gmm_posterior_expected_value import gmm_posterior_expected_value
from nets import *
from callback import LogProgress

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--path',required=True,help='path to dataset root')
parser.add_argument('--dataset',required=True,help='dataset name e.g. Confocal_MICE')
parser.add_argument('--mode',default='uncalib',help='noise model: mse, uncalib, gaussian, poisson, poissongaussian')
parser.add_argument('--reg',type=float,default=0.1,help='regularization weight on prior std. dev.')
parser.add_argument('--components',type=int,default=1,help='number of mixture components')
parser.add_argument('--tag',type=str,default="",help='id tag to add to weights path')

args = parser.parse_args()

if args.components != 1 and args.mode != "uncalib":
    raise ValueError("Components != 1 must be used with mode uncalib")    

""" Re-create the model and load the weights """

model = gaussian_blindspot_network((512, 512, 1),'uncalib',components=args.components)

if args.mode == 'uncalib' or args.mode == 'mse':
    if args.components == 1:
        if args.tag == "":
            weights_path = 'weights/weights.%s.%s.latest.hdf5'%(args.dataset,args.mode)
        else:
            weights_path = 'weights/weights.%s.%s.%s.latest.hdf5'%(args.dataset,args.mode,args.tag)
    else:
        weights_path = 'weights/weights.%s.%s.%dcomponents.latest.hdf5'%(args.dataset,args.mode,args.components)
else:
    weights_path = 'weights/weights.%s.%s.%0.3f.latest.hdf5'%(args.dataset,args.mode,args.reg)

model.load_weights(weights_path)

""" Load test images """

test_images = []

def load_images(noise):
    basepath = args.path + '/' + args.dataset + '/' + noise
    images = []
    for path in sorted(glob.glob(basepath + '/*.tif')):
        images.append(imread(path))
    # only use last 10% for testing
    split = int(len(images) * 9 / 10)
    images = images[split:]
    return np.stack(images,axis=0)[:,:,:,None]/255.


X = load_images('raw')
Y = load_images('gt')
gt = np.squeeze(Y)*255

""" Denoise test images """
def poisson_gaussian_loss(x,y,a,b):
    var = np.maximum(1e-4,a*x+b)
    loss = (y-x)**2 / var + np.log(var)
    return np.mean(loss)
optfun = lambda p, x, y : poisson_gaussian_loss(x,y,p[0],p[1])

def denoise_uncalib(y,loc,std,a,b):
    total_var = std**2
    noise_var = np.maximum(1e-3,a*loc+b)
    noise_std = noise_var**0.5
    prior_var = np.maximum(1e-4,total_var-noise_var)
    prior_std = prior_var**0.5
    return np.squeeze(gaussian_posterior_mean(y,loc,prior_std,noise_std))

def gmm_sum_weighted_means(locs, weights):
    """
    get expected value of gmm prior, as the sum of weighted means
    """
    weighted = locs * weights
    return np.sum(weighted, axis=-1)



experiment_name = '%s.%s'%(args.dataset,args.mode)
if args.tag != "":
    experiment_name += '.%s'%(args.tag)
if args.mode == 'uncalib' or args.mode == 'mse':
    if args.components != 1:
        experiment_name += '.%dcomponents'%(args.components)
else:
    experiment_name += '.%0.3f'%(args.reg)

    
os.makedirs("results/%s"%experiment_name,exist_ok=True)
results_path = 'results/%s.tab'%experiment_name

def do_psnr(gt, test, message):
   test = np.squeeze(test*255)
   test = np.clip(test,0,255)
   print(message + " psnr:", peak_signal_noise_ratio(gt, test, data_range=255))

logger = LogProgress(experiment_name, for_test=True)

# all_sqr_errs = []

with open(results_path,'w') as f:
    f.write('inputPSNR\tdenoisedPSNR\n')
    for index,im in enumerate(X):
        pred = model.predict(im.reshape(1,512,512,1), callbacks=[logger])

        if args.mode == 'uncalib':
            # select only pixels above bottom 2% and below top 3% of noisy image
            good = np.logical_and(im >= np.quantile(im,0.02), im <= np.quantile(im,0.97))[None,:,:,:]
            if args.components == 1:
                pseudo_clean = pred[0][good]
            else:
                full_pseudo_clean = gmm_sum_weighted_means(pred[0], pred[2])
                pseudo_clean = full_pseudo_clean[np.squeeze(good, axis=-1)]
            noisy = im[np.squeeze(good, axis=0)]

            # estimate noise level
            res = minimize(optfun, (0.01,0), (np.squeeze(pseudo_clean),np.squeeze(noisy)), method='Nelder-Mead')
            print('bootstrap poisson-gaussian fit: a = %f, b=%f, loss=%f'%(res.x[0],res.x[1],res.fun))
            a,b = res.x

            # run denoising
            if args.components == 1:
                denoised = denoise_uncalib(im[None,:,:,:],pred[0],pred[1],a,b)
            else:
                # Gaussian mixture model
                do_psnr(gt, full_pseudo_clean, "pseudoclean")

                noise_sigma = np.sqrt( np.maximum(1e-3, a*full_pseudo_clean+b) )
                stacked_total_std = np.tile( (a*full_pseudo_clean+b).reshape((1,512,512,1)), (1,1,1,args.components))
                prior_var = pred[1]**2 - stacked_total_std**2
                prior_std = np.sqrt(np.clip(prior_var, 1e-4, None))
                denoised = gmm_posterior_expected_value(components=args.components,
                                                        mus=pred[0],
                                                        sigs=prior_std,
                                                        weights=pred[2],
                                                        z=im[None,:,:,:],
                                                        noisesig=noise_sigma)
                denoised = K.eval(denoised)

        else:
            denoised = pred[0]
                 
        # scale and clip to 8-bit
        denoised = np.squeeze(denoised*255)
        denoised = np.clip(denoised, 0, 255)
        
        # write out image
        imwrite('results/%s/%02d.png'%(experiment_name,index),denoised.astype('uint8'))

        noisy = np.squeeze(im)*255
        psnr_noisy = peak_signal_noise_ratio(gt, noisy, data_range = 255)
        psnr_denoised = peak_signal_noise_ratio(gt, denoised, data_range = 255)

        print(psnr_noisy,psnr_denoised)
        f.write('%.15f\t%.15f\n'%(psnr_noisy,psnr_denoised))

        # if args.mode == "uncalib":
        #     low = (noisy < np.quantile(noisy, 0.02))
        #     high = (noisy > np.quantile(noisy, 0.97))
        #     good = np.squeeze(good)
        #     print(np.sum(low), "low,", np.sum(high), "high pixels out of", 512*512)
        #     squared_err = np.square(denoised - gt)
        #     print("good:", squared_err[good].mean(), ", low:", squared_err[low].mean(), ", high:", squared_err[high].mean())
        #     all_sqr_errs.append(squared_err)
        #     plt.yscale("log")
        #     plt.scatter(gt.flatten(), squared_err.flatten())
        #     plt.savefig("correlation-plt" + str(index) + ".png")
        #     plt.clf()

""" Print averages """
results = np.loadtxt(results_path,delimiter='\t',skiprows=1)
print('averages:')
avgs = np.mean(results,axis=0)
print(avgs)
logger.log_psnr(avgs)

# def normalize(arr):
#     """
#     normalize a an array to range from 0 to 255
#     """
#     arr = arr - np.min(arr)
#     return arr / np.max(arr) * 255

# if args.mode == "uncalib":
#     all_sqr_errs = np.array(all_sqr_errs)
#     avged = np.mean(all_sqr_errs, axis=0)
#     avged = normalize(avged)
#     correlation = np.abs(avged - noisy)
#     os.makedirs("misc", exist_ok=True)
#     imwrite("misc/mse-correlation." + experiment_name + ".png", correlation.astype("uint8"))

