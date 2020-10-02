import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from photutils.detection import IRAFStarFinder
from photutils import aperture_photometry, CircularAperture
from astropy.stats import mad_std
from astropy.io import fits
import astropy.units as u
from photutils.psf import IntegratedGaussianPRF, DAOGroup, IterativelySubtractedPSFPhotometry
from photutils.background import MMMBackground, MADStdBackgroundRMS
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.modeling import models, fitting


parser = argparse.ArgumentParser(description="plot ap sum for stars in folder",
                         usage="PlotMags.py folder")

parser.add_argument("folder", help="folder")

args = parser.parse_args()

def processData(filen):
    fitsData = fits.getdata(filen, ext=0)
    header = fits.getheader(filen)
    filter = header['TRFWNAME']
    mask = fits.getdata('BadPix_1014Hz.fits', ext=0)
    maskedData = np.multiply(fitsData, mask)
    if filter == 'H':
            background = fits.getdata('H_sky.fits')
    else:
            background = fits.getdata('ks_sky.fits')
    finaldat = maskedData - background
    return finaldat

def findStars(f):
    image = processData(f)
    sigma_psf = 20
    image = image[0:1360, 0:2048].clip(min=0)
    bkgrms = MADStdBackgroundRMS()
    std = bkgrms(image)
    iraffind = IRAFStarFinder(threshold=20*std,
                            fwhm=sigma_psf*gaussian_sigma_to_fwhm,
                            minsep_fwhm=0.01, roundhi=5.0, roundlo=-5.0,
                            sharplo=0.0, sharphi=2.0)
    sources = iraffind(image)
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=15.)
    phot_table = aperture_photometry(image, apertures)
    brightest_source_id = phot_table['aperture_sum'].argmax()
    x = int(phot_table[brightest_source_id]['xcenter']/u.pix)
    xmin = x-40
    xmax = x+40
    y = int(phot_table[brightest_source_id]['ycenter']/u.pix)
    ymin = y-40
    ymax = y+40
    x_line = image[y, xmin:xmax]
    y_line = image[ymin:ymax, x]
    return x_line, y_line

def fitStars(x_line, y_line):
    x = np.linspace(-40, 40, 80)
    model_gauss = models.Gaussian1D()
    fitter_gauss = fitting.LevMarLSQFitter()
    gx = fitter_gauss(model_gauss, x, x_line)
    gy = fitter_gauss(model_gauss, x, y_line)
    plt.scatter(x, x_line)
    plt.scatter(x, y_line)
    plt.plot(x, gx(x), label='GaussianX')
    gy = fitter_gauss(model_gauss, x, y_line)
    plt.plot(x, gy(x), label='GaussianY')
    plt.show()
    amplitude = (gx.amplitude.value+gy.amplitude.value)/2
    fwhm = ((gx.stddev.value+gy.stddev.value)/2)*0.118 #2.355*PixelScale
    gauss_list = [amplitude, fwhm]
    return gauss_list

def plotStars(filename):
    amp, fwhm = np.loadtxt(filename, delimiter=',', unpack=True)
    plt.scatter(fwhm, amp)
    plt.show()

def main():
    datafile = "StarData.txt"
    measurements = []
    for filename in os.listdir(args.folder):
        gauss_list = []
        f = os.path.join(args.folder, filename)
        print(f)
        try:
            x, y = findStars(f)
            gauss_list = fitStars(x, y)
            measurements.append(gauss_list)
        except TypeError:
            print("Could not find star")
        except ValueError:
            print("Need to fix edges")
    f = open(datafile, "w+")
    for star in measurements:
        f.write(f"{str(star[0])}, {str(star[1])}\n")
    f.close()
    plotStars(datafile)

if __name__ =='__main__':
    main()
