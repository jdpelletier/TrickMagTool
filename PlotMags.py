import argparse
import os

import numpy as np
from photutils import DAOStarFinder
from photutils import aperture_photometry, CircularAperture
from astropy.stats import mad_std
from astropy.io import fits
import matplotlib.pyplot as plt

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

def findApSum(folder):
    ap_list=[]
    for filename in os.listdir(args.folder):
        f = os.path.join(args.folder, filename)
        print(f)
        image = processData(f)
        image = image[0:1360, 0:2048].clip(min=0)
        bkg_sigma = mad_std(image)
        daofind = DAOStarFinder(fwhm=30., threshold=100.*bkg_sigma)
        sources = daofind(image)
        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
        apertures = CircularAperture(positions, r=15.)
        phot_table = aperture_photometry(image, apertures)
        brightest_source_id = phot_table['aperture_sum'].argmax()
        ap_list.append(phot_table[brightest_source_id]['aperture_sum'])
    return ap_list

def sumPlot(sums):
    plt.plot(sums)
    plt.savefig("Aps.jpeg")
    plt.close()

def main():
    sums = findApSum(args.folder)
    sumPlot(sums)

if __name__ =='__main__':
    main()
