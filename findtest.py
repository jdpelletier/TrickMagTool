import argparse

import numpy as np
from photutils import DAOStarFinder
from photutils import aperture_photometry, CircularAperture
from astropy.stats import mad_std
from astropy.io import fits
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="find stars in image test",
                         usage="findtest.py filename")

parser.add_argument("filename", help="filename")


args = parser.parse_args()

def processData(filen):
    fitsData = fits.getdata(filen, ext=0)
    header = fits.getheader(filen)
    mask = fits.getdata('BadPix_1014Hz.fits', ext=0)
    maskedData = np.multiply(fitsData, mask)
    filter = header['TRFWNAME']
    maskedData = np.multiply(fitsData, mask)
    if filter == 'H':
            background = fits.getdata('H_sky.fits')
    else:
            background = fits.getdata('ks_sky.fits')
    finaldat = maskedData - background
    return finaldat

def findStars(image):
    image = image[0:1360, 0:2048].clip(min=0)
    bkg_sigma = mad_std(image)
    daofind = DAOStarFinder(fwhm=30., threshold=100.*bkg_sigma)
    sources = daofind(image)
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=15.)
    phot_table = aperture_photometry(image, apertures)
    brightest_source_id = phot_table['aperture_sum'].argmax()
    print(phot_table[brightest_source_id]['aperture_sum'])
    plt.imshow(image, cmap='gray_r', origin='lower')
    apertures.plot(color='blue', lw=1.5, alpha=0.5)
    plt.show()

def main():
    pdata = processData(args.filename)
    findStars(pdata)


if __name__ =='__main__':
    main()
