import os, time, sys, threading
from os import listdir
from os.path import abspath, isfile, join
from pathlib import Path
import math
import subprocess
import datetime

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy import wcs
import astropy.units as u
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.modeling import models, fitting
import PIL.Image as PILimage

from ginga import Bindings
from ginga.misc import log
from ginga.qtw.QtHelp import QtGui, QtCore
from ginga.qtw.ImageViewQt import CanvasView, ScrolledView
from ginga.util import iqcalc
from ginga.util.loader import load_data


class FitsViewer(QtGui.QMainWindow):

    def __init__(self, logger):
        super(FitsViewer, self).__init__()
        self.logger = logger

        self.rawfile = ''
        self.filelist = []
        self.fnumber = 0
        self.amplitude = 0
        self.fwhm = 0

        self.iqcalc = iqcalc.IQCalc(self.logger)

        # create the ginga viewer and configure it
        fi = CanvasView(self.logger, render='widget')
        fi.enable_autocuts('on')
        fi.set_autocut_params('zscale')
        fi.enable_autozoom('on')
        # fi.set_callback('drag-drop', self.drop_file)
        fi.set_bg(0.2, 0.2, 0.2)
        fi.ui_set_active(True)
        self.fitsimage = fi

        # enable some user interaction
        self.bd = fi.get_bindings()
        self.bd.enable_all(True)

        w = fi.get_widget()
        w.resize(512, 512)

        # add scrollbar interface around this viewer
        si = ScrolledView(fi)

        vbox = QtGui.QVBoxLayout()
        vbox.setContentsMargins(QtCore.QMargins(2, 2, 2, 2))
        vbox.setSpacing(1)
        vbox.addWidget(si, stretch=1)

        hbox = QtGui.QHBoxLayout()
        hbox.setContentsMargins(QtCore.QMargins(4, 2, 4, 2))

        self.readout = QtGui.QLabel("")

        hbox.addStretch(1)
        hbox.addWidget(self.readout, stretch = 0)

        self.box_readout = QtGui.QLabel("")

        hbox.addStretch(1)
        hbox.addWidget(self.box_readout, stretch = 0)

        hw = QtGui.QWidget()
        hw.setLayout(hbox)
        vbox.addWidget(hw, stretch=0)

        hbox2 = QtGui.QHBoxLayout()
        hbox2.setContentsMargins(QtCore.QMargins(4, 2, 4, 2))
        self.wnext = QtGui.QPushButton("Next image")
        self.wnext.clicked.connect(self.next_image)
        self.wnext.setEnabled(False)
        self.wmark = QtGui.QPushButton("Mark Star")
        self.wmark.clicked.connect(self.mark_star)
        self.wmark.setEnabled(False)
        wopend = QtGui.QPushButton("Open Directory")
        wopend.clicked.connect(self.open_directory)
        wopenf = QtGui.QPushButton("Open File")
        wopenf.clicked.connect(self.open_file)
        wsky = QtGui.QPushButton("Load Sky")
        wsky.clicked.connect(self.load_sky)
        wquit = QtGui.QPushButton("Quit")
        wquit.clicked.connect(self.quit)
        fi.set_callback('cursor-changed', self.motion_cb)
        fi.add_callback('cursor-down', self.btndown)
        hbox2.addStretch(1)
        for w in (self.wmark, self.wnext, wopend, wopenf, wsky, wquit):
            hbox2.addWidget(w, stretch=0)

        hw2 = QtGui.QWidget()
        hw2.setLayout(hbox2)
        vbox.addWidget(hw2, stretch=0)

        vw = QtGui.QWidget()
        self.setCentralWidget(vw)
        vw.setLayout(vbox)
        self.recdc, self.compdc = self.add_canvas()
        self.picktag = "pick-box"


    def add_canvas(self, tag=None):
        # add a canvas to the view
        my_canvas = self.fitsimage.get_canvas()
        RecCanvas = my_canvas.get_draw_class('rectangle')
        CompCanvas = my_canvas.get_draw_class('compass')
        return RecCanvas, CompCanvas

    def load_file(self, filepath):
        image = load_data(filepath, logger=self.logger)
        self.fitsimage.set_image(image)
        # self.setWindowTitle(filepath)
        try:
            self.fitsimage.get_canvas().get_object_by_tag(self.picktag)
            self.fitsimage.get_canvas().delete_object_by_tag(self.picktag)
        except KeyError:
            pass

    def open_directory(self):
        self.filelist = self.read_folder(QtGui.QFileDialog.getExistingDirectory(
                                                self, "Open FITS folder", '.'))
        self.open_image(self.filelist[0])
        self.fnumber = 0
        self.wnext.setEnabled(True)
        self.wmark.setEnabled(True)


    def next_image(self):
        self.fnumber = self.fnumber + 1
        try:
            self.open_image(self.filelist[self.fnumber])
            self.wmark.setEnabled(True)
        except IndexError:
            print("end of files")

    def mark_star(self):
        filename = "GalCenterData.txt"
        text = f"{self.amplitude:.2f} {self.fwhm:.2f}"
        f = open(filename, "a")
        f.write(text)
        f.close()
        self.wmark.setEnabled(False)

    def open_file(self):
        res = QtGui.QFileDialog.getOpenFileName(self, "Open FITS file",
                                                '.')
        self.open_image(res)

    def open_image(self, res):
        if isinstance(res, tuple):
            fileName = res[0]
        else:
            fileName = str(res)
        if len(fileName) != 0:
            self.processData(fileName)

    def read_folder(self, directory):
        return [abspath(join(directory, f)) for f in listdir(directory) if isfile(join(directory, f))]

    def load_sky(self):
        res = QtGui.QFileDialog.getOpenFileName(self, "Open Sky file",
                                                '.')
        if isinstance(res, tuple):
            fileName = res[0]
        else:
            fileName = str(res)
        if len(fileName) != 0:
            self.subtract_sky(fileName)

    def subtract_sky(self, filename):
        skyheader, skyfitsData, skyfilter = self.addWcs(filename)
        header, fitsData, filter = self.addWcs(self.rawfile)
        # if filter == 'H':
        #     oldbackground = fits.getdata('/kroot/rel/ao/qfix/data/Trick/H_sky.fits')
        # else:
        #     oldbackground = fits.getdata('/kroot/rel/ao/qfix/data/Trick/ks_sky.fits')
        # no_sky = fitsData + oldbackground
        with_sky = fitsData - skyfitsData
        mask = fits.getdata('BadPix_1014Hz.fits', ext=0)
        self.load_file(self.writeFits(header, np.multiply(with_sky, mask)))


    def cut_change(self):
        self.fitsimage.set_autocut_params(self.wcut.currentText())

    def color_change(self):
        self.fitsimage.set_color_algorithm(self.wcolor.currentText())

    def motion_cb(self, viewer, button, data_x, data_y):

        # Get the value under the data coordinates
        try:
            # We report the value across the pixel, even though the coords
            # change halfway across the pixel
            value = viewer.get_data(int(data_x + 0.5), int(data_y + 0.5))

        except Exception:
            value = None

        fits_x, fits_y = data_x, data_y

        # Calculate WCS RA
        try:
            # NOTE: image function operates on DATA space coords
            image = viewer.get_image()
            if image is None:
                # No image loaded
                return
            ra_txt, dec_txt = image.pixtoradec(fits_x, fits_y,
                                               format='str', coords='fits')
        except Exception as e:
            self.logger.warning("Bad coordinate conversion: %s" % (
                str(e)))
            ra_txt = 'BAD WCS'
            dec_txt = 'BAD WCS'

        text = "X: %.2f  Y: %.2f  Value: %s" % (fits_x, fits_y, value)
        self.readout.setText(text)

    def quit(self, *args):
        self.logger.info("Attempting to shut down the application...")
        self.deleteLater()

    def processData(self, filename):
        self.rawfile = filename
        header, fitsData, filter = self.addWcs(filename)
        mask = fits.getdata('BadPix_1014Hz.fits', ext=0)
        if filter == 'H':
            background = fits.getdata('/kroot/rel/ao/qfix/data/Trick/H_sky.fits')
        else:
            background = fits.getdata('/kroot/rel/ao/qfix/data/Trick/ks_sky.fits')
        subtracted_data = fitsData-background
        self.load_file(self.writeFits(header, np.multiply(subtracted_data, mask)))

    def addWcs(self, filen):
        w = wcs.WCS(naxis=2)
        fitsData = fits.getdata(filen, ext=0)
        header = fits.getheader(filen)
        ht, wd = fitsData.shape[:2]
        y = ht//2
        x = wd//2
        ra = float(header['RA'])
        dec = float(header['DEC'])
        filter = header['TRFWNAME']
        w.wcs.crpix = [y, x]
        w.wcs.cdelt = np.array([-0.05, 0.05])
        w.wcs.crval = [ra, dec]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        pixcrd = np.array([[0, 0], [24, 38], [45, 98]], dtype=np.float64)
        world = w.wcs_pix2world(pixcrd, 0)
        # Convert the same coordinates back to pixel coordinates.
        pixcrd2 = w.wcs_world2pix(world, 0)
        # These should be the same as the original pixel coordinates, modulo
        # some floating-point error.
        assert np.max(np.abs(pixcrd - pixcrd2)) < 1e-6
        # Now, write out the WCS object as a FITS header
        header = w.to_header()
        return header, fitsData, filter

    def writeFits(self, headerinfo, image_data):
        hdu = fits.PrimaryHDU(header=headerinfo, data=image_data)
        filename = 'procImage.fits'
        try:
            hdu.writeto(filename)
        except OSError:
            os.remove(filename)
            hdu.writeto(filename)
        return filename

    ##Find star stuff
    def cutdetail(self, image, shape_obj):
        view, mask = image.get_shape_view(shape_obj)

        data = image._slice(view)

        y1, y2 = view[0].start, view[0].stop
        x1, x2 = view[1].start, view[1].stop

        # mask non-containing members
        mdata = np.ma.array(data, mask=np.logical_not(mask))

        return x1, y1, x2, y2, mdata

    def findstar(self):
        image = self.fitsimage.get_image()
        obj = self.pickbox
        shape = obj
        x1, y1, x2, y2, data = self.cutdetail(image, shape)
        ht, wd = data.shape[:2]
        xc, yc = wd // 2, ht // 2
        radius = min(xc, yc)
        peaks = [(xc, yc)]
        peaks = self.iqcalc.find_bright_peaks(data,
                                              threshold=None,
                                              radius=radius)

        xc, yc = peaks[0]
        xc += 1
        yc += 1
        return int(xc), int(yc), data

    def fitstars(self, y_line):
        x = np.linspace(-10, 10, 21)
        model_gauss = models.Gaussian1D()
        fitter_gauss = fitting.LevMarLSQFitter()
        # gx = fitter_gauss(model_gauss, x, x_line)
        gy = fitter_gauss(model_gauss, x, y_line)
        # plt.scatter(x, x_line)
        # plt.scatter(x, y_line)
        # # plt.plot(x, gx(x), label='GaussianX')
        # plt.plot(x, gy(x), label='GaussianY')
        # plt.show()
        # amplitude = (gx.amplitude.value+gy.amplitude.value)/2
        amplitude = gy.amplitude.value
        # fwhm = ((gx.stddev.value+gy.stddev.value)/2)*0.118 #2.355*PixelScale
        fwhm = (gy.stddev.value)*0.118 #2.355*PixelScale
        return amplitude, fwhm

    def pickstar(self, viewer):
        try:
            self.fitsimage.get_canvas().get_object_by_tag(self.picktag)
            self.fitsimage.get_canvas().delete_object_by_tag(self.picktag)
            self.pickbox = self.recdc(self.xclick-10, self.yclick-10, self.xclick+10, self.yclick+10, color='red')
            self.fitsimage.get_canvas().add(self.pickbox, tag=self.picktag, redraw=True)
        except KeyError:
            self.pickbox = self.recdc(self.xclick-10, self.yclick-10, self.xclick+10, self.yclick+10, color='red')
            self.fitsimage.get_canvas().add(self.pickbox, tag=self.picktag, redraw=True)
        image = self.fitsimage.get_image()
        try:
            xc, yc, data = self.findstar()
            # x_line = data[40-yc, 0:40] doesn't work well for some reason
            y_line = data[0:60, xc]
            # amplitude, fwhm = self.fitstars(x_line, y_line)
            self.amplitude, self.fwhm = self.fitstars(y_line)
            text = f"Amplitude: {self.amplitude:.2f} FWHM: {self.fwhm:.2f}"
            self.box_readout.setText(text)
        except IndexError:
            text = "Amplitude: N/A FWHM: N/A"
            self.box_readout.setText(text)


    def btndown(self, canvas, event, data_x, data_y):
        # self.fitsimage.set_pan(data_x, data_y)
        self.xclick = data_x
        self.yclick = data_y
        self.pickstar(self.fitsimage)


def main():
    ##Write dummy file so walkDirectory caches it in the beginning

    app = QtGui.QApplication([])

    # ginga needs a logger.
    # If you don't want to log anything you can create a null logger by
    # using null=True in this call instead of log_stderr=True
    logger = log.get_logger("example1", log_stderr=True, level=40)

    w = FitsViewer(logger)
    w.resize(600, 700)
    w.show()
    app.setActiveWindow(w)
    w.raise_()
    w.activateWindow()
    app.exec_()

if __name__ == "__main__":
    main()
