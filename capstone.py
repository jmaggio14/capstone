import imagepypelines as ip
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
# from photutils import DAOStarFinder
ip.require('image', 'astro')

from photutils import datasets
hdu = datasets.load_star_image()
data = hdu.data[0:401, 0:401]

moon_fname = ip.astro.moon()

# DAOStarFinder(fwhm=5.0, threshold=20)

# BorderCrop
# IRAF Star Finder
# Ensemble photometry
# DrawSourceOutlines
#

Resizer

tasks = {
        # 'filenames':ip.Input(0),
        # ('headers','images'): (ip.astro.LoadPrimaryHDU(),'filenames'),
        'images':ip.Input(0),
        # shrink the images for viewing and make them normalized to 8bit
        'small' : (ip.image.Resize(scale_w=1/8,scale_h=1/8,interp='nearest'), 'images'),
        'normalized_0_255' : (ip.image.NormDtype(np.uint8), 'small'),
        #
        'stars' : (ip.astro.DAOStarFinder(),'images'),
        'null'  : (ip.astro.ViewSourceOutlines(radius=5), 'images', 'stars'),
        }
fits_loader = ip.Pipeline(tasks)


# headers,data = fits_loader.process_and_grab([moon_fname],
#                                                 fetch=['headers',' small'])
processed = fits_loader.process([data])



import pdb; pdb.set_trace()
