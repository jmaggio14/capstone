import imagepypelines as ip
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import glob
from photutils.psf import IntegratedGaussianPRF
# from photutils import DAOStarFinder

# # Import ImagePypelines
# import imagepypelines as ip
# # require the astronomy plugin, which contains this project
# ip.require('astro')
#
# filenames = ['code to fetch your dataset here']
#
# # construct the pipeline from the astronomy plugin
# pipeline = ip.astro.ExoplanetCandidateFinder(
#                             # size of exclusion zone from edge of sensor [pixels]
#                             border_mask=100,
#                             # Best guess for the Full-Width Half Max of sources [pixels]
#                             fwhm=8,
#                             # bitdepth of the sensor
#                             sensor_bitdepth=16,
#                             # radius of aperture used for photometry [pixels]
#                             aperture_radius=6,
#                             # inner ring of annus radius [pixels]
#                             inner_annulus_radius=11,
#                             # outer ring of annulus radius [pixels]
#                             outer_annulus_radius=17,
#                             # number of stars for the ensemble corrrection
#                             n_stars_ensemble=100,
#                             # z score to be classified as a variable star
#                             variable_z_score=2,
#                             # width of outlier detection window [magnitudes]
#                             outlier_detection_window_range=2,
#                             )
# # process the data
# processed = pipeline.process( filenames )
# # print out the pixel coordinates of suspected variable objects
# print(processed['variable_candidate_locations'])


BORDER_MASK_WIDTH = 100 # pixels
FWHM = 12 # pixels
DETECTION_THRESHOLD_SIGMA = 5 # in sigma above image
SENSOR_BITDEPTH = 16 # bits per DN integer
APERTURE_RADIUS = 7
INNER_ANNULUS_RADIUS = 11
OUTER_ANNULUS_RADIUS = 17


# CONSTANTS TO TWEAK TELESCOPE PARAMETERS
# plate scale
# sigma clip, etc
# psf model

### TO CONSTUCT BLOCKS FOR HDU LOADING

# # to fetch the primary header no matter it's location in the file
# load_primary = ip.astro.LoadPrimaryHDU()
#
# # to fetch the first three HDUs - 0 based indexing
# load_first_hdu = ip.astro.LoadHDU0()
# load_second_hdu = ip.astro.LoadHDU1()
# load_third_hdu = ip.astro.LoadHDU2()
#
# # load an arbitrary header by label or by index
# hdu_index = 'EXAMPLE'
# load_example = ip.astro.HduLoader(hdu_index)

### TO CONSTRUCT BLOCKS FOR HEADER METADATA FETCHING
exposure_time_fetch = ip.astro.HeaderFetch('EXPTIME')

from photutils import datasets
# hdu = datasets.load_star_image()
# data = hdu.data[0:401, 0:401]
#
#
#
#
#
# moon_fname = ip.astro.moon()

# filenames = sorted(glob.glob(r"C:\Users\jmagg\Desktop\School-Online-Spring\ObsAstro\final project data\aligned_clean\*.fits"))
filenames = sorted(glob.glob(r"C:\Users\jmagg\Desktop\School-Online-Spring\ObsAstro\final project data\aligned_clean\*.fits"))
# DAOStarFinder(fwhm=5.0, threshold=20)

# BorderCrop
# IRAF Star Finder
# Ensemble photometry
# DrawSourceOutlines
#
plt.ion()

@ip.blockify()
def logscale(image):
    return np.log10(image)

bit12 = 2**12

@ip.blockify(batch_type='all',void=True)
def quick_plot(x,y):
    plt.figure()
    plt.plot(x,y)
    return None

# finder = ip.astro.IRAFStarFinder(mask=int(BORDER_MASK_WIDTH),
#                                     fwhm=float(FWHM),
#                                     threshold=500,
#                                     roundhi=1.0,
#                                     sharplo=0.0,
#                                     sharphi=3.0,
#                                     )

finder = ip.astro.DAOStarFinder(
                                mask=int(BORDER_MASK_WIDTH),
                                fwhm=float(FWHM),
                                sharplo=0.0,
                                sharphi=3.0,
                                threshold=2**SENSOR_BITDEPTH * .005,
                                peakmax = 2**SENSOR_BITDEPTH * .95,
                                    )

find_reliable_stars = {
        # Inputs
        'filenames':ip.Input(0),
        # load in data files
        ('headers','images') : (ip.astro.LoadPrimaryHDU(), 'filenames'),
        # Fetch Julian Date
        'jd': (ip.astro.HeaderFetch('JD'), 'headers'),
        # stack all images
        'stacked' : (ip.astro.Stack(mode='median'), 'images'),
        # Detect Stars in the Image
        'stars' : (finder, 'stacked'),
        ('apertures','annuli') : (ip.astro.AperturesAndAnnuli(APERTURE_RADIUS, INNER_ANNULUS_RADIUS, OUTER_ANNULUS_RADIUS), 'stars'),

        # shrink the images for viewing and make them normalized to 8bit
        'gray' : (ip.image.NormDtype(np.uint8), 'stacked'),
        'enhanced' : (ip.image.LinearHistEnhance(0,0.1), 'gray'),
        'color_display': (ip.image.Gray2RGB(), 'enhanced'),
        'source_outlined' : (ip.astro.DrawSourceOutlines(radius=20, thickness=4), 'color_display', 'stars'),
        'numbered_sources' : (ip.image.NumberImage(), 'source_outlined'),
        'viewable' : (ip.image.Resize(scale_w=0.5,scale_h=0.5), 'numbered_sources'),
        # display the sources in the image
        'null.1' : (ip.image.QuickView(100), 'viewable'),
        }
star_judge = ip.Pipeline(find_reliable_stars, name='StarJudge')
processed = star_judge.process(filenames)



aper_phot_tasks = {
            # inputs
            # 'headers': ip.Input(0),
            'images' : ip.Input(0),
            'apertures': ip.Input(1),
            'annuli': ip.Input(2),
            #  integrate stellar counts
            ('counts','count_errs'): (ip.astro.AperturePhotometry(),'images','apertures','annuli'),
            # calculate instrumental magnitudes
            'mags' : (ip.astro.InstrumentalMag(10.43),'counts'),
            'mag_errs' : (ip.astro.InstrumentalMag(),'count_errs'),
            }

aper_phot = ip.Pipeline(aper_phot_tasks, name='AperturePhotometry')
processed2 = aper_phot.process(list(processed['images']),
                                list(processed['apertures']) * len(processed['images']),
                                list(processed['annuli']) * len(processed['images'])
                                )



ensemble_out = ('corrected_magnitudes', 'image_correction_factors', 'image_correction_error', 'mean_stellar_magnitude', 'mean_stellar_mag_error')
ensemble_tasks = {
                'stars':ip.Input(0),
                'mags' : ip.Input(1),
                'mag_errs':ip.Input(2),
                ensemble_out : (ip.astro.EnsemblePhotometry(), 'mags', 'mag_errs'),
                }

ensemble = ip.Pipeline(ensemble_tasks, name='EnsemblePhotometry')
processed3 = ensemble.process([processed['stars']] * len(processed2['mags']),
                                list(processed2['mags']),
                                list(processed2['mag_errs']),
                                )

plt.figure()
plt.scatter(processed3['mean_stellar_magnitude'],np.sqrt(processed3['mean_stellar_mag_error']))
plt.xlabel('mean_stellar_magnitude')
plt.ylabel('mean_stellar_mag_error')

plt.figure()
plt.scatter(processed['jd'],processed3['image_correction_factors']+10.43)
plt.xlabel('Julian date')
plt.ylabel('image correction factor')

plt.show()
#
# aper_phot =
#
# ensemble_tasks =


# ip.ExoplanetCandidateFinder(**telescope_kwargs)


# daophot = ip.astro.DAOPhotometry(crit_separation=30,
#                                     psf_model=IntegratedGaussianPRF(),
#                                     fitshape=(21,21),
#                                     fwhm=3.0,
#                                     threshold=0.05*bit12,
#                                     roundlo=-0.3,
#                                     roundhi=0.3,
#                                     sharplo=.45,
#                                     sharphi=1.0,
#                                     peakmax=.75*bit12,
#                                     )
#
# photometry_tasks = {
#                     # Input
#                     'images':ip.Input(0),
#                     # perform photometry on this set
#                     'results' : (daophot, 'images'),
#                     # make a viewable image
#                     'gray_display' : (ip.image.NormDtype(np.uint8), 'images'),
#                     'color_display': (ip.image.Gray2RGB(), 'gray_display'),
#                     'source_outlined' : (ip.astro.DrawSourceOutlines(radius=5), 'color_display', 'results'),
#                     # display the source outline images found by the photometer
#                     'null.1' : (ip.image.CompareView(10e3,title1='no sources',title2='source outlines'), 'color_display', 'source_outlined'),
#                      }
# photometer = ip.Pipeline(photometry_tasks)
#
# processed = photometer.process([data])





import pdb; pdb.set_trace()
