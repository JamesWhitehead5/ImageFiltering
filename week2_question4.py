import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':

    im = Image.open("GeorgeVanderbiltII_cropped.jpg")
    im = np.array(im)[:, :, 0]  # select a single color channel so the array is 2D

    # display the image
    plt.figure("")
    plt.imshow(im, cmap="gray")

    # get the number of pixels along a single dimension
    n_pixels, _ = im.shape

    # take the DFT of the image
    image_dft = np.fft.fft2(im)

    f_nyquist = 0.5  # the maximum frequency that can be represented in the dft. Used for plotting.
    # To view the Fourier transform without the weights arranged unintuitively, we must apply the fftshift.
    plt.figure()
    plt.imshow(np.abs(np.fft.fftshift(image_dft)), extent=[-f_nyquist, f_nyquist, -f_nyquist, f_nyquist])
    plt.title("Magnitude of the Fourier transform of the image")
    plt.xlabel("Spatial frequency x (cycles/pixel)")
    plt.ylabel("Spatial frequency y (cycles/pixel)")
    plt.colorbar()

    # Low pass filter removes all spatial frequencies that are higher than f_max
    fx = np.fft.fftfreq(n=n_pixels)  # 1d array of spatial frequencies in the x axis
    fy = fx  # 1d array of spatial frequencies in the y axis (they are the same since the image is square)

    # meshgrid generates fxx and fyy which are 2D arrays. For any index in the 2D DFT,
    # the x frequency is given by `fxx` at that same index and the y frequency is given by `fyy` at the same index.
    fxx, fyy = np.meshgrid(fx, fy, indexing='xy')
    f_min = 0.1  # the maximum frequency that can be allowed the through the filter
    f_max = 0.5

    img_low_pass_dft = image_dft.copy()
    img_low_pass_dft[fxx ** 2 + fyy ** 2 < f_min ** 2] = 0.  # apply the filter
    img_low_pass_dft[fxx ** 2 + fyy ** 2 > f_max ** 2] = 0.
    img_low_pass = np.fft.ifft2(img_low_pass_dft)  # reverse the DFT on the filtered transform

    plt.figure()
    plt.imshow(np.real(img_low_pass), cmap='gray')
    plt.colorbar()
    plt.show()
