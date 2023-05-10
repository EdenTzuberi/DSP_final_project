# Final Project

import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib import patches


def zplane(b, a):
    """Plot the complex z-plane given a transfer function."""

    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0, 0), radius=1, fill=False, color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coefficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = b / float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a / float(kd)
    else:
        kd = 1

    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn / float(kd)

    # Plot the zeros and set marker properties
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp(t1, markersize=10.0, markeredgewidth=1.0, markeredgecolor='k', markerfacecolor='g')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp(t2, markersize=12.0, markeredgewidth=3.0, markeredgecolor='r', markerfacecolor='r')

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set the ticks
    r = 1.5
    plt.axis('scaled')
    plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1]
    plt.xticks(ticks)
    plt.yticks(ticks)

    return z, p, k


def main():
    # Load arrays or pickled objects from .npy, .npz or pickled files.
    image = np.load('input_signal.npy')

    plt.figure()
    w = np.linspace(0, 2 * np.pi, 122500)

    # DFT of the image
    dft_image = np.fft.fft(image)
    plt.xlabel("Frequency", fontsize=10)
    plt.ylabel("Amplitude", fontsize=10)
    plt.title("image with 'DFT(fft)' command", fontsize=10)
    plt.plot(abs(dft_image))
    plt.grid()

    # reshaped image
    plt.figure()
    reshaped_image = np.reshape(image, [350, 350])  # image size: 350 X 350
    plt.imshow(reshaped_image, cmap='gray')

    print(scipy.signal.find_peaks(abs(dft_image), 0.5 * 10 ** 7))


    # normalized noise freq:  (15313/122500) * 2 * pi = pi/4

    plt.figure()
    # b - zeros
    num = np.array([1, -(np.exp(-1jnoise_freq = scipy.signal.find_peaks(abs(dft_image), 0.5 * 10 ** 7)[0][0]
    print(noise_freq) * np.pi / 4) + np.exp(1j * np.pi / 4)), 1], dtype=complex)
    # a = poles
    den = np.array([1, -0.98 * (np.exp(-1j * np.pi / 4) + np.exp(1j * np.pi / 4)), 0.98 ** 2], dtype=complex)

    zplane(num, den)

    # h - freq response
    plt.figure()
    w = np.linspace(0, 0.999 * 2 * np.pi, 1000)
    plt.xlabel("Frequency", fontsize=10)
    plt.ylabel("Amplitude", fontsize=10)
    z = np.exp(1j * w)
    plt.title('Frequency response of the notch filter', fontsize=10)
    num = z ** 2 - z * (np.exp(-1j * np.pi / 4) + np.exp(1j * np.pi / 4)) + 1
    den = z ** 2 - 0.98 * z * (np.exp(-1j * np.pi / 4) + np.exp(1j * np.pi / 4)) + 0.98 ** 2
    freq_res = num / den
    plt.plot(w, abs(freq_res))
    plt.grid()

    # i + j + k
    plt.figure()
    y = np.zeros(122500)
    x = image

    for num in range(2, 122500):
        y[num] = x[num] - np.sqrt(2) * x[num - 1] + x[num - 2] + np.sqrt(2) * 0.98 * y[num - 1] - 0.98 ** 2 * y[num - 2]

    filtered_image = np.reshape(y, [350, 350])
    plt.imshow(filtered_image, cmap='gray')
    plt.title('filtered image with the notch filter', fontsize=10)

    # m - draw the frequency response of Hmin and Hap
    # H_min
    plt.figure()
    w = np.linspace(0, 0.999 * 2 * np.pi, 1000)
    plt.xlabel("Frequency", fontsize=10)
    plt.ylabel("Amplitude", fontsize=10)
    z = np.exp(1j * w)
    plt.title('Frequency response of H_min', fontsize=10)
    num_hmin = -1.5 * (z - (2 / 3))
    den_hmin = z
    freq_res_hmin = num_hmin / den_hmin
    plt.plot(w, abs(freq_res_hmin))
    plt.grid()

    # H_ap
    plt.figure()
    w = np.linspace(0, 0.999 * 2 * np.pi, 1000)
    plt.xlabel("Frequency", fontsize=10)
    plt.ylabel("Amplitude", fontsize=10)
    z = np.exp(1j * w)
    plt.title('Frequency response of H_ap', fontsize=10)
    num_hap = -(2/3) * (z - (3 / 2))
    den_hap = z-(2/3)
    freq_res_hap = num_hap / den_hap
    plt.plot(w, abs(freq_res_hap))

    # o - H_correct
    plt.figure()
    w = np.linspace(0, 0.999 * 2 * np.pi, 1000)
    plt.xlabel("Frequency", fontsize=10)
    plt.ylabel("Amplitude", fontsize=10)
    z = np.exp(1j * w)
    plt.title('Frequency response of H_c', fontsize=10)
    num_hc = -3 * z
    den_hc = 1.5 * (3 * z - 2)
    freq_res_hc = num_hc / den_hc
    plt.plot(w, abs(freq_res_hc))

    # p
    plt.figure()
    num_hc_array = np.array([-3, 0])
    den_hc_array = np.array([1.5 * 3, -2 * 1.5])
    image_with_hc = scipy.signal.filtfilt(num_hc_array,  den_hc_array, y)
    filtered_image_hc = np.reshape(image_with_hc, [350, 350])
    plt.imshow(filtered_image_hc, cmap='gray')
    plt.title('image after H_c', fontsize=10)

    plt.show()


if __name__ == "__main__":
    main()
