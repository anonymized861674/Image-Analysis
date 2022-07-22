import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.interpolate import interp1d
from scipy.interpolate import LSQUnivariateSpline
from scipy.fftpack import rfft, irfft, fftfreq
from operator import itemgetter
from scipy.optimize import curve_fit
import copy
import math



def generate_results_directory(filename):
    """
    generates a directory in which to store the analysis results for the .tif file with name "filename"
    :param filename: name of .tif file to be analyzed
    :return: results directory
    """
    image = plt.imread(filename)

    path = os.getcwd()
    new_directory = path + '/' + filename[:-5] + '_results'
    try:
        os.mkdir(new_directory)
    except OSError:
        print('Creation of directory filed; likely already exists')
    stored_data = [new_directory, filename]
    np.save(new_directory + '/' + 'temporary_data', stored_data)
    return new_directory
    
   

def tif_image_fix(filename):
    """

    :param filename:
    :return:
    """
    image = plt.imread(filename)
    plt.imshow(image)
    # plt.show()
    xl = len(image[0, :])
    yl = len(image[:, 0])
    image_new = np.zeros((yl, xl, 3), dtype=float)
    for x in range(0, xl):
        for y in range(0, yl):
            maxval = int(image[y, x]) / 255

            # print(maxval)
            image_new[y, x, 0] = maxval
            image_new[y, x, 1] = maxval
            image_new[y, x, 2] = maxval
    plt.imshow(image_new)
    plt.show()
    plt.imsave(filename[:-4] + 'fix.tif', image_new)
    image = plt.imread(filename[:-4] + 'fix.tif')
    print(image[1, 1, :])
    plt.imshow(image_new)
    plt.show()


def read_temporary_data(new_directory):
    """
    loads the stored temporary data
    :param new_directory: the directory (including path) of the location of the temporary data
    :return:
    """
    temporary_data = np.load(new_directory + '/temporary_data.npy')
    return temporary_data


def scale_bar_px(file_name):
    """
    Measures the length of the scale bar assuming the scale bar is green.
    :param file_name: the file name of the .tif file given as a string.
    :return: returns the length, in pixels, of the scale bar.
    """
    print('Calculating scale bar length')
    image = plt.imread(file_name)
    first = len(image[0, :, 0])
    last = 0
    for i in range(0, len(image[0, :, 0])):
        for j in range(0, len(image[:, 0, 0])):
            if image[j, i, 1] == 255 and image[j, i, 0] == 0:
                if i < first:
                    first = i
                if i > last:
                    last = i
    print('Scale bar is ', last - first, ' pixels long')
    return last - first


def image_reduced_by_mouse(filename, show):
    """
        returns the writable reduced image from the specified tif file. Uses mouse inputs in order to define the
        reduced image's area
        :param directory: directory for storing resulting reduced image as a .jpg\
        :param show: bool control for displaying pyplot of reduced image
        :return: reduced image
    """
    image = plt.imread(filename)

    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)
    print('Image size: ', image.shape)

    print('Generating reduced image')
    plt.imsave(filename[:-5] + '_v000.tiff', image)
    image = plt.imread(filename[:-5] + '_v000.tiff')
    image = np.copy(image)
    red_image = np.copy(image)
    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_subplot(111)
    plt.imshow(red_image)
    plt.title('Please click the bounds of the desired area')
    plt.xlabel('x - axis (px)')
    plt.ylabel('y - axis (px)')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(24)
    for item in (ax.get_xticklabels()):
        item.set_fontsize(16)
    for item in (ax.get_yticklabels()):
        item.set_fontsize(16)
    point1 = fig.ginput(1, timeout = 0)
    point2 = fig.ginput(1, timeout = 0)
    plt.close(fig)
    xll = int(min(point1[0][1], point2[0][1]))
    xur = int(max(point1[0][1], point2[0][1]))
    yur = int(max(point1[0][0], point2[0][0]))
    yll = int(min(point1[0][0], point2[0][0]))
    red_image = red_image[yll:yur, xll:xur, :]
    red_image.setflags(write=1)
    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_subplot(111)
    plt.imshow(red_image)
    plt.title(filename[:-4] + ' original')
    plt.xlabel('x - axis (px)')
    plt.ylabel('y - axis (px)')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(24)
    for item in (ax.get_xticklabels()):
        item.set_fontsize(16)
    for item in (ax.get_yticklabels()):
        item.set_fontsize(16)
    # fig.savefig(directory + '/' + filename[0:-4] + '_original.tif', dpi=300)
    version_counter = 1
    
    if show:
        print('Displaying reduced image')
        plt.show()
    else:
        print('Reduced image was not displayed (show = 0)')
    plt.close(fig) 
    
    red_image = image[xll:xur, yll:yur, 0:3]   
    plt.imsave(filename[:-5] + '_v' + str(version_counter) + '.tiff', red_image)

    np.save('version_counter.npy', version_counter)
    plt.close(fig)
    os.chdir(path)
    return red_image
    

def image_reduced_by_mouse0p2(filename, show):
    """
        returns the writable reduced image from the specified tif file. Uses mouse inputs in order to define the
        reduced image's area
        :param directory: directory for storing resulting reduced image as a .jpg\
        :param show: bool control for displaying pyplot of reduced image
        :return: reduced image
    """
    image = plt.imread(filename)

    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)

    print('Generating reduced image')
    plt.imsave(filename[:-5] + '_v000.tiff', image)
    image = plt.imread(filename[:-5] + '_v000.tiff')
    image = copy.deepcopy(image) 
    red_image = image
    red_image.setflags(write=1)
    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_subplot(111)
    plt.imshow(red_image)
    plt.title('Please click the bounds of the desired area')
    plt.xlabel('x - axis (px)')
    plt.ylabel('y - axis (px)')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(24)
    for item in (ax.get_xticklabels()):
        item.set_fontsize(16)
    for item in (ax.get_yticklabels()):
        item.set_fontsize(16)
    point1 = fig.ginput(1, timeout = 0)
    point2 = fig.ginput(1, timeout = 0)
    plt.close(fig)
    xll = int(min(point1[0][0], point2[0][0]))
    xur = int(max(point1[0][0], point2[0][0]))
    yur = int(max(point1[0][1], point2[0][1]))
    yll = int(min(point1[0][1], point2[0][1]))
    red_image = red_image[yll:yur, xll:xur, :]
    red_image.setflags(write=1)
    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_subplot(111)
    plt.imshow(red_image)
    plt.title(filename[:-5] + ' original')
    plt.xlabel('x - axis (px)')
    plt.ylabel('y - axis (px)')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(24)
    for item in (ax.get_xticklabels()):
        item.set_fontsize(16)
    for item in (ax.get_yticklabels()):
        item.set_fontsize(16)
    # fig.savefig(directory + '/' + filename[0:-4] + '_original.tif', dpi=300)
    version_counter = 1
    red_image.copy(order='C') 
    plt.imsave(filename[:-4] + '_v' + str(version_counter) + '.tiff', red_image)
    np.save('version_counter.npy', version_counter)


    if show:
        print('Displaying reduced image')
        plt.show()
    else:
        print('Reduced image was not displayed (show = 0)')
    plt.close(fig)
    os.chdir(path)
    return red_image


def gaussian_smooth(filename, version_number, show):
    """
    1
    :param filename:
    :param version_number:
    :param show:
    :return:
    """

    path = os.getcwd()
    directory = path + '/' + filename[:-4] + '_results'
    os.chdir(directory)

    current_filename = filename[:-4] + '_v' + str(version_number) + '.tif'
    image = plt.imread(current_filename)

    version_counter = np.load('version_counter.npy')
    version_counter += 1
    np.save('version_counter.npy', version_counter)
    smoothed_filename = filename[:-4] + '_v' + str(version_counter) + '.tif'
    plt.imsave(smoothed_filename, image)
    image_smoothed = plt.imread(smoothed_filename)
    image_smoothed.setflags(write=1)

    lx = len(image[0, :, 0])
    ly = len(image[:, 0, 0])

    gaussian_width = 2
    gaussian_coeff = np.zeros((gaussian_width * 2 + 1, gaussian_width * 2 + 1))
    sigma = (gaussian_width + 1) / 2
    leading_coef = 1 / (sigma * np.sqrt(2 * np.pi)) * 100
    sum_coef = 0
    for i in range(0, len(gaussian_coeff[0, :])):
        for j in range(0, len(gaussian_coeff[:, 0])):
            dx = i - gaussian_width
            dy = j - gaussian_width
            val_ji = leading_coef
            val_ji *= np.e ** (-0.5 * (dx / sigma) ** 2)
            val_ji *= np.e ** (-0.5 * (dy / sigma) ** 2)
            gaussian_coeff[j, i] = val_ji
            sum_coef += val_ji

    for i in range(0, len(gaussian_coeff[0, :])):
        for j in range(0, len(gaussian_coeff[:, 0])):
            gaussian_coeff[j, i] *= 1 / sum_coef

    for i in range(gaussian_width, lx - gaussian_width):
        for j in range(gaussian_width, ly - gaussian_width):
            convolution = 0
            for di in range(0, 2 * gaussian_width + 1):
                for dj in range(0, 2 * gaussian_width + 1):
                    convolution += image[j - gaussian_width + dj, i - gaussian_width + di, 0] * gaussian_coeff[dj, di]
            image_smoothed[j, i, 0] = convolution
            image_smoothed[j, i, 1] = convolution
            image_smoothed[j, i, 2] = convolution
    if show:
        plt.imshow(image_smoothed)
        plt.show()

    plt.imsave(smoothed_filename, image_smoothed)

    os.chdir(path)


def image_smooth(filename, version_number, show):
    """
    1
    :param directory:
    :param filename:
    :param show:
    :return:
    """

    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)

    if version_number == 0:
        version_counter = np.load('version_counter.npy')
    else:
        version_counter = version_number

    current_filename = filename[:-5] + '_v' + str(version_counter) + '.tiff'
    image = plt.imread(current_filename)

    print('Smoothing Image')
    version_counter += 1
    np.save('version_counter.npy', version_counter)
    smoothed_filename = filename[:-5] + '_v' + str(version_counter) + '.tiff'
    plt.imsave(smoothed_filename, image)
    image_smoothed = plt.imread(smoothed_filename)
    image_smoothed = np.copy(image_smoothed) 
    lx = len(image[0, :, 0])
    ly = len(image[:, 0, 0])
    for i in range(1, lx - 1):
        for j in range(1, ly - 1):
            image_smoothed[j, i, 0] = np.mean(image[j - 1:j + 2, i - 1:i + 2, 0])
            image_smoothed[j, i, 1] = np.mean(image[j - 1:j + 2, i - 1:i + 2, 1])
            image_smoothed[j, i, 2] = np.mean(image[j - 1:j + 2, i - 1:i + 2, 2])
    plt.imsave(smoothed_filename, image_smoothed)

    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_subplot(111)
    plt.imshow(image_smoothed)
    plt.title(filename[:-5] + ' smoothed')
    plt.xlabel('x - axis (px)')
    plt.ylabel('y - axis (px)')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(24)
    for item in (ax.get_xticklabels()):
        item.set_fontsize(16)
    for item in (ax.get_yticklabels()):
        item.set_fontsize(16)

    if show:
        print('Displaying smoothed image')
        plt.show()
    else:
        print('Smoothed image was not displayed (show = 0)')
    plt.close(fig)
    os.chdir(path)
    

def image_smooth0p2(filename, version_number, show):
    """
    1
    :param directory:
    :param filename:
    :param show:
    :return:
    """

    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)

    if version_number == 0:
        version_counter = np.load('version_counter.npy')
    else:
        version_counter = version_number

    current_filename = filename[:-5] + '_v' + str(version_counter) + '.tiff'
    image = plt.imread(current_filename)

    print('Smoothing Image')
    version_counter += 1
    np.save('version_counter.npy', version_counter)
    smoothed_filename = filename[:-5] + '_v' + str(version_counter) + '.tiff'
    plt.imsave(smoothed_filename, image)
    image_smoothed = plt.imread(smoothed_filename)
    image_smoothed.setflags(write=1)
    lx = len(image[0, :, 0])
    ly = len(image[:, 0, 0])
    for i in range(1, lx - 1):
        for j in range(1, ly - 1):
            image_smoothed[j, i, 0] = np.mean(image[j - 1:j + 2, i - 1:i + 2, 0])
            image_smoothed[j, i, 1] = np.mean(image[j - 1:j + 2, i - 1:i + 2, 1])
            image_smoothed[j, i, 2] = np.mean(image[j - 1:j + 2, i - 1:i + 2, 2])
    plt.imsave(smoothed_filename, image_smoothed)

    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_subplot(111)
    plt.imshow(image_smoothed)
    plt.title(filename[:-4] + ' smoothed')
    plt.xlabel('x - axis (px)')
    plt.ylabel('y - axis (px)')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(24)
    for item in (ax.get_xticklabels()):
        item.set_fontsize(16)
    for item in (ax.get_yticklabels()):
        item.set_fontsize(16)

    if show:
        print('Displaying smoothed image')
        plt.show()
    else:
        print('Smoothed image was not displayed (show = 0)')
    plt.close(fig)
    os.chdir(path)
    
    
    
def image_smooth_1(filename, show):
    """
    1
    :param directory:
    :param filename:
    :param show:
    :return:
    """

    image = plt.imread(filename)

    print('Smoothing Image')
    smoothed_filename = filename[:-5] + '_s.tiff'
    plt.imsave(smoothed_filename, image)
    image_smoothed = plt.imread(smoothed_filename)
    image_smoothed = np.copy(image_smoothed) 
    
    lx = len(image[0, :, 0])
    ly = len(image[:, 0, 0])

    gaussian_width = 2
    gaussian_coeff = np.zeros((gaussian_width * 2 + 1, gaussian_width * 2 + 1))
    sigma = (gaussian_width + 1) / 2
    leading_coef = 1 / (sigma * np.sqrt(2 * np.pi)) * 100
    sum_coef = 0
    for i in range(0, len(gaussian_coeff[0, :])):
        for j in range(0, len(gaussian_coeff[:, 0])):
            dx = i - gaussian_width
            dy = j - gaussian_width
            val_ji = leading_coef
            val_ji *= np.e ** (-0.5 * (dx / sigma) ** 2)
            val_ji *= np.e ** (-0.5 * (dy / sigma) ** 2)
            gaussian_coeff[j, i] = val_ji
            sum_coef += val_ji

    for i in range(0, len(gaussian_coeff[0, :])):
        for j in range(0, len(gaussian_coeff[:, 0])):
            gaussian_coeff[j, i] *= 1 / sum_coef

    for i in range(gaussian_width, lx - gaussian_width):
        print(100 * i / lx) 
        for j in range(gaussian_width, ly - gaussian_width):
            convolution = 0
            for di in range(0, 2 * gaussian_width + 1):
                for dj in range(0, 2 * gaussian_width + 1):
                    convolution += image[j - gaussian_width + dj, i - gaussian_width + di, 0] * gaussian_coeff[dj, di]
            image_smoothed[j, i, 0] = convolution
            image_smoothed[j, i, 1] = convolution
            image_smoothed[j, i, 2] = convolution
    if show:
        plt.imshow(image_smoothed)
        plt.show()

    plt.imsave(smoothed_filename, image_smoothed)
    


def gauss_func(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        center = params[i]
        amp = params[i + 1]
        width = params[i + 2]
        y = y + amp * np.exp(-np.power(((x - center) / width),  2))
    return y


def lorentz_func(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        center = params[i]
        amp = params[i + 1]
        width = params[i + 2] / 10
        y = y + amp / (1 + ((x - center) / width) ** 2)
    return y


def first_derivative(x):
    f_1 = np.zeros_like(x)
    for i in range(1, len(f_1[1, :]) - 1):
        forward = x[1, i + 1] - x[1, i]
        backward = x[1, i] - x[1, i - 1]
        f_1[1, i] = (forward + backward) / 2
        f_1[0, i] = x[0, i]
    f_1[0, 0] = x[0, 0]
    f_1[1, 0] = f_1[1, 1]
    f_1[0, -1] = x[0, -1]
    f_1[1, -1] = f_1[1, -2]
    return f_1


def first_derivative_2(x):
    f_1 = np.zeros_like(x)
    for i in range(2, len(f_1[1, :]) - 2):
        forward = x[1, i + 1] - x[1, i]
        forward_2 = x[1, i + 2] - x[1, i + 1]
        backward = x[1, i] - x[1, i - 1]
        backward_2 = x[1, i - 1] - x[1, i - 2]
        f_1[1, i] = (forward + forward_2 + backward + backward_2) / 4
        f_1[0, i] = x[0, i]
    f_1[0, 1] = x[0, 1]
    f_1[1, 1] = f_1[1, 2]
    f_1[0, 0] = x[0, 0]
    f_1[1, 0] = f_1[1, 1]
    f_1[0, -2] = x[0, -2]
    f_1[1, -2] = f_1[1, -3]
    f_1[0, -1] = x[0, -1]
    f_1[1, -1] = f_1[1, -2]
    return f_1


def guesses(x):
    min_val = np.min(x[1, :])
    guess_vals = []
    catch_1 = 0
    catch_2 = 0
    peak_x = 0
    peak_val = 0
    for i in range(0, len(x[1, :])):

        if (x[1, i] < 0.02 * min_val) and catch_1 != 1:
            catch_1 = 1
            peak_val = x[1, i]
            peak_x = x[0, i]
        if (x[1, i] < 0) and catch_1 == 1 and x[1, i] < peak_val:
            peak_val = x[1, i]
            peak_x = x[0, i]
            catch_2 = 1
        if (x[1, i] > 0) and catch_2 == 1:
            catch_1 = 0
            catch_2 = 0
            guess_vals = np.concatenate((guess_vals, [peak_x]))
    return guess_vals


def brightness_cutoff_calc(x, brightness_percentile, peaks_skip, width, show):
    x_first_derivative = first_derivative_2(x)
    x_second_derivative = first_derivative_2(x_first_derivative)

    plt.plot(x[0, :], x[1, :])
    plt.plot(x_second_derivative[0, :], x_second_derivative[1, :])
    plt.show()

    guess_vals_0 = guesses(x_second_derivative)
    guesses_0 = []
    for i in range(0, len(x[0, :])):
        for guess_val in guess_vals_0:
            if np.abs((x[0, i] - guess_val)) < 0.1:
                guesses_0 += [guess_val, x[1, i], width]
    if len(guesses_0) == 3:
        guesses_0 = np.concatenate((guesses_0, guesses_0))
        guesses_0[0] = guesses_0[0] * 0.2
        guesses_0[1] = guesses_0[1] - 30
    print(guesses_0)
    popt, pcov = curve_fit(gauss_func, x[0, :], x[1, :], p0=guesses_0, maxfev = 14000)
    print(popt)
    fig = plt.figure(figsize=(16, 9), dpi=100)
    plt.bar(x[0, :], x[1, :])
    fit = gauss_func(x[0, :], *popt[int(3 * peaks_skip):])
    x[1, :] += - fit
    plt.bar(x[0, :], x[1, :])
    # plt.plot([152, 152], [0, np.max(x[1, :])], 'r-')
    if show:
        plt.show()
    else:
        plt.close()
    sum = np.sum(x[1, :])
    for i in range(0, len(x[1, :])):
        x[1, i] = x[1, i] / sum
    fig = plt.figure(figsize=(16, 9), dpi=100)
    plt.bar(x[0, :], x[1, :])

    partial_sum = 0
    i_partial_sum = int(0)
    while partial_sum <= brightness_percentile:
        partial_sum += x[1, i_partial_sum]
        i_partial_sum += 1
    plt.plot([i_partial_sum - 1, i_partial_sum - 1], [0, np.max(x[1, :])], 'r-')
    if show:
        plt.show()
    else:
        plt.close()
    print(i_partial_sum - 1)
    return x, i_partial_sum - 1


def dist_brightness(filename, version_number, fix, show):
    """
    Calculates the distribution of pixel brightnesses across the input image.
    For the UDNF SEM pixel brightnesses of 0 mod 9 are mapped to a brightness of 1 less. In other words, if a pixel has
    a brightness of 81, the actual reported brightness is 80 (because 81 is 0 mod 9). The distribution is corrected by
    taking counts from brightnesses of 8 mod 9 and giving them back to the next brightness until the distribution is
    smooth.
    :param image: pyplot image
    :param directory: directory for storing resulting brightness distribution as a .jpg
    :param filename: filename of the original .tif (string)
    :param show: bool control for displaying pyplot of reduced image
    :return:
    """
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)

    version_counter = version_number
    current_filename = filename[:-5] + '_v' + str(version_counter) + '.tiff'
    image = plt.imread(current_filename)

    print('Calculating brightness distribution')
    dist = np.zeros((2, 256))
    x = np.arange(0, 256)
    for i in range(0, len(image[0, :, 0])):
        for j in range(0, len(image[:, 0, 0])):
            brightness = int(image[j, i, 0])
            dist[1, brightness] += 1

    for i in range(0, len(dist[0, :])):
        dist[0, i] = x[i]
    # 'fix' the distribution
    if fix:
        for n in range(0, len(dist[0, :])):
            if n % 9 == 0:
                while dist[1, n] < (dist[1, n - 1] + dist[1, n + 1]) / 2:
                    dist[1, n - 1] -= 1
                    dist[1, n] += 1
    np.save(current_filename[:-5] + 'brightness.npy', dist)
    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_subplot(111)
    plt.bar(dist[0, :], dist[1, :])
    plt.xlabel('Pixel Brightness (0-255)')
    plt.ylabel('Counts')
    plt.title(filename[:-5] + ' Histogram of Pixel Brightness')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(24)
    for item in (ax.get_xticklabels()):
        item.set_fontsize(16)
    for item in (ax.get_yticklabels()):
        item.set_fontsize(16)
    print('Plotting brightness distribution')
    fig.savefig(directory + '/brightness_dist_' + current_filename, dpi=300)
    if show:
        plt.show()
    else:
        print('Brightness distribution not shown (show = 0)')
    plt.close(fig)
    os.chdir(path)
    return dist


def boolean_mapping_v2(filename, version_number, cutoff, boolean_val, show):
    """
    Maps pixel birghtness to boolean based on a brightness cutoff
    :param directory:
    :param filename:
    :param cutoff:
    :param boolean_val:
    :param show:
    :return:
    """
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)

    version_counter = version_number

    current_filename = filename[:-5] + '_v' + str(version_counter) + '.tiff'
    image = plt.imread(current_filename)
    version_counter = np.load('version_counter.npy')
    version_counter += 1
    np.save('version_counter.npy', version_counter)
    bool_image_filename = filename[:-5] + '_v' + str(version_counter) + '.tiff'
    plt.imsave(bool_image_filename, image)
    image = plt.imread(bool_image_filename)
    image = np.copy(image) 
    print('Generating boolean mapping')

    for i in range(0, len(image[0, :, 0])):
        for j in range(0, len(image[:, 0, 0])):
            if image[j, i, 0] > cutoff:
                image[j, i, 0] = int(255 * boolean_val)
                image[j, i, 1] = int(255 * boolean_val)
                image[j, i, 2] = int(255 * boolean_val)
            else:
                image[j, i, 0] = int(255 * (not boolean_val))
                image[j, i, 1] = int(255 * (not boolean_val))
                image[j, i, 2] = int(255 * (not boolean_val))

    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_subplot(111)
    plt.imshow(image)
    plt.title(filename[:-5] + 'Boolean mapping')
    plt.xlabel('x - axis (px)')
    plt.ylabel('y - axis (px)')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(24)
    for item in (ax.get_xticklabels()):
        item.set_fontsize(16)
    for item in (ax.get_yticklabels()):
        item.set_fontsize(16)
    fig.savefig(bool_image_filename, dpi=300)
    if show:
        print('Displaying boolean mapping')
        plt.show()
    else:
        print('Boolean mapping was not displayed (show = 0)')
    plt.close(fig)
    plt.imsave(bool_image_filename, image)
    os.chdir(path)


def object_search_single(path, filename, ignore_n, minimum_size, show):
    """
    Hole search algorithm used to find objects in a given boolean image.
    :param boolimage: boolean image array generated by boolean_mapping
    :param image: original reduced image
    :param ignore_n: ignore the n largest objects (likely the fill between actual holes)
    :param directory: directory in which to store the resulting image of the holes found by the algorithm
    :param filename: name of the original .tif file
    :param show: boolean control wheather or not to show the pyplot
    :return: the list of centroids x_n = points[n, 0] y_n = points[n, 1], the original image, the original boolean image
    """


    current_filename = filename
    image = plt.imread(current_filename)

    version_counter = 1
    version_counter += 1
    np.save('version_counter.npy', version_counter)
    new_image_file_name = filename[:-5] + '_v' + str(version_counter) + '.tiff'
    plt.imsave(new_image_file_name, image)

    image = plt.imread(new_image_file_name)
    image = np.copy(image) 
    print('Beginning object search')
    # edge points must be removed as a True value on the edge of the boolimage would cause the search algorithm to
    # call a element outside the boolimage
    lx = len(image[0, :, 0])
    ly = len(image[:, 0, 0])
    for i in range(0, lx):
        image[0, i, 0] = 255
        image[0, i, 1] = 255
        image[0, i, 2] = 255
        image[-1, i, 0] = 255
        image[-1, i, 1] = 255
        image[-1, i, 2] = 255
    for j in range(0, ly):
        image[j, 0, 0] = 255
        image[j, 0, 1] = 255
        image[j, 0, 2] = 255
        image[j, -1, 0] = 255
        image[j, -1, 1] = 255
        image[j, -1, 2] = 255
    image2 = np.copy(image)

    # algorithm for finding centroid of all holes in graph
    imagesize = len(image[0, :, 0]) * len(image[:, 0, 0])
    list1 = np.zeros((imagesize, 2))
    list2 = np.zeros((imagesize, 2))
    list3 = np.zeros((imagesize, 3))
    counter1 = 0
    counter2 = 0
    counter3 = 0
    for i in range(0, len(image[0, :, 0])):
        for j in range(0, len(image[:, 0, 0])):
            if image[j, i, 0] == 0:
                list1[counter1, 0] = i
                list1[counter1, 1] = j
                counter1 += 1
                image[j, i, 0] = 255
                image[j, i, 1] = 255
                image[j, i, 2] = 255
                list2[counter2, 0] = i
                list2[counter2, 1] = j
                counter2 += 1
                while int(counter2) != 0:
                    counter2 -= 1
                    icheck = int(list2[counter2, 0])
                    jcheck = int(list2[counter2, 1])
                    for i2 in range(icheck - 1, icheck + 2):
                        for j2 in range(jcheck - 1, jcheck + 2):
                            if image[j2, i2, 0] == 0:
                                list1[counter1, 0] = i2
                                list1[counter1, 1] = j2
                                counter1 += 1
                                list2[counter2, 0] = i2
                                list2[counter2, 1] = j2
                                counter2 += 1
                                image[j2, i2, 0] = 255
                                image[j2, i2, 1] = 255
                                image[j2, i2, 2] = 255
                list3[counter3, 0] = np.average(list1[0:counter1, 0]) # x position of centroid of object "counter3"
                list3[counter3, 1] = np.average(list1[0:counter1, 1]) # y position of centroid of object "counter3"
                list3[counter3, 2] = counter1
                counter3 += 1
                counter1 = 0
    # ignores the "ignore_n" largest objects
    for m in range(0, ignore_n):
        holesizemax = np.max(list3[0:counter3, 2])
        list4 = np.zeros((counter3, 3))
        counter4 = 0
        for n in range(0, counter3):
            if list3[n, 2] != holesizemax:
                list4[counter4, :] = list3[n, :]
                counter4 += 1
        counter3 -= 1
        list3 = list4

    # ignores all objects smaller the "minimum_size" in percentage
    if counter3 <= 0: 
        holesizemax = 999 
    else: 
        holesizemax = np.max(list3[0:counter3, 2])
    list4 = np.zeros((counter3, 3))
    counter4 = 0
    for n in range(0, counter3):
        if list3[n, 2] > minimum_size * holesizemax:
            list4[counter4, :] = list3[n, :]
            counter4 += 1
    list4_temp = list4[0:counter4, :]
    list4 = list4_temp
    if counter4 <= 0: 
        max_1 = 999
        min_1 = 0 
    else: 
        max_1 = np.max(list4[:counter4, 2])
        min_1 = np.min(list4[:counter4, 2])
    print('Hole search complete; generating image')
    for j in range(0, len(image[:, 0, 0])):
        for i in range(0, len(image[0, :, 0])):
            image[j, i, 0] = 255
            image[j, i, 1] = 255
            image[j, i, 2] = 255
    imagesize = len(image2[0, :, 0]) * len(image2[:, 0, 0])
    list1 = np.zeros((imagesize, 2))
    list2 = np.zeros((imagesize, 2))
    list3 = np.zeros((imagesize, 3))
    counter1 = 0
    counter2 = 0
    counter3 = 0
    for i in range(0, len(image2[0, :, 0])):
        for j in range(0, len(image2[:, 0])):
            if image2[j, i, 0] == 0:
                list1[counter1, 0] = i
                list1[counter1, 1] = j
                counter1 += 1
                image2[j, i, 0] = 255
                image2[j, i, 1] = 255
                image2[j, i, 2] = 255
                list2[counter2, 0] = i
                list2[counter2, 1] = j
                counter2 += 1
                while int(counter2) != 0:
                    counter2 -= 1
                    icheck = int(list2[counter2, 0])
                    jcheck = int(list2[counter2, 1])
                    for i2 in range(icheck - 1, icheck + 2):
                        for j2 in range(jcheck - 1, jcheck + 2):
                            if image2[j2, i2, 0] == 0:
                                list1[counter1, 0] = i2
                                list1[counter1, 1] = j2
                                counter1 += 1
                                list2[counter2, 0] = i2
                                list2[counter2, 1] = j2
                                counter2 += 1
                                image2[j2, i2, 0] = 255
                                image2[j2, i2, 1] = 255
                                image2[j2, i2, 2] = 255
                if not np.any(list1[0:counter1, 0] < 3) and not np.any(list1[0:counter1, 1] < 3) and not np.any(list1[0:counter1, 0] > lx - 3)and not np.any(list1[0:counter1, 1] > ly - 3):
                    if min_1 <= counter1 <= max_1:
                        list3[counter3, 0] = np.average(list1[0:counter1, 0])
                        list3[counter3, 1] = np.average(list1[0:counter1, 1])
                        list3[counter3, 2] = counter1
                        # re-place all points in hole
                        for n in range(0, len(list1[0:counter1, 0])):
                            i1 = int(list1[n, 0])
                            j1 = int(list1[n, 1])
                            image[j1, i1, 0] = 0
                            image[j1, i1, 1] = 0
                            image[j1, i1, 2] = 0
                        counter3 += 1
                counter1 = 0

    image2 = plt.imread(new_image_file_name)

    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_subplot(111)
    plt.imshow(image, alpha=0.7)
    plt.imshow(image2, alpha=0.3)
    plt.scatter(list3[0:counter3, 0], list3[0:counter3, 1], marker='.', linewidths=0.5, s=10)
    plt.title(filename[:-4] + 'Hole scatter plot')
    plt.xlabel('x - axis (px)')
    plt.ylabel('y - axis (px)')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(24)
    for item in (ax.get_xticklabels()):
        item.set_fontsize(16)
    for item in (ax.get_yticklabels()):
        item.set_fontsize(16)
    print(current_filename)
    version_counter += -1
    plt.imsave(current_filename, image2)
    version_counter += 1
    np.save('version_counter.npy', version_counter)
    new_image_file_name = filename[:-5] + '_v' + str(version_counter) + '.tiff'
    plt.imsave(new_image_file_name, image)
    version_counter += 1
    np.save('version_counter.npy', version_counter)
    new_image_file_name = filename[:-5] + '_v' + str(version_counter) + '.tiff'
    fig.savefig(new_image_file_name, dpi=600)
    if show:
        print('Displaying hole scatter plot')
        plt.show()
    else:
        print('Hole scatter plot not displayer (show = 0)')
    plt.close(fig)
    centroids = list3[0:counter3, :]
    np.save(filename[:-5] + 'centroids.npy', centroids)
    np.save(filename[:-5] + 'max_object_length.npy', max_1)
    os.chdir(path)


def object_search_v5(filename, version_number, ignore_n, minimum_size, show):
    """
    Hole search algorithm used to find objects in a given boolean image.
    :param boolimage: boolean image array generated by boolean_mapping
    :param image: original reduced image
    :param ignore_n: ignore the n largest objects (likely the fill between actual holes)
    :param directory: directory in which to store the resulting image of the holes found by the algorithm
    :param filename: name of the original .tif file
    :param show: boolean control wheather or not to show the pyplot
    :return: the list of centroids x_n = points[n, 0] y_n = points[n, 1], the original image, the original boolean image
    """

    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)

    if version_number == 0:
        version_counter = np.load('version_counter.npy')
    else:
        version_counter = version_number
    current_filename = filename[:-5] + '_v' + str(version_counter) + '.tiff'
    image = plt.imread(current_filename)

    version_counter += 1
    np.save('version_counter.npy', version_counter)
    new_image_file_name = filename[:-5] + '_v' + str(version_counter) + '.tiff'
    plt.imsave(new_image_file_name, image)

    image = plt.imread(new_image_file_name)
    image = np.copy(image) 
    print('Beginning object search')
    # edge points must be removed as a True value on the edge of the boolimage would cause the search algorithm to
    # call a element outside the boolimage
    lx = len(image[0, :, 0])
    ly = len(image[:, 0, 0])
    for i in range(0, lx):
        image[0, i, 0] = 255
        image[0, i, 1] = 255
        image[0, i, 2] = 255
        image[1, i, 0] = 255
        image[1, i, 1] = 255
        image[1, i, 2] = 255
        image[-1, i, 0] = 255
        image[-1, i, 1] = 255
        image[-1, i, 2] = 255
        image[-2, i, 0] = 255
        image[-2, i, 1] = 255
        image[-2, i, 2] = 255
    for j in range(0, ly):
        image[j, 0, 0] = 255
        image[j, 0, 1] = 255
        image[j, 0, 2] = 255
        image[j, 1, 0] = 255
        image[j, 1, 1] = 255
        image[j, 1, 2] = 255
        image[j, -1, 0] = 255
        image[j, -1, 1] = 255
        image[j, -1, 2] = 255
        image[j, -2, 0] = 255
        image[j, -2, 1] = 255
        image[j, -2, 2] = 255
    image2 = np.copy(image)

    # algorithm for finding centroid of all holes in graph
    imagesize = len(image[0, :, 0]) * len(image[:, 0, 0])
    list1 = np.zeros((imagesize, 2))
    list2 = np.zeros((imagesize, 2))
    list3 = np.zeros((imagesize, 3))
    counter1 = 0
    counter2 = 0
    counter3 = 0
    for i in range(0, len(image[0, :, 0])):
        for j in range(0, len(image[:, 0, 0])):
            if image[j, i, 0] == 0:
                list1[counter1, 0] = i
                list1[counter1, 1] = j
                counter1 += 1
                image[j, i, 0] = 255
                image[j, i, 1] = 255
                image[j, i, 2] = 255
                list2[counter2, 0] = i
                list2[counter2, 1] = j
                counter2 += 1
                while int(counter2) != 0:
                    counter2 -= 1
                    icheck = int(list2[counter2, 0])
                    jcheck = int(list2[counter2, 1])
                    for i2 in range(icheck - 1, icheck + 2):
                        for j2 in range(jcheck - 1, jcheck + 2):
                            # if i2 > 1267 or j2 > 1000:
                            #     plt.imshow(image)
                            #     # plt.show()
                            #     print(icheck, jcheck)
                            if image[j2, i2, 0] == 0:
                                list1[counter1, 0] = i2
                                list1[counter1, 1] = j2
                                counter1 += 1
                                list2[counter2, 0] = i2
                                list2[counter2, 1] = j2
                                counter2 += 1
                                image[j2, i2, 0] = 255
                                image[j2, i2, 1] = 255
                                image[j2, i2, 2] = 255
                list3[counter3, 0] = np.average(list1[0:counter1, 0]) # x position of centroid of object "counter3"
                list3[counter3, 1] = np.average(list1[0:counter1, 1]) # y position of centroid of object "counter3"
                list3[counter3, 2] = counter1
                counter3 += 1
                counter1 = 0
    # ignores the "ignore_n" largest objects
    for m in range(0, ignore_n):
        holesizemax = np.max(list3[0:counter3, 2])
        list4 = np.zeros((counter3, 3))
        counter4 = 0
        for n in range(0, counter3):
            if list3[n, 2] != holesizemax:
                list4[counter4, :] = list3[n, :]
                counter4 += 1
        counter3 -= 1
        list3 = list4

    # ignores all objects smaller the "minimum_size" in percentage
    holesizemax = np.max(list3[0:counter3, 2])
    list4 = np.zeros((counter3, 3))
    counter4 = 0
    for n in range(0, counter3):
        if list3[n, 2] > minimum_size * holesizemax:
            list4[counter4, :] = list3[n, :]
            counter4 += 1
    list4_temp = list4[0:counter4, :]
    list4 = list4_temp
    max_1 = np.max(list4[:counter4, 2])
    min_1 = np.min(list4[:counter4, 2])
    print('Hole search complete; generating image')
    for j in range(0, len(image[:, 0, 0])):
        for i in range(0, len(image[0, :, 0])):
            image[j, i, 0] = 255
            image[j, i, 1] = 255
            image[j, i, 2] = 255
    imagesize = len(image2[0, :, 0]) * len(image2[:, 0, 0])
    list1 = np.zeros((imagesize, 2))
    list2 = np.zeros((imagesize, 2))
    list3 = np.zeros((imagesize, 3))
    counter1 = 0
    counter2 = 0
    counter3 = 0
    for i in range(0, len(image2[0, :, 0])):
        for j in range(0, len(image2[:, 0])):
            if image2[j, i, 0] == 0:
                list1[counter1, 0] = i
                list1[counter1, 1] = j
                counter1 += 1
                image2[j, i, 0] = 255
                image2[j, i, 1] = 255
                image2[j, i, 2] = 255
                list2[counter2, 0] = i
                list2[counter2, 1] = j
                counter2 += 1
                while int(counter2) != 0:
                    counter2 -= 1
                    icheck = int(list2[counter2, 0])
                    jcheck = int(list2[counter2, 1])
                    for i2 in range(icheck - 1, icheck + 2):
                        for j2 in range(jcheck - 1, jcheck + 2):
                            if image2[j2, i2, 0] == 0:
                                list1[counter1, 0] = i2
                                list1[counter1, 1] = j2
                                counter1 += 1
                                list2[counter2, 0] = i2
                                list2[counter2, 1] = j2
                                counter2 += 1
                                image2[j2, i2, 0] = 255
                                image2[j2, i2, 1] = 255
                                image2[j2, i2, 2] = 255
                if not np.any(list1[0:counter1, 0] < 3) and not np.any(list1[0:counter1, 1] < 3) and not np.any(list1[0:counter1, 0] > lx - 3)and not np.any(list1[0:counter1, 1] > ly - 3):
                    if min_1 <= counter1 <= max_1:
                        list3[counter3, 0] = np.average(list1[0:counter1, 0])
                        list3[counter3, 1] = np.average(list1[0:counter1, 1])
                        list3[counter3, 2] = counter1
                        # re-place all points in hole
                        for n in range(0, len(list1[0:counter1, 0])):
                            i1 = int(list1[n, 0])
                            j1 = int(list1[n, 1])
                            image[j1, i1, 0] = 0
                            image[j1, i1, 1] = 0
                            image[j1, i1, 2] = 0
                        counter3 += 1
                counter1 = 0

    for n in range(0, counter3):
        y = int(list3[n, 1])
        x = int(list3[n, 0])
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                image[y + dy, x + dx, 0] = 0
                image[y + dy, x + dx, 1] = 0
                image[y + dy, x + dx, 2] = 0

    image2 = plt.imread(new_image_file_name)

    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_subplot(111)
    plt.imshow(image, alpha=0.7)
    plt.imshow(image2, alpha=0.3)
    plt.scatter(list3[0:counter3, 0], list3[0:counter3, 1], marker='.', linewidths=0.5, s=10)
    plt.title(filename[:-5] + 'Hole scatter plot')
    plt.xlabel('x - axis (px)')
    plt.ylabel('y - axis (px)')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(24)
    for item in (ax.get_xticklabels()):
        item.set_fontsize(16)
    for item in (ax.get_yticklabels()):
        item.set_fontsize(16)
    print(current_filename)
    version_counter += -1
    plt.imsave(current_filename, image2)
    version_counter += 1
    np.save('version_counter.npy', version_counter)
    new_image_file_name = filename[:-5] + '_v' + str(version_counter) + '.tiff'
    plt.imsave(new_image_file_name, image)
    version_counter += 1
    np.save('version_counter.npy', version_counter)
    new_image_file_name = filename[:-5] + '_v' + str(version_counter) + '.tiff'
    fig.savefig(new_image_file_name, dpi=600)
    if show:
        print('Displaying hole scatter plot')
        plt.show()
    else:
        print('Hole scatter plot not displayer (show = 0)')
    plt.close(fig)
    centroids = list3[0:counter3, :]
    np.save(filename[:-5] + 'centroids.npy', centroids)
    np.save(filename[:-5] + 'max_object_length.npy', max_1)
    os.chdir(path)


def object_points_list_v2(filename, version_number, show):
    """
    1
    :param centroids:
    :param boolimage_file:
    :param image_file:
    :param directory:
    :param filename:
    :param show:
    :return:
    """

    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)

    if version_number == 0:
        version_counter = np.load('version_counter.npy')
    else:
        version_counter = version_number
    current_filename = filename[:-5] + '_v' + str(version_counter) + '.tiff'
    image = plt.imread(current_filename)

    max_length_file = filename[:-5] + 'max_object_length.npy'
    centroids_file = filename[:-5] + 'centroids.npy'

    temporary_data = read_temporary_data(directory)
    os.chdir(temporary_data[0])
    max_length = int(np.load(max_length_file))
    print(max_length)

    plt.imsave(current_filename[:-5] + 'temp.tiff', image)
    version_counter += 1
    np.save('version_counter.npy', version_counter)
    plt.imsave(filename[:-5] + '_v' + str(version_counter) + '.tiff', image)
    image = plt.imread(current_filename[:-5] + 'temp.tiff')
    image = np.copy(image) 
    image_copy = np.copy(image)
    image_tosave = plt.imread(filename[:-5] + '_v' + str(version_counter) + '.tiff')
    image_tosave = np.copy(image_tosave) 
    image_tosave_xl = len(image_tosave[0, :, 0])
    image_tosave_yl = len(image_tosave[:, 0, 0])
    for x in range(0, image_tosave_xl):
        for y in range(0, image_tosave_yl):
            image_tosave[y, x, 0] = 255
            image_tosave[y, x, 1] = 255
            image_tosave[y, x, 2] = 255
    centroids = np.load(centroids_file)
    # I want to find all points in object with center in centroids. I can start individual object search algorithms and
    # give each point a flag of interior vs exterior based on the number of local points
    imagesize = len(image[0, :, 0]) * len(image[:, 0, 0])
    list1 = np.zeros((max_length, 2))
    list2 = np.zeros((max_length, 2))
    print(centroids)
    number_of_centroids = len(centroids[:, 0])
    print(number_of_centroids)

    points_list = np.empty((number_of_centroids, max_length + 1, 3))
    points_list[:, :, :] = np.nan
    for i in range(0, number_of_centroids):
        points_list[i, 0, 2] = centroids[i, 2]
    # points_list[n, d, i]: n-th object, d-th data point (1st reserved for object size), i holds x, y, point type (interior vs exterior)
    # ex. points_list[1, 5, 0] is x position of the 5th point in object 1
    # ex. points_list[1, 5, 1] is y position of the 5th point in object 1
    # ex. points_list[1, 5, 2] is the point type of the 5th point in object 1 (interior vs exterior)
    object_number = 0
    circular_results = np.zeros(len(centroids[:, 0]))
    for x, y in centroids[:, 0:2]:
        x = int(x)
        y = int(y)
        counter1 = 0
        counter2 = 0
        if image[y, x, 0] == 0:
            list1[counter1, 0] = x
            list1[counter1, 1] = y
            counter1 += 1
            image[y, x, 0] = 255
            image[y, x, 1] = 255
            image[y, x, 2] = 255
            list2[counter2, 0] = x
            list2[counter2, 1] = y
            counter2 += 1
            while int(counter2) != 0:
                counter2 -= 1
                icheck = int(list2[counter2, 0])
                jcheck = int(list2[counter2, 1])
                for i2 in range(icheck - 1, icheck + 2):
                    for j2 in range(jcheck - 1, jcheck + 2):
                        if image[j2, i2, 0] == 0:
                            list1[counter1, 0] = i2
                            list1[counter1, 1] = j2
                            counter1 += 1
                            list2[counter2, 0] = i2
                            list2[counter2, 1] = j2
                            counter2 += 1
                            image[j2, i2, 0] = 255
                            image[j2, i2, 1] = 255
                            image[j2, i2, 2] = 255

            point_number = 1
            for x, y in list1[0:counter1, 0:2]:
                points_list[object_number, point_number, 0] = x
                points_list[object_number, point_number, 1] = y
                bool_image_section = np.zeros((3,3))
                for i in range(0, len(bool_image_section[:, 0])):
                    for j in range(0, len(bool_image_section[0, :])):
                        xi = int(x-1+i)
                        yj = int(y-1+j)
                        if image_copy[yj, xi, 0]== 0:
                            bool_image_section[i, j] = 1
                        else:
                            bool_image_section[i, j] = 0
                neighbors = np.sum(bool_image_section)
                # print(neighbors)
                x = int(x)
                y = int(y)
                if neighbors < 8:
                    points_list[object_number, point_number, 2] = 0
                    image[y, x, 0] = 255
                    image[y, x, 1] = 0
                    image[y, x, 2] = 0
                    image_tosave[y, x, 0] = 0
                    image_tosave[y, x, 1] = 0
                    image_tosave[y, x, 2] = 0

                else:
                    points_list[object_number, point_number, 2] = 1
                    image[y, x, 0] = 0
                    image[y, x, 1] = 255
                    image[y, x, 2] = 0
                    image_tosave[y, x, 0] = 0
                    image_tosave[y, x, 1] = 0
                    image_tosave[y, x, 2] = 0
                point_number += 1
        circular_results[object_number] = np.sqrt((counter1 - 1) / np.pi)
        print('circularized radius=', np.sqrt((counter1 - 1) / np.pi))
        object_number += 1
    print(points_list[0, 0:5, :])
    print(points_list[1, 0:5, :])
    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_subplot(111)
    plt.imshow(image)
    plt.title(filename[:-5] + 'Hole scatter plot')
    plt.xlabel('x - axis (px)')
    plt.ylabel('y - axis (px)')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(24)
    for item in (ax.get_xticklabels()):
        item.set_fontsize(16)
    for item in (ax.get_yticklabels()):
        item.set_fontsize(16)
    plt.imsave(filename[:-5] + '_v' + str(version_counter) + '.tiff', image)
    fig.savefig(filename[:-5] + '_v' + str(version_counter) + 'point_labeled.tiff', dpi=600)
    if show:
        print('Displaying hole scatter plot')
        plt.show()
    else:
        print('Hole scatter plot not displayer (show = 0)')
    plt.close(fig)
    print(points_list[0, 0:5, :])
    print(points_list[1, 0:5, :])
    print(points_list[2, 0:5, :])
    print(points_list.shape)
    np.save(filename[:-5] + '_circular_results.npy', circular_results)
    np.save(filename[:-5] + '_object_points_list.npy', points_list)
    print('saved_points_list')
    plt.imsave(current_filename[:-5] + 'temp.tiff', image_tosave)
    os.chdir(path)


def pin_removal(filename, version_number, show):
    """

    :param filename:
    :param version_number_show:
    :return:
    """
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)

    if version_number == 0:
        version_counter = np.load('version_counter.npy')
    else:
        version_counter = version_number

    current_filename = filename[:-5] + '_v' + str(version_counter) + '.tiff'
    image = plt.imread(current_filename)
    image2 = np.copy(image)
    version_counter += 1
    np.save('version_counter.npy', version_counter)
    plt.imsave(filename[:-5] + '_v' + str(version_counter) + '.tiff', image)
    image = plt.imread(filename[:-5] + '_v' + str(version_counter) + '.tiff')
    image = np.copy(image) 

    for i in range(0, len(image[0, :, 0])):
        for j in range(0, len(image[:, 0, 0])):
            if image[j, i, 0] == 0:
                image[j, i, 0] = 255
                image[j, i, 1] = 255
                image[j, i, 2] = 255
            else:
                image[j, i, 0] = 0
                image[j, i, 1] = 0
                image[j, i, 2] = 0
    plt.imsave('pins_temp_v99.tiff', image)
    object_search_single(path, 'pins_temp_v99.tiff', 1, 0, 0)
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)

    image3 = plt.imread('pins_temp_v99_v2.tiff')

    for i in range(0, len(image2[0, :, 0])):
        for j in range(0, len(image2[:, 0, 0])):
            if image3[j, i, 0] == 0:
                image2[j, i, 0] = 0
                image2[j, i, 1] = 0
                image2[j, i, 2] = 0
    version_counter += 1
    np.save('version_counter.npy', version_counter)
    plt.imsave(filename[:-5] + '_v' + str(version_counter) + '.tiff', image2)
    os.chdir(path)


def object_boundry_analysis_v2(filename, version_number, show):
    """

    :param filename:
    :param version_number_show:
    :return:
    """
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)

    if version_number == 0:
        version_counter = np.load('version_counter.npy')
    else:
        version_counter = version_number

    current_filename = filename[:-5] + '_v' + str(version_counter) + '.tiff'
    image = plt.imread(current_filename)
    version_counter += 1
    np.save('version_counter.npy', version_counter)
    plt.imsave(filename[:-5] + '_v' + str(version_counter) + '.tiff', image)
    image = plt.imread(filename[:-5] + '_v' + str(version_counter) + '.tiff')
    image = np.copy(image) 

    version_counter += 1
    np.save('version_counter.npy', version_counter)
    plt.imsave(filename[:-5] + '_v' + str(version_counter) + '.tiff', image)
    image2 = plt.imread(filename[:-5] + '_v' + str(version_counter) + '.tiff')
    image2 = np.copy(image2) 

    object_data = np.load(filename[:-5] + '_object_points_list.npy')
    print(object_data[0, :, :])

    centroid_data = np.load(filename[:-5] + 'centroids.npy')
    # for n in range(0, len(object_data[:, 0, 0])):
    for n in range(0, len(object_data[:, 0, 0])):
        number_boundary = 0
        for i in range(0, len(object_data[0, 1:, 2])):
            if object_data[n, i, 2] == 0:
                number_boundary += 1
        boundary_points = np.zeros((number_boundary, 2))
        boundary_points_radial = np.zeros((number_boundary, 2))
        boundary_point_number = 0
        centroidx = centroid_data[n, 0]
        centroidy = centroid_data[n, 1]
        for i in range(0, len(object_data[0, 1:, 0])):
            if object_data[n, i, 2] == 0:
                boundary_points[boundary_point_number, 0] = object_data[n, i, 0]
                boundary_points[boundary_point_number, 1] = object_data[n, i, 1]
                dx = object_data[n, i, 0] - centroidx
                dy = object_data[n, i, 1] - centroidy
                boundary_points_radial[boundary_point_number, 0] = np.angle(dx + 1j * dy)
                boundary_points_radial[boundary_point_number, 1] = np.sqrt(dx * dx + dy * dy)
                boundary_point_number += 1
        print(boundary_points_radial[0:5, :])
        print(number_boundary)
        # move boundary points to -pi and pi
        print(np.argmin(boundary_points_radial[:, 0]))
        boundary_points_radial[np.argmin(boundary_points_radial[:, 0]), 0] = - np.pi
        boundary_points_radial[np.argmax(boundary_points_radial[:, 0]), 0] = + np.pi
        # print(np.argmax(boundary_points_radial[:, 0]))

        number_points = len(boundary_points_radial[:, 0])
        angle_linespace = np.linspace(-np.pi, np.pi, number_points * 20)
        # print(boundary_points_radial.shape)
        boundary_points_radial = np.array(sorted(boundary_points_radial, key=itemgetter(0)))
        boundary_points[:, 0] = np.cos(boundary_points_radial[:, 0]) * boundary_points_radial[:, 1]
        boundary_points[:, 1] = np.sin(boundary_points_radial[:, 0]) * boundary_points_radial[:, 1]
        # print(boundary_points_radial.shape)
        theta_points = boundary_points_radial[:, 0]
        radial_points = boundary_points_radial[:, 1]

        radial_function = interp1d(boundary_points_radial[:, 0], boundary_points_radial[:, 1])
        radial_interpolated = radial_function(angle_linespace)
        # plt.plot(angle_linespace + np.pi, radial_interpolated)
        # plt.show()
        W = fftfreq(angle_linespace.size, d = (angle_linespace[1] - angle_linespace[0]))
        f_signal = rfft(radial_interpolated)
        cut_f_signal = f_signal.copy()
        cut_f_signal[(W > 8)] = 0
        cut_signal = irfft(cut_f_signal)

        radial_function_new = interp1d(angle_linespace, cut_signal)
        angle_linspace_new = np.linspace(-np.pi, np.pi, number_points * 2)
        cut_signal_new = radial_function_new(angle_linspace_new)
        # plt.plot(angle_linespace, radial_interpolated)
        # plt.plot(angle_linespace, cut_signal)
        # plt.show()
        position_x_new = np.cos(angle_linspace_new) * cut_signal_new
        position_y_new = np.sin(angle_linspace_new) * cut_signal_new
        # plt.plot(boundary_points[:, 0], boundary_points[:, 1])
        # plt.plot(position_x_new, position_y_new)
        # plt.show()
        for i in range(0, len(position_x_new)):

            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 2] = 0

        cut_signal_new[:] += 1
        position_x_new = np.cos(angle_linspace_new) * cut_signal_new
        position_y_new = np.sin(angle_linspace_new) * cut_signal_new
        for i in range(0, len(position_x_new)):
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 2] = 0
        cut_signal_new[:] += 1
        position_x_new = np.cos(angle_linspace_new) * cut_signal_new
        position_y_new = np.sin(angle_linspace_new) * cut_signal_new
        for i in range(0, len(position_x_new)):
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 2] = 0
        cut_signal_new[:] += 1
        position_x_new = np.cos(angle_linspace_new) * cut_signal_new
        position_y_new = np.sin(angle_linspace_new) * cut_signal_new
        for i in range(0, len(position_x_new)):
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 2] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 2] = 0
        cut_signal_new[:] += 1
        position_x_new = np.cos(angle_linspace_new) * cut_signal_new
        position_y_new = np.sin(angle_linspace_new) * cut_signal_new
        for i in range(0, len(position_x_new)):
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 2] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 2] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx + 1), 0] = 255
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx + 1), 1] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx + 1), 2] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx + 1), 0] = 255
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx + 1), 1] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx + 1), 2] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx - 1), 0] = 255
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx - 1), 1] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx - 1), 2] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx - 1), 0] = 255
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx - 1), 1] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx - 1), 2] = 0



    plt.imsave(filename[:-5] + '_v' + str(version_counter) + '.tiff', image)

    for i in range(0, len(image2[0, :, 0])):
        for j in range(0, len(image2[:, 0, 0])):
            image2[j, i, 0] = 255
            image2[j, i, 1] = 255
            image2[j, i, 2] = 255
    for n in range(0, len(object_data[:, 0, 0])):
        imagesize = len(image[0, :, 0]) * len(image[:, 0, 0])
        list1 = np.zeros((imagesize, 2))
        list2 = np.zeros((imagesize, 2))
        centroidx = centroid_data[n, 0]
        centroidy = centroid_data[n, 1]
        x = int(centroidx)
        y = int(centroidy)
        counter1 = 0
        counter2 = 0
        list1[counter1, 0] = x
        list1[counter1, 1] = y
        counter1 += 1
        image2[y, x, 0] = 255
        image2[y, x, 1] = 255
        image2[y, x, 2] = 255
        list2[counter2, 0] = x
        list2[counter2, 1] = y
        counter2 += 1
        while int(counter2) != 0:
            counter2 -= 1
            icheck = int(list2[counter2, 0])
            jcheck = int(list2[counter2, 1])
            for i2 in range(icheck - 1, icheck + 2):
                for j2 in range(jcheck - 1, jcheck + 2):
                    # if i2 > 1267 or j2 > 1000:
                        # plt.imshow(image)
                        # plt.show()
                    if (image[j2, i2, 0] == 255 and image[j2, i2, 1] == 255 and image[j2, i2, 2] == 255) or (image[j2, i2, 0] == 0 and image[j2, i2, 1] == 0 and image[j2, i2, 2] == 0):
                        list1[counter1, 0] = i2
                        list1[counter1, 1] = j2
                        counter1 += 1
                        list2[counter2, 0] = i2
                        list2[counter2, 1] = j2
                        counter2 += 1
                        image2[j2, i2, 0] = 0
                        image2[j2, i2, 1] = 0
                        image2[j2, i2, 2] = 0
                        image[j2, i2, 0] = 255
                        image[j2, i2, 1] = 255
                        image[j2, i2, 2] = 0
                    elif image[j2, i2, 0] == 255 and image[j2, i2, 1] == 0 and image[j2, i2, 2] == 0:

                        # image2[j2, i2, 0] = 255
                        image2[j2, i2, 0] = 0
                        image2[j2, i2, 1] = 0
                        image2[j2, i2, 2] = 0
                        image[j2, i2, 0] = 255
                        image[j2, i2, 1] = 255
                        image[j2, i2, 2] = 0
    version_counter += 1
    print(filename[:-5] + '_v' + str(version_counter) + '.tiff')
    plt.imsave(filename[:-5] + '_v' + str(version_counter) + '.tiff', image2)
    os.chdir(path)


def visualization(filename, version_number, show):
    """

    :param filename:
    :param version_number_show:
    :return:
    """
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)

    if version_number == 0:
        version_counter = np.load('version_counter.npy')
    else:
        version_counter = version_number
        
    original_version = 2

    current_filename = filename[:-5] + '_v' + str(version_counter) + '.tiff'
    image = plt.imread(current_filename)
    version_counter += 1
    np.save('version_counter.npy', version_counter)
    plt.imsave(filename[:-5] + '_v' + str(version_counter) + '.tiff', image)
    image = plt.imread(filename[:-5] + '_v' + str(version_counter) + '.tiff')
    image = np.copy(image) 

    version_counter += 1
    np.save('version_counter.npy', version_counter)
    plt.imsave(filename[:-5] + '_v' + str(version_counter) + '.tiff', image)
    image2 = plt.imread(filename[:-5] + '_v' + str(version_counter) + '.tiff')
    image2 = np.copy(image2) 

    object_data = np.load(filename[:-5] + '_object_points_list.npy')
    print(object_data[0, :, :])

    centroid_data = np.load(filename[:-5] + 'centroids.npy')
    # for n in range(0, len(object_data[:, 0, 0])):
    for n in range(0, len(object_data[:, 0, 0])):
        number_boundary = 0
        for i in range(0, len(object_data[0, 1:, 2])):
            if object_data[n, i, 2] == 0:
                number_boundary += 1
        boundary_points = np.zeros((number_boundary, 2))
        boundary_points_radial = np.zeros((number_boundary, 2))
        boundary_point_number = 0
        centroidx = centroid_data[n, 0]
        centroidy = centroid_data[n, 1]
        for i in range(0, len(object_data[0, 1:, 0])):
            if object_data[n, i, 2] == 0:
                boundary_points[boundary_point_number, 0] = object_data[n, i, 0]
                boundary_points[boundary_point_number, 1] = object_data[n, i, 1]
                dx = object_data[n, i, 0] - centroidx
                dy = object_data[n, i, 1] - centroidy
                boundary_points_radial[boundary_point_number, 0] = np.angle(dx + 1j * dy)
                boundary_points_radial[boundary_point_number, 1] = np.sqrt(dx * dx + dy * dy)
                boundary_point_number += 1
        print(boundary_points_radial[0:5, :])
        print(number_boundary)
        # move boundary points to -pi and pi
        print(np.argmin(boundary_points_radial[:, 0]))
        boundary_points_radial[np.argmin(boundary_points_radial[:, 0]), 0] = - np.pi
        boundary_points_radial[np.argmax(boundary_points_radial[:, 0]), 0] = + np.pi
        # print(np.argmax(boundary_points_radial[:, 0]))

        number_points = len(boundary_points_radial[:, 0])
        angle_linespace = np.linspace(-np.pi, np.pi, number_points * 20)
        # print(boundary_points_radial.shape)
        boundary_points_radial = np.array(sorted(boundary_points_radial, key=itemgetter(0)))
        boundary_points[:, 0] = np.cos(boundary_points_radial[:, 0]) * boundary_points_radial[:, 1]
        boundary_points[:, 1] = np.sin(boundary_points_radial[:, 0]) * boundary_points_radial[:, 1]
        # print(boundary_points_radial.shape)
        theta_points = boundary_points_radial[:, 0]
        radial_points = boundary_points_radial[:, 1]

        radial_function = interp1d(boundary_points_radial[:, 0], boundary_points_radial[:, 1])
        radial_interpolated = radial_function(angle_linespace)
        # plt.plot(angle_linespace + np.pi, radial_interpolated)
        # plt.show()
        W = fftfreq(angle_linespace.size, d = (angle_linespace[1] - angle_linespace[0]))
        f_signal = rfft(radial_interpolated)
        cut_f_signal = f_signal.copy()
        cut_f_signal[(W > 8)] = 0
        cut_signal = irfft(cut_f_signal)

        radial_function_new = interp1d(angle_linespace, cut_signal)
        angle_linspace_new = np.linspace(-np.pi, np.pi, number_points * 2)
        cut_signal_new = radial_function_new(angle_linspace_new)
        # plt.plot(angle_linespace, radial_interpolated)
        # plt.plot(angle_linespace, cut_signal)
        # plt.show()
        position_x_new = np.cos(angle_linspace_new) * cut_signal_new
        position_y_new = np.sin(angle_linspace_new) * cut_signal_new
        # plt.plot(boundary_points[:, 0], boundary_points[:, 1])
        # plt.plot(position_x_new, position_y_new)
        # plt.show()
        for i in range(0, len(position_x_new)):

            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 2] = 0

        cut_signal_new[:] += 1
        position_x_new = np.cos(angle_linspace_new) * cut_signal_new
        position_y_new = np.sin(angle_linspace_new) * cut_signal_new
        for i in range(0, len(position_x_new)):
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 2] = 0
        cut_signal_new[:] += 1
        position_x_new = np.cos(angle_linspace_new) * cut_signal_new
        position_y_new = np.sin(angle_linspace_new) * cut_signal_new
        for i in range(0, len(position_x_new)):
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 2] = 0
        cut_signal_new[:] += 1
        position_x_new = np.cos(angle_linspace_new) * cut_signal_new
        position_y_new = np.sin(angle_linspace_new) * cut_signal_new
        for i in range(0, len(position_x_new)):
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 2] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 2] = 0
        cut_signal_new[:] += 1
        position_x_new = np.cos(angle_linspace_new) * cut_signal_new
        position_y_new = np.sin(angle_linspace_new) * cut_signal_new
        for i in range(0, len(position_x_new)):
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 2] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 2] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx + 1), 0] = 255
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx + 1), 1] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx + 1), 2] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx + 1), 0] = 255
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx + 1), 1] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx + 1), 2] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx - 1), 0] = 255
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx - 1), 1] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx - 1), 2] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx - 1), 0] = 255
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx - 1), 1] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx - 1), 2] = 0



    plt.imsave(filename[:-5] + '_v' + str(version_counter) + '.tiff', image)

    for i in range(0, len(image2[0, :, 0])):
        for j in range(0, len(image2[:, 0, 0])):
            image2[j, i, 0] = 255
            image2[j, i, 1] = 255
            image2[j, i, 2] = 255
    for n in range(0, len(object_data[:, 0, 0])):
        imagesize = len(image[0, :, 0]) * len(image[:, 0, 0])
        list1 = np.zeros((imagesize, 2))
        list2 = np.zeros((imagesize, 2))
        centroidx = centroid_data[n, 0]
        centroidy = centroid_data[n, 1]
        x = int(centroidx)
        y = int(centroidy)
        counter1 = 0
        counter2 = 0
        list1[counter1, 0] = x
        list1[counter1, 1] = y
        counter1 += 1
        image2[y, x, 0] = 255
        image2[y, x, 1] = 255
        image2[y, x, 2] = 255
        list2[counter2, 0] = x
        list2[counter2, 1] = y
        counter2 += 1
        while int(counter2) != 0:
            counter2 -= 1
            icheck = int(list2[counter2, 0])
            jcheck = int(list2[counter2, 1])
            for i2 in range(icheck - 1, icheck + 2):
                for j2 in range(jcheck - 1, jcheck + 2):
                    # if i2 > 1267 or j2 > 1000:
                        # plt.imshow(image)
                        # plt.show()
                    if (image[j2, i2, 0] == 255 and image[j2, i2, 1] == 255 and image[j2, i2, 2] == 255) or (image[j2, i2, 0] == 0 and image[j2, i2, 1] == 0 and image[j2, i2, 2] == 0):
                        list1[counter1, 0] = i2
                        list1[counter1, 1] = j2
                        counter1 += 1
                        list2[counter2, 0] = i2
                        list2[counter2, 1] = j2
                        counter2 += 1
                        image2[j2, i2, 0] = 0
                        image2[j2, i2, 1] = 0
                        image2[j2, i2, 2] = 0
                        image[j2, i2, 0] = 255
                        image[j2, i2, 1] = 255
                        image[j2, i2, 2] = 0
                    elif image[j2, i2, 0] == 255 and image[j2, i2, 1] == 0 and image[j2, i2, 2] == 0:

                        # image2[j2, i2, 0] = 255
                        image2[j2, i2, 0] = 0
                        image2[j2, i2, 1] = 0
                        image2[j2, i2, 2] = 0
                        image[j2, i2, 0] = 255
                        image[j2, i2, 1] = 255
                        image[j2, i2, 2] = 0
    version_counter += 1
    print(filename[:-5] + '_v' + str(version_counter) + '.tiff')
    plt.imsave(filename[:-5] + '_v' + str(version_counter) + '.tiff', image2)
    os.chdir(path)
    

def object_boundry_results_v1(filename, version_number, show):
    """

    :param filename:
    :param version_number_show:
    :return:
    """
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)

    if version_number == 0:
        version_counter = np.load('version_counter.npy')
    else:
        version_counter = version_number

    current_filename = filename[:-5] + '_v' + str(version_counter) + '.tiff'
    image = plt.imread(current_filename)
    version_counter += 1
    np.save('version_counter.npy', version_counter)
    plt.imsave(filename[:-5] + '_v' + str(version_counter) + '.tiff', image)
    image = plt.imread(filename[:-5] + '_v' + str(version_counter) + '.tiff')
    image = np.copy(image) 

    version_counter += 1
    np.save('version_counter.npy', version_counter)
    plt.imsave(filename[:-5] + '_v' + str(version_counter) + '.tiff', image)
    image2 = plt.imread(filename[:-5] + '_v' + str(version_counter) + '.tiff')
    image2 = np.copy(image2) 

    object_data = np.load(filename[:-5] + '_object_points_list.npy')
    print(object_data[0, :, :])

    centroid_data = np.load(filename[:-5] + 'centroids.npy')
    results = np.zeros((3, len(object_data[:, 0, 0])))
    # for n in range(0, len(object_data[:, 0, 0])):
    for n in range(0, len(object_data[:, 0, 0])):
        number_boundary = 0
        for i in range(0, len(object_data[0, 1:, 2])):
            if object_data[n, i, 2] == 0:
                number_boundary += 1
        boundary_points = np.zeros((number_boundary, 2))
        boundary_points_radial = np.zeros((number_boundary, 2))
        boundary_point_number = 0
        centroidx = centroid_data[n, 0]
        centroidy = centroid_data[n, 1]
        for i in range(0, len(object_data[0, 1:, 0])):
            if object_data[n, i, 2] == 0:
                boundary_points[boundary_point_number, 0] = object_data[n, i, 0]
                boundary_points[boundary_point_number, 1] = object_data[n, i, 1]
                dx = object_data[n, i, 0] - centroidx
                dy = object_data[n, i, 1] - centroidy
                boundary_points_radial[boundary_point_number, 0] = np.angle(dx + 1j * dy)
                boundary_points_radial[boundary_point_number, 1] = np.sqrt(dx * dx + dy * dy)
                boundary_point_number += 1
        print(boundary_points_radial[0:5, :])
        print(number_boundary)
        # move boundary points to -pi and pi
        print(np.argmin(boundary_points_radial[:, 0]))
        boundary_points_radial[np.argmin(boundary_points_radial[:, 0]), 0] = - np.pi
        boundary_points_radial[np.argmax(boundary_points_radial[:, 0]), 0] = + np.pi
        # print(np.argmax(boundary_points_radial[:, 0]))

        number_points = len(boundary_points_radial[:, 0])
        angle_linespace = np.linspace(-np.pi, np.pi, ((number_points * 20) // 2) * 2)
        # print(boundary_points_radial.shape)
        boundary_points_radial = np.array(sorted(boundary_points_radial, key=itemgetter(0)))
        boundary_points[:, 0] = np.cos(boundary_points_radial[:, 0]) * boundary_points_radial[:, 1]
        boundary_points[:, 1] = np.sin(boundary_points_radial[:, 0]) * boundary_points_radial[:, 1]
        # print(boundary_points_radial.shape)
        theta_points = boundary_points_radial[:, 0]
        radial_points = boundary_points_radial[:, 1]

        radial_function = interp1d(boundary_points_radial[:, 0], boundary_points_radial[:, 1])
        radial_interpolated = radial_function(angle_linespace)
        cross_section_interpolated = np.copy(radial_interpolated[0:int(len(radial_interpolated) / 2)])
        cross_section_interpolated += radial_interpolated[int(len(radial_interpolated) / 2):]
        cross_section_interpolated = cross_section_interpolated / 2

        for i in range(0, len(cross_section_interpolated)):
            print(cross_section_interpolated[i], (radial_interpolated[i] + radial_interpolated[i + int(len(radial_interpolated) / 2)]) / 2)
        # plt.plot(angle_linespace + np.pi, radial_interpolated)
        # plt.show()
        print(np.mean(radial_interpolated), np.max(radial_interpolated), np.min(radial_interpolated))
        print(np.mean(radial_interpolated), np.percentile(radial_interpolated, 90), np.percentile(radial_interpolated, 10))
        # results[0, n] = np.mean(radial_interpolated)
        # results[1, n] = np.percentile(radial_interpolated, 90)
        # results[2, n] = np.percentile(radial_interpolated, 10)

        results[0, n] = np.mean(cross_section_interpolated)
        results[1, n] = np.percentile(cross_section_interpolated, 90)
        results[2, n] = np.percentile(cross_section_interpolated, 10)


        W = fftfreq(angle_linespace.size, d=(angle_linespace[1] - angle_linespace[0]))
        f_signal = rfft(radial_interpolated)
        cut_f_signal = f_signal.copy()
        cut_f_signal[(W > 8)] = 0
        cut_signal = irfft(cut_f_signal)

        radial_function_new = interp1d(angle_linespace, cut_signal)
        angle_linspace_new = np.linspace(-np.pi, np.pi, number_points * 2)
        cut_signal_new = radial_function_new(angle_linspace_new)
        # plt.plot(angle_linespace, radial_interpolated)
        # plt.plot(angle_linespace, cut_signal)
        # plt.show()
        position_x_new = np.cos(angle_linspace_new) * cut_signal_new
        position_y_new = np.sin(angle_linspace_new) * cut_signal_new
        # plt.plot(boundary_points[:, 0], boundary_points[:, 1])
        # plt.plot(position_x_new, position_y_new)
        # plt.show()
        for i in range(0, len(position_x_new)):
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 2] = 0

        cut_signal_new[:] += 1
        position_x_new = np.cos(angle_linspace_new) * cut_signal_new
        position_y_new = np.sin(angle_linspace_new) * cut_signal_new
        for i in range(0, len(position_x_new)):
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 2] = 0
        cut_signal_new[:] += 1
        position_x_new = np.cos(angle_linspace_new) * cut_signal_new
        position_y_new = np.sin(angle_linspace_new) * cut_signal_new
        for i in range(0, len(position_x_new)):
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 2] = 0
        cut_signal_new[:] += 1
        position_x_new = np.cos(angle_linspace_new) * cut_signal_new
        position_y_new = np.sin(angle_linspace_new) * cut_signal_new
        for i in range(0, len(position_x_new)):
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 2] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 2] = 0
        cut_signal_new[:] += 1
        position_x_new = np.cos(angle_linspace_new) * cut_signal_new
        position_y_new = np.sin(angle_linspace_new) * cut_signal_new
        for i in range(0, len(position_x_new)):
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 2] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 2] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx + 1), 0] = 255
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx + 1), 1] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx + 1), 2] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx + 1), 0] = 255
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx + 1), 1] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx + 1), 2] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx - 1), 0] = 255
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx - 1), 1] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx - 1), 2] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx - 1), 0] = 255
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx - 1), 1] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx - 1), 2] = 0

    print(results[:, -1])
    print(np.mean(results[0, :]), np.mean(results[1, :]), np.mean(results[2, :]))

    plt.imsave(filename[:-5] + '_v' + str(version_counter) + '.tiff', image)

    for i in range(0, len(image2[0, :, 0])):
        for j in range(0, len(image2[:, 0, 0])):
            image2[j, i, 0] = 255
            image2[j, i, 1] = 255
            image2[j, i, 2] = 255
    for n in range(0, len(object_data[:, 0, 0])):
        imagesize = len(image[0, :, 0]) * len(image[:, 0, 0])
        list1 = np.zeros((imagesize, 2))
        list2 = np.zeros((imagesize, 2))
        centroidx = centroid_data[n, 0]
        centroidy = centroid_data[n, 1]
        x = int(centroidx)
        y = int(centroidy)
        counter1 = 0
        counter2 = 0
        list1[counter1, 0] = x
        list1[counter1, 1] = y
        counter1 += 1
        image2[y, x, 0] = 255
        image2[y, x, 1] = 255
        image2[y, x, 2] = 255
        list2[counter2, 0] = x
        list2[counter2, 1] = y
        counter2 += 1
        while int(counter2) != 0:
            counter2 -= 1
            icheck = int(list2[counter2, 0])
            jcheck = int(list2[counter2, 1])
            for i2 in range(icheck - 1, icheck + 2):
                for j2 in range(jcheck - 1, jcheck + 2):
                    # if i2 > 1267 or j2 > 1000:
                        # plt.imshow(image)
                        # plt.show()
                    if (image[j2, i2, 0] == 255 and image[j2, i2, 1] == 255 and image[j2, i2, 2] == 255) or (image[j2, i2, 0] == 0 and image[j2, i2, 1] == 0 and image[j2, i2, 2] == 0):
                        list1[counter1, 0] = i2
                        list1[counter1, 1] = j2
                        counter1 += 1
                        list2[counter2, 0] = i2
                        list2[counter2, 1] = j2
                        counter2 += 1
                        image2[j2, i2, 0] = 0
                        image2[j2, i2, 1] = 0
                        image2[j2, i2, 2] = 0
                        image[j2, i2, 0] = 255
                        image[j2, i2, 1] = 255
                        image[j2, i2, 2] = 0
                    elif image[j2, i2, 0] == 255 and image[j2, i2, 1] == 0 and image[j2, i2, 2] == 0:

                        image2[j2, i2, 0] = 255
                        # image2[j2, i2, 0] = 0
                        image2[j2, i2, 1] = 0
                        image2[j2, i2, 2] = 0
                        image[j2, i2, 0] = 255
                        image[j2, i2, 1] = 255
                        image[j2, i2, 2] = 0
    version_counter += 1
    print(filename[:-5] + '_v' + str(version_counter) + '.tiff')
    plt.imsave(filename[:-5] + '_v' + str(version_counter) + '.tiff', image2)
    os.chdir(path)
    return results


def object_boundry_results_v2(filename, version_number, show):
    """

    :param filename:
    :param version_number_show:
    :return:
    """
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)

    if version_number == 0:
        version_counter = np.load('version_counter.npy')
    else:
        version_counter = version_number

    current_filename = filename[:-5] + '_v' + str(version_counter) + '.tiff'
    image = plt.imread(current_filename)
    version_counter += 1
    np.save('version_counter.npy', version_counter)
    plt.imsave(filename[:-5] + '_v' + str(version_counter) + '.tiff', image)
    image = plt.imread(filename[:-5] + '_v' + str(version_counter) + '.tiff')
    image = np.copy(image) 

    version_counter += 1
    np.save('version_counter.npy', version_counter)
    plt.imsave(filename[:-5] + '_v' + str(version_counter) + '.tiff', image)
    image2 = plt.imread(filename[:-5] + '_v' + str(version_counter) + '.tiff')
    image2 = np.copy(image2) 

    object_data = np.load(filename[:-5] + '_object_points_list.npy')
    print(object_data[0, :, :])

    centroid_data = np.load(filename[:-5] + 'centroids.npy')
    results = np.zeros((5, len(object_data[:, 0, 0])))
    # for n in range(0, len(object_data[:, 0, 0])):
    for n in range(0, len(object_data[:, 0, 0])):
        number_boundary = 0
        for i in range(0, len(object_data[0, 1:, 2])):

            if object_data[n, i, 2] == 0:
                number_boundary += 1
        boundary_points = np.zeros((number_boundary, 2))
        boundary_points_radial = np.zeros((number_boundary, 2))
        boundary_point_number = 0
        centroidx = centroid_data[n, 0]
        centroidy = centroid_data[n, 1]
        for i in range(0, len(object_data[0, 1:, 0])):
            if object_data[n, i, 2] == 0:
                boundary_points[boundary_point_number, 0] = object_data[n, i, 0]
                boundary_points[boundary_point_number, 1] = object_data[n, i, 1]
                dx = object_data[n, i, 0] - centroidx
                dy = object_data[n, i, 1] - centroidy
                boundary_points_radial[boundary_point_number, 0] = np.angle(dx + 1j * dy)
                boundary_points_radial[boundary_point_number, 1] = np.sqrt(dx * dx + dy * dy)
                boundary_point_number += 1
        print(boundary_points_radial[0:5, :])
        print(number_boundary)
        # move boundary points to -pi and pi
        print(np.argmin(boundary_points_radial[:, 0]))
        boundary_points_radial[np.argmin(boundary_points_radial[:, 0]), 0] = - np.pi
        boundary_points_radial[np.argmax(boundary_points_radial[:, 0]), 0] = + np.pi
        # print(np.argmax(boundary_points_radial[:, 0]))

        number_points = len(boundary_points_radial[:, 0])
        angle_linespace = np.linspace(-np.pi, np.pi, ((number_points * 20) // 2) * 2)
        # print(boundary_points_radial.shape)
        boundary_points_radial = np.array(sorted(boundary_points_radial, key=itemgetter(0)))
        boundary_points[:, 0] = np.cos(boundary_points_radial[:, 0]) * boundary_points_radial[:, 1]
        boundary_points[:, 1] = np.sin(boundary_points_radial[:, 0]) * boundary_points_radial[:, 1]
        # print(boundary_points_radial.shape)
        theta_points = boundary_points_radial[:, 0]
        radial_points = boundary_points_radial[:, 1]

        radial_function = interp1d(boundary_points_radial[:, 0], boundary_points_radial[:, 1])
        radial_interpolated = radial_function(angle_linespace)
        cross_section_interpolated = np.copy(radial_interpolated[0:int(len(radial_interpolated) / 2)])
        cross_section_interpolated += radial_interpolated[int(len(radial_interpolated) / 2):]
        cross_section_interpolated = cross_section_interpolated / 2

        for i in range(0, len(cross_section_interpolated)):
            print(cross_section_interpolated[i], (radial_interpolated[i] + radial_interpolated[i + int(len(radial_interpolated) / 2)]) / 2)
        # plt.plot(angle_linespace + np.pi, radial_interpolated)
        # plt.show()
        print(np.mean(radial_interpolated), np.max(radial_interpolated), np.min(radial_interpolated))
        print(np.mean(radial_interpolated), np.percentile(radial_interpolated, 90), np.percentile(radial_interpolated, 10))
        # results[0, n] = np.mean(radial_interpolated)
        # results[1, n] = np.percentile(radial_interpolated, 90)
        # results[2, n] = np.percentile(radial_interpolated, 10)

        results[0, n] = np.mean(cross_section_interpolated)
        results[1, n] = np.percentile(cross_section_interpolated, 90)
        results[2, n] = np.percentile(cross_section_interpolated, 10)
        l_test = int(len(cross_section_interpolated) * 0.10 / 2)
        l_half = int(len(cross_section_interpolated) / 2) 
        # print(cross_section_interpolated[-l_test:], cross_section_interpolated[:l_test]) 
        # print((np.mean(cross_section_interpolated[-l_test:]) + np.mean(cross_section_interpolated[:l_test])) / 2)
        # plt.plot(angle_linespace[0:len(cross_section_interpolated)], cross_section_interpolated) 
        # plt.show() 
        results[3, n] = (np.mean(cross_section_interpolated[-l_test:]) + np.mean(cross_section_interpolated[:l_test])) / 2
        results[4, n] = np.mean(cross_section_interpolated[(l_half-l_test):(l_half+l_test)]) 
        

        W = fftfreq(angle_linespace.size, d=(angle_linespace[1] - angle_linespace[0]))
        f_signal = rfft(radial_interpolated)
        cut_f_signal = f_signal.copy()
        cut_f_signal[(W > 8)] = 0
        cut_signal = irfft(cut_f_signal)

        radial_function_new = interp1d(angle_linespace, cut_signal)
        angle_linspace_new = np.linspace(-np.pi, np.pi, number_points * 2)
        cut_signal_new = radial_function_new(angle_linspace_new)
        # plt.plot(angle_linespace, radial_interpolated)
        # plt.plot(angle_linespace, cut_signal)
        # plt.show()
        position_x_new = np.cos(angle_linspace_new) * cut_signal_new
        position_y_new = np.sin(angle_linspace_new) * cut_signal_new
        # plt.plot(boundary_points[:, 0], boundary_points[:, 1])
        # plt.plot(position_x_new, position_y_new)
        # plt.show()
        for i in range(0, len(position_x_new)):
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 2] = 0

        cut_signal_new[:] += 1
        position_x_new = np.cos(angle_linspace_new) * cut_signal_new
        position_y_new = np.sin(angle_linspace_new) * cut_signal_new
        for i in range(0, len(position_x_new)):
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 2] = 0
        cut_signal_new[:] += 1
        position_x_new = np.cos(angle_linspace_new) * cut_signal_new
        position_y_new = np.sin(angle_linspace_new) * cut_signal_new
        for i in range(0, len(position_x_new)):
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 2] = 0
        cut_signal_new[:] += 1
        position_x_new = np.cos(angle_linspace_new) * cut_signal_new
        position_y_new = np.sin(angle_linspace_new) * cut_signal_new
        for i in range(0, len(position_x_new)):
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 2] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 2] = 0
        cut_signal_new[:] += 1
        position_x_new = np.cos(angle_linspace_new) * cut_signal_new
        position_y_new = np.sin(angle_linspace_new) * cut_signal_new
        for i in range(0, len(position_x_new)):
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx), 2] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx - 1), 2] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx + 1), 2] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx + 1), 0] = 255
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx + 1), 1] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx + 1), 2] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx + 1), 0] = 255
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx + 1), 1] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx + 1), 2] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx - 1), 0] = 255
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx - 1), 1] = 0
            image[int(position_y_new[i] + centroidy - 1), int(position_x_new[i] + centroidx - 1), 2] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx - 1), 0] = 255
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx - 1), 1] = 0
            image[int(position_y_new[i] + centroidy + 1), int(position_x_new[i] + centroidx - 1), 2] = 0

    print(results[:, -1])
    print(np.mean(results[0, :]), np.mean(results[1, :]), np.mean(results[2, :]))

    plt.imsave(filename[:-5] + '_v' + str(version_counter) + '.tiff', image)

    for i in range(0, len(image2[0, :, 0])):
        for j in range(0, len(image2[:, 0, 0])):
            image2[j, i, 0] = 255
            image2[j, i, 1] = 255
            image2[j, i, 2] = 255
    for n in range(0, len(object_data[:, 0, 0])):
        imagesize = len(image[0, :, 0]) * len(image[:, 0, 0])
        list1 = np.zeros((imagesize, 2))
        list2 = np.zeros((imagesize, 2))
        centroidx = centroid_data[n, 0]
        centroidy = centroid_data[n, 1]
        x = int(centroidx)
        y = int(centroidy)
        counter1 = 0
        counter2 = 0
        list1[counter1, 0] = x
        list1[counter1, 1] = y
        counter1 += 1
        image2[y, x, 0] = 255
        image2[y, x, 1] = 255
        image2[y, x, 2] = 255
        list2[counter2, 0] = x
        list2[counter2, 1] = y
        counter2 += 1
        while int(counter2) != 0:
            counter2 -= 1
            icheck = int(list2[counter2, 0])
            jcheck = int(list2[counter2, 1])
            for i2 in range(icheck - 1, icheck + 2):
                for j2 in range(jcheck - 1, jcheck + 2):
                    # if i2 > 1267 or j2 > 1000:
                        # plt.imshow(image)
                        # plt.show()
                    if (image[j2, i2, 0] == 255 and image[j2, i2, 1] == 255 and image[j2, i2, 2] == 255) or (image[j2, i2, 0] == 0 and image[j2, i2, 1] == 0 and image[j2, i2, 2] == 0):
                        list1[counter1, 0] = i2
                        list1[counter1, 1] = j2
                        counter1 += 1
                        list2[counter2, 0] = i2
                        list2[counter2, 1] = j2
                        counter2 += 1
                        image2[j2, i2, 0] = 0
                        image2[j2, i2, 1] = 0
                        image2[j2, i2, 2] = 0
                        image[j2, i2, 0] = 255
                        image[j2, i2, 1] = 255
                        image[j2, i2, 2] = 0
                    elif image[j2, i2, 0] == 255 and image[j2, i2, 1] == 0 and image[j2, i2, 2] == 0:

                        image2[j2, i2, 0] = 255
                        # image2[j2, i2, 0] = 0
                        image2[j2, i2, 1] = 0
                        image2[j2, i2, 2] = 0
                        image[j2, i2, 0] = 255
                        image[j2, i2, 1] = 255
                        image[j2, i2, 2] = 0
    version_counter += 1
    print(filename[:-5] + '_v' + str(version_counter) + '.tiff')
    plt.imsave(filename[:-5] + '_v' + str(version_counter) + '.tiff', image2)
    os.chdir(path)
    
    return results
    

def visualazation_displacement_v1(filename, displacement, version_number, displacement_type, alpha, show):
    """

    :param filename:
    :param centroid: Object centdoid values 
    :return:
    """
    print("visualization_displacement_v1") 
    displacement_max = np.max(np.abs(displacement)) 
    
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    # print(displacement) 
    
    if version_number == 0:
        version_counter = np.load('version_counter.npy')
    else:
        version_counter = version_number
    
    centroid_data = np.load(filename[:-5] + 'centroids.npy')
    # print(centroid_data.shape, centroid_data[0:20, :])
    
    # original image
    image0 = plt.imread(filename[:-5] + '_v1.tiff')
    
    # image1 will be alpha overlay 
    plt.imsave(displacement_type + '.tiff', image0)
    image1 = plt.imread(displacement_type + '.tiff')
    image1 = np.copy(image0) 
    
    # image2 will be just displacement value 
    plt.imsave(displacement_type + '2.tiff', image0)
    image2 = plt.imread(displacement_type + '2.tiff')
    image2 = np.copy(image0) 
    
    # imageref is the boolean mapped image used to set the area to be colored in 
    imageref = plt.imread(filename[:-5] + '_v' + str(version_counter) + '.tiff') 
    print("Loaded all images") 
    # print(image0.shape, image1.shape, image2.shape, imageref.shape) 
    for j in range(0, len(image0[:, 0, 0])): 
        print(j / len(image0[:, 0, 0]) * 100) 
        for i in range(0, len(image0[0, :, 0])): 
            image2[j, i, 0] = 255 
            image2[j, i, 1] = 255 
            image2[j, i, 2] = 255 
            if imageref[j, i, 0] == 255 and imageref[j, i, 1] == 0: 
                image2[j, i, 0] = 0 
                image2[j, i, 1] = 0 
                image2[j, i, 2] = 0 
                
            if imageref[j, i, 0] == 0: 
                # print(centroid_data[:, 0].shape) 
                dist = np.zeros(shape = centroid_data[:, 0].shape)
                for n in range(0, len(dist)): 
                    # dist[n] = ((centroid_data[n, 0] - j) ** 2) + ((centroid_data[n, 1] - i) ** 2) * 0.0000001
                    dist[n] = math.dist([centroid_data[n, 1], centroid_data[n, 0]], [j, i])  
                # print(dist, np.argmin(dist), np.min(dist), centroid_data[np.argmin(dist), :]) 
                displacement_val = displacement[np.argmin(dist)] / displacement_max 
                displacement_mag = np.abs(displacement_val) 
                displacement_sign = displacement_val / displacement_mag
                if displacement_sign > 0: 
                    image2[j, i, 0] = 255
                    image2[j, i, 2] = int(255 - displacement_mag * 254) 
                    
                    image1[j, i, 0] = image0[j, i, 0] * (1 - alpha) + image2[j, i, 0] * alpha  
                else:
                    image2[j, i, 0] = int(255 - displacement_mag * 254) 
                    image2[j, i, 2] = 255 
                image2[j, i, 1] = int(255 - displacement_mag * 254)  
                
                
                image1[j, i, 0] = image0[j, i, 0] * (1 - alpha) + image2[j, i, 0] * alpha  
                image1[j, i, 1] = image0[j, i, 1] * (1 - alpha) + image2[j, i, 1] * alpha  
                image1[j, i, 2] = image0[j, i, 2] * (1 - alpha) + image2[j, i, 2] * alpha  
                # print(displacement_val, displacement_mag, displacement_sign) 
                # plt.imshow(image2)
                # plt.show()
                # image2[j, i, 0] = 255 
                # image2[j, i, 1] = 0 
                # image2[j, i, 2] = 0 
    print("Saving Images") 
    if show: 
        plt.imshow(image2)
        plt.show()
    print(displacement_type, ' scale max: ', displacement_max) 
    plt.imsave(displacement_type + '.tiff', image1)
    plt.imsave(displacement_type + '2.tiff', image2)
    
    
    
def visualazation_displacement_vtemp(filename, displacement, version_number, displacement_type, alpha, show, displacement_max):
    """

    :param filename:
    :param centroid: Object centdoid values 
    :return:
    """
    displacement_max = displacement_max
    
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    # print(displacement) 
    
    if version_number == 0:
        version_counter = np.load('version_counter.npy')
    else:
        version_counter = version_number
    
    centroid_data = np.load(filename[:-5] + 'centroids.npy')
    # print(centroid_data.shape, centroid_data[0:20, :])
    
    # original image
    image0 = plt.imread(filename[:-5] + '_v1.tiff')
    
    # image1 will be alpha overlay 
    plt.imsave(displacement_type + '.tiff', image0)
    image1 = plt.imread(displacement_type + '.tiff')
    image1 = np.copy(image0) 
    
    # image2 will be just displacement value 
    plt.imsave(displacement_type + '2.tiff', image0)
    image2 = plt.imread(displacement_type + '2.tiff')
    image2 = np.copy(image0) 
    
    # imageref is the boolean mapped image used to set the area to be colored in 
    imageref = plt.imread(filename[:-5] + '_v' + str(version_counter) + '.tiff') 
    
    # print(image0.shape, image1.shape, image2.shape, imageref.shape) 
    for j in range(0, len(image0[:, 0, 0])): 
        for i in range(0, len(image0[0, :, 0])): 
            image2[j, i, 0] = 255 
            image2[j, i, 1] = 255 
            image2[j, i, 2] = 255 
            if imageref[j, i, 0] == 255 and imageref[j, i, 1] == 0: 
                image2[j, i, 0] = 0 
                image2[j, i, 1] = 0 
                image2[j, i, 2] = 0 
                
            if imageref[j, i, 0] == 0: 
                # print(centroid_data[:, 0].shape) 
                dist = np.zeros(shape = centroid_data[:, 0].shape)
                for n in range(0, len(dist)): 
                    # dist[n] = ((centroid_data[n, 0] - j) ** 2) + ((centroid_data[n, 1] - i) ** 2) * 0.0000001
                    dist[n] = math.dist([centroid_data[n, 1], centroid_data[n, 0]], [j, i])  
                # print(dist, np.argmin(dist), np.min(dist), centroid_data[np.argmin(dist), :]) 
                displacement_val = displacement[np.argmin(dist)] / displacement_max 
                displacement_mag = np.abs(displacement_val) 
                displacement_sign = displacement_val / displacement_mag
                if displacement_sign > 0: 
                    image2[j, i, 0] = 255
                    image2[j, i, 2] = int(255 - displacement_mag * 254) 
                    
                    image1[j, i, 0] = image0[j, i, 0] * (1 - alpha) + image2[j, i, 0] * alpha  
                else:
                    image2[j, i, 0] = int(255 - displacement_mag * 254) 
                    image2[j, i, 2] = 255 
                image2[j, i, 1] = int(255 - displacement_mag * 254)  
                
                
                image1[j, i, 0] = image0[j, i, 0] * (1 - alpha) + image2[j, i, 0] * alpha  
                image1[j, i, 1] = image0[j, i, 1] * (1 - alpha) + image2[j, i, 1] * alpha  
                image1[j, i, 2] = image0[j, i, 2] * (1 - alpha) + image2[j, i, 2] * alpha  
                # print(displacement_val, displacement_mag, displacement_sign) 
                # plt.imshow(image2)
                # plt.show()
                # image2[j, i, 0] = 255 
                # image2[j, i, 1] = 0 
                # image2[j, i, 2] = 0 
    if show: 
        plt.imshow(image2)
        plt.show()
    print(displacement_type, ' scale max: ', displacement_max) 
    plt.imsave(displacement_type + 'adj.tiff', image1)
    plt.imsave(displacement_type + '2adj.tiff', image2)
    


def object_boundry_analysis_vunknown(filename, version_number, show):
    """

    :param filename:
    :param version_number_show:
    :return:
    """
    path = os.getcwd()
    directory = path + '/' + filename[:-4] + '_results'
    os.chdir(directory)

    if version_number == 0:
        version_counter = np.load('version_counter.npy')
    else:
        version_counter = version_number

    current_filename = filename[:-4] + '_v' + str(version_counter) + '.tif'
    image = plt.imread(current_filename)
    version_counter += 1
    np.save('version_counter.npy', version_counter)
    plt.imsave(filename[:-4] + '_v' + str(version_counter) + '.tif', image)
    image = plt.imread(filename[:-4] + '_v' + str(version_counter) + '.tif')
    image.setflags(write=1)

    object_data = np.load(filename[:-4] + '_object_points_list.npy')

    centroid_data = np.load(filename[:-4] + 'centroids.npy')
    # for n in range(0, len(object_data[:, 0, 0])):
    for n in range(0, len(object_data[:, 0, 0])):
        number_boundary = 0
        for i in range(0, len(object_data[0, 1:, 2])):
            if object_data[n, i, 2] == 0:
                number_boundary += 1
        boundary_points = np.zeros((number_boundary, 2))
        boundary_points_radial = np.zeros((number_boundary, 2))
        boundary_point_number = 0
        centroidx = centroid_data[n, 0]
        centroidy = centroid_data[n, 1]
        for i in range(0, len(object_data[0, 1:, 0])):
            if object_data[n, i, 2] == 0:
                boundary_points[boundary_point_number, 0] = object_data[n, i, 0]
                boundary_points[boundary_point_number, 1] = object_data[n, i, 1]
                dx = object_data[n, i, 0] - centroidx
                dy = object_data[n, i, 1] - centroidy
                boundary_points_radial[boundary_point_number, 0] = np.angle(dx + 1j * dy)
                boundary_points_radial[boundary_point_number, 1] = np.sqrt(dx * dx + dy * dy)
                boundary_point_number += 1
        print(boundary_points_radial[0:5, :])
        # move boundary points to -pi and pi
        # print(np.argmin(boundary_points_radial[:, 0]))
        boundary_points_radial[np.argmin(boundary_points_radial[:, 0]), 0] = - np.pi
        boundary_points_radial[np.argmax(boundary_points_radial[:, 0]), 0] = + np.pi
        # print(np.argmax(boundary_points_radial[:, 0]))

        number_points = len(boundary_points_radial[:, 0])
        angle_linespace = np.linspace(-np.pi, np.pi, number_points * 20)
        # print(boundary_points_radial.shape)
        boundary_points_radial = np.array(sorted(boundary_points_radial, key=itemgetter(0)))
        boundary_points[:, 0] = np.cos(boundary_points_radial[:, 0]) * boundary_points_radial[:, 1]
        boundary_points[:, 1] = np.sin(boundary_points_radial[:, 0]) * boundary_points_radial[:, 1]
        # print(boundary_points_radial.shape)
        theta_points = boundary_points_radial[:, 0]
        radial_points = boundary_points_radial[:, 1]

        radial_function = interp1d(boundary_points_radial[:, 0], boundary_points_radial[:, 1])
        radial_interpolated = radial_function(angle_linespace)
        # plt.plot(angle_linespace + np.pi, radial_interpolated)
        # plt.show()
        W = fftfreq(angle_linespace.size, d = (angle_linespace[1] - angle_linespace[0]))
        f_signal = rfft(radial_interpolated)
        cut_f_signal = f_signal.copy()
        cut_f_signal[(W > 8)] = 0
        cut_signal = irfft(cut_f_signal)

        radial_function_new = interp1d(angle_linespace, cut_signal)
        angle_linspace_new = np.linspace(-np.pi, np.pi, number_points)
        cut_signal_new = radial_function_new(angle_linspace_new)
        # plt.plot(angle_linespace, radial_interpolated)
        # plt.plot(angle_linespace, cut_signal)
        # plt.show()
        position_x_new = np.cos(angle_linspace_new) * cut_signal_new
        position_y_new = np.sin(angle_linspace_new) * cut_signal_new
        # plt.plot(boundary_points[:, 0], boundary_points[:, 1])
        # plt.plot(position_x_new, position_y_new)
        # plt.show()
        for i in range(0, len(position_x_new)):

            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 2] = 0

    plt.imsave(filename[:-4] + '_v' + str(version_counter) + '.tif', image)


def object_boundry_analysis(filename, version_number, show):
    """

    :param filename:
    :param version_number_show:
    :return:
    """
    path = os.getcwd()
    directory = path + '/' + filename[:-4] + '_results'
    os.chdir(directory)

    if version_number == 0:
        version_counter = np.load('version_counter.npy')
    else:
        version_counter = version_number

    current_filename = filename[:-4] + '_v' + str(version_counter) + '.tif'
    image = plt.imread(current_filename)
    version_counter += 1
    np.save('version_counter.npy', version_counter)
    plt.imsave(filename[:-4] + '_v' + str(version_counter) + '.tif', image)
    image = plt.imread(filename[:-4] + '_v' + str(version_counter) + '.tif')
    image.setflags(write=1)

    object_data = np.load(filename[:-4] + '_object_points_list.npy')

    centroid_data = np.load(filename[:-4] + 'centroids.npy')
    # for n in range(0, len(object_data[:, 0, 0])):
    for n in range(0, len(object_data[:, 0, 0])):
        number_boundary = 0
        for i in range(0, len(object_data[0, 1:, 2])):
            if object_data[n, i, 2] == 0:
                number_boundary += 1
        boundary_points = np.zeros((number_boundary, 2))
        boundary_points_radial = np.zeros((number_boundary, 2))
        boundary_point_number = 0
        centroidx = centroid_data[n, 0]
        centroidy = centroid_data[n, 1]
        for i in range(0, len(object_data[0, 1:, 0])):
            if object_data[n, i, 2] == 0:
                boundary_points[boundary_point_number, 0] = object_data[n, i, 0]
                boundary_points[boundary_point_number, 1] = object_data[n, i, 1]
                dx = object_data[n, i, 0] - centroidx
                dy = object_data[n, i, 1] - centroidy
                boundary_points_radial[boundary_point_number, 0] = np.angle(dx + 1j * dy)
                boundary_points_radial[boundary_point_number, 1] = np.sqrt(dx * dx + dy * dy)
                boundary_point_number += 1
        print(boundary_points_radial[0:5, :])
        # move boundary points to -pi and pi
        # print(np.argmin(boundary_points_radial[:, 0]))
        boundary_points_radial[np.argmin(boundary_points_radial[:, 0]), 0] = - np.pi
        boundary_points_radial[np.argmax(boundary_points_radial[:, 0]), 0] = + np.pi
        # print(np.argmax(boundary_points_radial[:, 0]))

        number_points = len(boundary_points_radial[:, 0])
        angle_linespace = np.linspace(-np.pi, np.pi, number_points * 20)
        # print(boundary_points_radial.shape)
        boundary_points_radial = np.array(sorted(boundary_points_radial, key=itemgetter(0)))
        boundary_points[:, 0] = np.cos(boundary_points_radial[:, 0]) * boundary_points_radial[:, 1]
        boundary_points[:, 1] = np.sin(boundary_points_radial[:, 0]) * boundary_points_radial[:, 1]
        # print(boundary_points_radial.shape)
        theta_points = boundary_points_radial[:, 0]
        radial_points = boundary_points_radial[:, 1]

        radial_function = interp1d(boundary_points_radial[:, 0], boundary_points_radial[:, 1])
        radial_interpolated = radial_function(angle_linespace)
        # plt.plot(angle_linespace + np.pi, radial_interpolated)
        # plt.show()
        W = fftfreq(angle_linespace.size, d = (angle_linespace[1] - angle_linespace[0]))
        f_signal = rfft(radial_interpolated)
        cut_f_signal = f_signal.copy()
        cut_f_signal[(W > 8)] = 0
        cut_signal = irfft(cut_f_signal)

        radial_function_new = interp1d(angle_linespace, cut_signal)
        angle_linspace_new = np.linspace(-np.pi, np.pi, number_points)
        cut_signal_new = radial_function_new(angle_linspace_new)
        # plt.plot(angle_linespace, radial_interpolated)
        # plt.plot(angle_linespace, cut_signal)
        # plt.show()
        position_x_new = np.cos(angle_linspace_new) * cut_signal_new
        position_y_new = np.sin(angle_linspace_new) * cut_signal_new
        # plt.plot(boundary_points[:, 0], boundary_points[:, 1])
        # plt.plot(position_x_new, position_y_new)
        # plt.show()
        for i in range(0, len(position_x_new)):

            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 0] = 255
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 1] = 0
            image[int(position_y_new[i] + centroidy), int(position_x_new[i] + centroidx), 2] = 0

    plt.imsave(filename[:-4] + '_v' + str(version_counter) + '.tif', image)

def remove_objects(filename, version_number, show):
    """

    :param boolimage_file:
    :param image_file:
    :param directory:
    :param filename:
    :param show:
    :return:
    """

    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)

    if version_number == 0:
        version_counter = np.load('version_counter.npy')
    else:
        version_counter = version_number
    current_filename = filename[:-5] + '_v' + str(version_counter) + '.tiff'
    image = plt.imread(current_filename)

    version_counter += 1
    np.save('version_counter.npy', version_counter)

    plt.imsave(filename[:-5] + '_v' + str(version_counter) + '.tiff', image)
    image = plt.imread(filename[:-5] + '_v' + str(version_counter) + '.tiff')
    image = np.copy(image) 

    # edge points must be removed as a True value on the edge of the boolimage would cause the search algorithm to
    # call a element outside the boolimage
    lx = len(image[0, :, 0])
    ly = len(image[:, 0, 0])
    for i in range(0, lx):
        image[0, i, 0] = 255
        image[0, i, 1] = 255
        image[0, i, 2] = 255
        image[-1, i, 0] = 255
        image[-1, i, 1] = 255
        image[-1, i, 2] = 255
    for j in range(0, ly):
        image[j, 0, 0] = 255
        image[j, 0, 1] = 255
        image[j, 0, 2] = 255
        image[j, -1, 0] = 255
        image[j, -1, 1] = 255
        image[j, -1, 2] = 255

    imagesize = len(image[0, :, 0]) * len(image[:, 0, 0])
    list1 = np.zeros((imagesize, 2))
    list2 = np.zeros((imagesize, 2))


    remove_flag = True
    while remove_flag:
        fig = plt.figure(figsize=(16, 9), dpi=100)
        ax = fig.add_subplot(111)
        plt.imshow(image)
        plt.title('Please click the bounds of the desired area')
        plt.xlabel('x - axis (px)')
        plt.ylabel('y - axis (px)')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
            item.set_fontsize(24)
        for item in (ax.get_xticklabels()):
            item.set_fontsize(16)
        for item in (ax.get_yticklabels()):
            item.set_fontsize(16)
        plt.imshow(image)
        point = fig.ginput(1, timeout = 0)
        y_point = int(point[0][1])
        x_point = int(point[0][0])
        remove_flag = not bool(image[y_point, x_point, 0])
        print(x_point, y_point, remove_flag, image[y_point, x_point, 0])
        counter1 = 0
        counter2 = 0
        if remove_flag:
            list1[counter1, 0] = x_point
            list1[counter1, 1] = y_point
            counter1 += 1
            image[y_point, x_point, 0] = 255
            image[y_point, x_point, 1] = 255
            image[y_point, x_point, 2] = 255
            list2[counter2, 0] = x_point
            print(counter2)
            list2[counter2, 1] = y_point
            counter2 += 1
            while int(counter2) != 0:
                counter2 -= 1
                icheck = int(list2[counter2, 0])
                jcheck = int(list2[counter2, 1])
                for i2 in range(icheck - 1, icheck + 2):
                    for j2 in range(jcheck - 1, jcheck + 2):
                        if image[j2, i2, 0] == 0:
                            list1[counter1, 0] = i2
                            list1[counter1, 1] = j2
                            counter1 += 1
                            list2[counter2, 0] = i2
                            list2[counter2, 1] = j2
                            counter2 += 1
                            image[j2, i2, 0] = 255
                            image[j2, i2, 1] = 255
                            image[j2, i2, 2] = 255
        plt.close(fig)

    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_subplot(111)
    plt.imshow(image)
    plt.title(filename[:-4] + 'Hole scatter plot')
    plt.xlabel('x - axis (px)')
    plt.ylabel('y - axis (px)')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(24)
    for item in (ax.get_xticklabels()):
        item.set_fontsize(16)
    for item in (ax.get_yticklabels()):
        item.set_fontsize(16)
    # fig.savefig(directory + '/' + filename[:-4] + 'point_labeled', dpi=600)
    if show:
        print('Displaying hole scatter plot')
        plt.show()
    else:
        print('Hole scatter plot not displayer (show = 0)')
    version_counter += 1
    np.save('version_counter.npy', version_counter)
    print(filename[:-5] + '_v' + str(version_counter) + '.tiff')
    plt.imsave(filename[:-5] + '_v' + str(version_counter) + '.tiff', image)
    plt.close(fig)
    os.chdir(path)
    

def centroid_analysis_v0(filename): 
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)
    
    a_final = 0 
    stdx_final = 0 
    stdy_final = 0 
    
    centroid_data = np.load(filename[:-5] + 'centroids.npy')

    fig = plt.figure(figsize=(16, 9), dpi=100)
    plt.scatter(centroid_data[:, 0], centroid_data[:, 1]) 
    plt.title('Click two lattice points along an axis')
    plt.xlabel('x - axis (px)')
    plt.ylabel('y - axis (px)')
   
    point1 = fig.ginput(1, timeout = 0)[0]
    point2 = fig.ginput(1, timeout = 0)[0]
    plt.close(fig) 

    n_a = float(int(input("How many latice Spacings?")))
    
    v0 = [(float(point2[0]) - float(point1[0]))/n_a, (float(point2[1]) - float(point1[1]))/n_a] 
    x_cen = centroid_data[0, 0] 
    y_cen = centroid_data[0, 1] 
    
    print(cost_centroid(centroid_data, v0, x_cen, y_cen)) 
    
    print(cost_centroid(centroid_data, v0, x_cen, y_cen)) 
    delta = 0.6
    scale_coef = 0.9
    n_tests = 200   # 1000 
    for scale in range(-20, 30): 
        scale0 = scale
        if scale0 <= 0: 
            scale0 = 1 
        print(v0, [x_cen, y_cen]) 
        a = (v0[0]**2+v0[1]**2)**(0.5)
        v0x = np.linspace(v0[0]*(1-delta**(scale_coef*scale0)), v0[0]*(1+delta**(scale_coef*scale0)), n_tests)
        print((1-delta**(scale_coef*scale0)), (1+delta**(scale_coef*scale0)))
        v0x_cost = np.zeros(shape = np.shape(v0x)) 
        for i in range(0, len(v0x)): 
            v0_temp = [v0x[i], v0[1]]
            # print(v0, v0_temp) 
            cost_i, stdx, stdy = cost_centroid(centroid_data, v0_temp, x_cen, y_cen) 
            v0x_cost[i] = cost_i 
        # plt.scatter(v0x, v0x_cost) 
        # plt.show() 
        v0[0] = v0x[np.argmin(v0x_cost)] 
        # print(v0[0]) 
        
        print(v0, [x_cen, y_cen], np.min(v0x_cost)) 
        
        v0y = np.linspace(v0[1]*(1-delta**(scale_coef*scale0)), v0[1]*(1+delta**(scale_coef*scale0)), n_tests)
        v0y_cost = np.zeros(shape = np.shape(v0y)) 
        for i in range(0, len(v0y)): 
            v0_temp = [v0[0], v0y[i]]
            cost_i, stdx, stdy = cost_centroid(centroid_data, v0_temp, x_cen, y_cen) 
            v0y_cost[i] = cost_i 
        # plt.scatter(v0y, v0y_cost) 
        # plt.show() 
        v0[1] = v0y[np.argmin(v0y_cost)] 
        
        print(v0, [x_cen, y_cen], np.min(v0y_cost)) 
        
        x_cen_vals = np.linspace(x_cen-a*delta**(scale_coef*scale0), x_cen+a*delta**(scale_coef*scale0), n_tests)
        x_cen_costs = np.zeros(shape = np.shape(x_cen_vals)) 
        for i in range(0, len(x_cen_vals)): 
            x_cen_temp = x_cen_vals[i] 
            cost_i, stdx, stdy = cost_centroid(centroid_data, v0, x_cen_temp, y_cen) 
            x_cen_costs[i] = cost_i 
        x_cen = x_cen_vals[np.argmin(x_cen_costs)]
        
        
        print(v0, [x_cen, y_cen], np.min(x_cen_costs)) 
        
        y_cen_vals = np.linspace(y_cen-a*delta**(scale_coef*scale0), y_cen+a*delta**(scale_coef*scale0), n_tests)
        y_cen_costs = np.zeros(shape = np.shape(y_cen_vals)) 
        for i in range(0, len(y_cen_vals)): 
            y_cen_temp = y_cen_vals[i] 
            cost_i, stdx, stdy = cost_centroid(centroid_data, v0, x_cen, y_cen_temp) 
            y_cen_costs[i] = cost_i 
        y_cen = y_cen_vals[np.argmin(y_cen_costs)]
        
        
        print(v0, [x_cen, y_cen], np.min(y_cen_costs)) 
        a_final = (v0[0]**2+v0[1]**2)**(0.5)
        stdx_final = stdx
        stdy_final = stdy 
        
        
        
        
        
        
        
        
        
    
    
    # input("Press Enter...") 
    os.chdir(path)
    return a_final, stdx_final, stdy_final 
    

def centroid_analysis_v1(filename): 
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)
    a = np.mgrid[:5, :5][0]
    print(a) 
    print(np.fft.rfft2(a))
    a_final = 0 
    stdx_final = 0 
    stdy_final = 0 
    
    centroid_data = np.load(filename[:-5] + 'centroids.npy')

    fig = plt.figure(figsize=(16, 9), dpi=100)
    plt.scatter(centroid_data[:, 0], centroid_data[:, 1]) 
    plt.title('Click two lattice points along an axis')
    plt.xlabel('x - axis (px)')
    plt.ylabel('y - axis (px)')
   
    point1 = fig.ginput(1, timeout = 0)[0]
    point2 = fig.ginput(1, timeout = 0)[0]
    plt.close(fig) 

    n_a = float(int(input("How many latice Spacings?")))
    
    v0 = [(float(point2[0]) - float(point1[0]))/n_a, (float(point2[1]) - float(point1[1]))/n_a] 
    x_cen = centroid_data[0, 0] 
    y_cen = centroid_data[0, 1] 
    
    print(cost_centroid(centroid_data, v0, x_cen, y_cen)) 
    
    print(cost_centroid(centroid_data, v0, x_cen, y_cen)) 
    delta = 0.3
    scale_coef = 0.85
    n_tests = 500   # 1000 
    for scale in range(-20, 50): 
        scale0 = scale
        if scale0 <= 0: 
            scale0 = 1 
        # print(v0, [x_cen, y_cen]) 
        a = (v0[0]**2+v0[1]**2)**(0.5)
        v0x = np.linspace(v0[0]*(1-delta*scale_coef**(scale0)), v0[0]*(1+delta*scale_coef**(scale0)), n_tests)
        print(2 * v0x[0] / (v0x[0] + v0x[-1]), 2 * v0x[-1] / (v0x[0] + v0x[-1]))
        v0x_cost = np.zeros(shape = np.shape(v0x)) 
        for i in range(0, len(v0x)): 
            v0_temp = [v0x[i], v0[1]]
            # print(v0, v0_temp) 
            cost_i, stdx, stdy = cost_centroid(centroid_data, v0_temp, x_cen, y_cen) 
            v0x_cost[i] = cost_i 
        # plt.scatter(v0x, v0x_cost) 
        # plt.show() 
        v0[0] = v0x[np.argmin(v0x_cost)] 
        # print(v0[0]) 
        
        print(v0, [x_cen, y_cen], np.min(v0x_cost)) 
        
        v0y = np.linspace(v0[1]*(1-delta*scale_coef**(scale0)), v0[1]*(1+delta*scale_coef**(scale0)), n_tests)
        # print(2 * v0y[0] / (v0y[0] + v0y[-1]), 2 * v0y[-1] / (v0y[0] + v0y[-1]))
        v0y_cost = np.zeros(shape = np.shape(v0y)) 
        for i in range(0, len(v0y)): 
            v0_temp = [v0[0], v0y[i]]
            cost_i, stdx, stdy = cost_centroid(centroid_data, v0_temp, x_cen, y_cen) 
            v0y_cost[i] = cost_i 
        # plt.scatter(v0y, v0y_cost) 
        # plt.show() 
        v0[1] = v0y[np.argmin(v0y_cost)] 
        
        print(v0, [x_cen, y_cen], np.min(v0y_cost)) 
        
        x_cen_vals = np.linspace(x_cen-a*delta*scale_coef**(scale0), x_cen+a*delta*scale_coef**(scale0), n_tests)
        x_cen_costs = np.zeros(shape = np.shape(x_cen_vals)) 
        for i in range(0, len(x_cen_vals)): 
            x_cen_temp = x_cen_vals[i] 
            cost_i, stdx, stdy = cost_centroid(centroid_data, v0, x_cen_temp, y_cen) 
            x_cen_costs[i] = cost_i 
        x_cen = x_cen_vals[np.argmin(x_cen_costs)]
        
        
        print(v0, [x_cen, y_cen], np.min(x_cen_costs)) 
        
        y_cen_vals = np.linspace(y_cen-a*delta*scale_coef**(scale0), y_cen+a*delta*scale_coef**(scale0), n_tests)
        y_cen_costs = np.zeros(shape = np.shape(y_cen_vals)) 
        for i in range(0, len(y_cen_vals)): 
            y_cen_temp = y_cen_vals[i] 
            cost_i, stdx, stdy = cost_centroid(centroid_data, v0, x_cen, y_cen_temp) 
            y_cen_costs[i] = cost_i 
        y_cen = y_cen_vals[np.argmin(y_cen_costs)]
        
        
        print(v0, [x_cen, y_cen], np.min(y_cen_costs)) 
        print(stdx, stdy, np.sqrt(stdx*stdx + stdy*stdy))
        a_final = (v0[0]**2+v0[1]**2)**(0.5)
        stdx_final = stdx
        stdy_final = stdy 
        
        # centroid_hist(centroid_data, v0, x_cen, y_cen) 
        
        
        
    # at this point x_cen, y_cen, and v0 are optomized. centroid_displacement should be similar to centroid 
    # and save a file with centroid_data's exact order. 
    centroid_displacement(centroid_data, v0, x_cen, y_cen) 
        
        
        
        
    # centroid_hist(centroid_data, v0, x_cen, y_cen) 
    
    # input("Press Enter...") 
    os.chdir(path)
    print("STDX: ", stdx_final, "STDY: ", stdy_final) 
    return a_final, stdx_final, stdy_final 
    

def centroid_analysis_v2(filename): 
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)
    a = np.mgrid[:5, :5][0]
    print(a) 
    print(np.fft.rfft2(a))
    a_final = 0 
    stdx_final = 0 
    stdy_final = 0
     
    
    centroid_data = np.load(filename[:-5] + 'centroids.npy')

    fig = plt.figure(figsize=(16, 9), dpi=100)
    plt.scatter(centroid_data[:, 0], centroid_data[:, 1]) 
    plt.title('Click two lattice points along an axis')
    plt.xlabel('x - axis (px)')
    plt.ylabel('y - axis (px)')
   
    point1 = fig.ginput(1, timeout = 0)[0]
    point2 = fig.ginput(1, timeout = 0)[0]
    plt.close(fig) 

    n_a = float(int(input("How many latice Spacings?")))
    
    v0 = [(float(point2[0]) - float(point1[0]))/n_a, (float(point2[1]) - float(point1[1]))/n_a] 
    x_cen = centroid_data[0, 0] 
    y_cen = centroid_data[0, 1] 
    print(v0) 
    v0 = [145.13, +0.2279] 
    # v0 = [121.005, 0.4422]
    x_cen = 58.566
    y_cen = 79.024
    centroid_data_temp = np.copy(centroid_data) 
    for n in range(0, len(centroid_data[:, 0])): 
        centroid_data_temp[n, 0] = centroid_data[n, 0] - x_cen 
        centroid_data_temp[n, 1] = centroid_data[n, 1] - y_cen 
        
    v0_mag = (np.sqrt(v0[0] * v0[0] + v0[1] * v0[1]))
    a0_mag = 1 / v0_mag
    a0_hat = v0 / v0_mag 
    a0 = a0_hat * a0_mag 
    window = 0.1
    window_N = 100
    u_vals = np.linspace(2 * a0[0] - window * a0_mag, 2 * a0[0] + window * a0_mag, window_N) 
    v_vals = np.linspace(2 * a0[1] - window * a0_mag, 2 * a0[1] + window * a0_mag, window_N) 
    print(a0[0], a0[1]) 
    print(u_vals[1], u_vals[-1])
    print(v_vals[1], v_vals[-1]) 
    
    
    # u_vals = np.linspace(-0.2 * a0_mag, 2.2 * a0_mag, window_N) 
    # v_vals = np.linspace(-0.2 * a0_mag, 2.2 * a0_mag, window_N) 
    um, ym = np.meshgrid(u_vals, v_vals) 
    zm = np.copy(um) 
    for i in range(0, len(u_vals)): 
        u_i = u_vals[i] 
        print(i / len(u_vals) * 100) 
        for j in range(0, len(v_vals)): 
            v_j = v_vals[j] 
            zm[i, j] = np.abs(fourier_points(centroid_data_temp, u_i, v_j))
    print(v0 / v0_mag * (1 / v0_mag)) 
    print(np.argmax(zm, axis = 0))
    print(zm.shape) 
    print(np.argmax(zm))
    index0 = np.argmax(zm) 
    print(index0%window_N)
    print((index0 - index0%window_N) / window_N) 
    print(u_vals[int((index0 - index0%window_N) / window_N)])
    print(v_vals[int(index0%window_N)])
    
     
    plt.contourf(u_vals, v_vals, zm)
    plt.axis('scaled') 
    plt.colorbar() 
    plt.show() 
    
    
    print(cost_centroid(centroid_data, v0, x_cen, y_cen)) 
    
    print(cost_centroid(centroid_data, v0, x_cen, y_cen)) 
    delta = 0.3
    scale_coef = 0.85
    n_tests = 500   # 1000 
    for scale in range(-20, 50): 
        scale0 = scale
        if scale0 <= 0: 
            scale0 = 1 
        # print(v0, [x_cen, y_cen]) 
        a = (v0[0]**2+v0[1]**2)**(0.5)
        v0x = np.linspace(v0[0]*(1-delta*scale_coef**(scale0)), v0[0]*(1+delta*scale_coef**(scale0)), n_tests)
        print(2 * v0x[0] / (v0x[0] + v0x[-1]), 2 * v0x[-1] / (v0x[0] + v0x[-1]))
        v0x_cost = np.zeros(shape = np.shape(v0x)) 
        for i in range(0, len(v0x)): 
            v0_temp = [v0x[i], v0[1]]
            # print(v0, v0_temp) 
            cost_i, stdx, stdy = cost_centroid(centroid_data, v0_temp, x_cen, y_cen) 
            v0x_cost[i] = cost_i 
        # plt.scatter(v0x, v0x_cost) 
        # plt.show() 
        v0[0] = v0x[np.argmin(v0x_cost)] 
        # print(v0[0]) 
        
        print(v0, [x_cen, y_cen], np.min(v0x_cost)) 
        
        v0y = np.linspace(v0[1]*(1-delta*scale_coef**(scale0)), v0[1]*(1+delta*scale_coef**(scale0)), n_tests)
        # print(2 * v0y[0] / (v0y[0] + v0y[-1]), 2 * v0y[-1] / (v0y[0] + v0y[-1]))
        v0y_cost = np.zeros(shape = np.shape(v0y)) 
        for i in range(0, len(v0y)): 
            v0_temp = [v0[0], v0y[i]]
            cost_i, stdx, stdy = cost_centroid(centroid_data, v0_temp, x_cen, y_cen) 
            v0y_cost[i] = cost_i 
        # plt.scatter(v0y, v0y_cost) 
        # plt.show() 
        v0[1] = v0y[np.argmin(v0y_cost)] 
        
        print(v0, [x_cen, y_cen], np.min(v0y_cost)) 
        
        x_cen_vals = np.linspace(x_cen-a*delta*scale_coef**(scale0), x_cen+a*delta*scale_coef**(scale0), n_tests)
        x_cen_costs = np.zeros(shape = np.shape(x_cen_vals)) 
        for i in range(0, len(x_cen_vals)): 
            x_cen_temp = x_cen_vals[i] 
            cost_i, stdx, stdy = cost_centroid(centroid_data, v0, x_cen_temp, y_cen) 
            x_cen_costs[i] = cost_i 
        x_cen = x_cen_vals[np.argmin(x_cen_costs)]

        
        
        print(v0, [x_cen, y_cen], np.min(x_cen_costs)) 
        
        y_cen_vals = np.linspace(y_cen-a*delta*scale_coef**(scale0), y_cen+a*delta*scale_coef**(scale0), n_tests)
        y_cen_costs = np.zeros(shape = np.shape(y_cen_vals)) 
        for i in range(0, len(y_cen_vals)): 
            y_cen_temp = y_cen_vals[i] 
            cost_i, stdx, stdy = cost_centroid(centroid_data, v0, x_cen, y_cen_temp) 
            y_cen_costs[i] = cost_i 
        y_cen = y_cen_vals[np.argmin(y_cen_costs)]
        
        
        print(v0, [x_cen, y_cen], np.min(y_cen_costs)) 
        print(stdx, stdy, np.sqrt(stdx*stdx + stdy*stdy))
        a_final = (v0[0]**2+v0[1]**2)**(0.5)
        stdx_final = stdx
        stdy_final = stdy 
        
        # centroid_hist(centroid_data, v0, x_cen, y_cen) 
        
    centroid_data_temp = np.copy(centroid_data) 
    for n in range(0, len(centroid_data[:, 0])): 
        centroid_data_temp[n, 0] = centroid_data[n, 0] - x_cen 
        centroid_data_temp[n, 1] = centroid_data[n, 1] - y_cen 
        
    v0_mag = (np.sqrt(v0[0] * v0[0] + v0[1] * v0[1]))
    a0_mag = 1 / v0_mag 
    a0_hat = v0 / v0_mag 
    a0 = a0_hat * a0_mag 
    window = 0.3
    window_N = 500
    u_vals = np.linspace(a0[0] - window * a0_mag, a0[0] + window * a0_mag, window_N) 
    v_vals = np.linspace(a0[1] - window * a0_mag, a0[1] + window * a0_mag, window_N) 
    um, ym = np.meshgrid(u_vals, v_vals) 
    zm = np.copy(um) 
    for i in range(0, len(u_vals)): 
        u_i = u_vals[i] 
        print(i / len(u_vals) * 100) 
        for j in range(0, len(v_vals)): 
            v_j = v_vals[j] 
            zm[i, j] = np.abs(fourier_points(centroid_data_temp, u_i, v_j))
    plt.contourf(u_vals, v_vals, zm)
    plt.axis('scaled') 
    plt.colorbar() 
    plt.show()     
        
    # at this point x_cen, y_cen, and v0 are optomized. centroid_displacement should be similar to centroid 
    # and save a file with centroid_data's exact order. 
    centroid_displacement(centroid_data, v0, x_cen, y_cen) 
        
        
        
        
    # centroid_hist(centroid_data, v0, x_cen, y_cen) 
    
    # input("Press Enter...") 
    os.chdir(path)
    print("STDX: ", stdx_final, "STDY: ", stdy_final) 
    return a_final, stdx_final, stdy_final 
    

def centroid_analysis_v3(filename): 
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)
    a = np.mgrid[:5, :5][0]
    print(a) 
    print(np.fft.rfft2(a))
    a_final = 0 
    stdx_final = 0 
    stdy_final = 0
     
    
    centroid_data = np.load(filename[:-5] + 'centroids.npy')

    fig = plt.figure(figsize=(16, 9), dpi=100)
    plt.scatter(centroid_data[:, 0], centroid_data[:, 1]) 
    plt.title('Click two lattice points along an axis')
    plt.xlabel('x - axis (px)')
    plt.ylabel('y - axis (px)')
   
    point1 = fig.ginput(1, timeout = 0)[0]
    point2 = fig.ginput(1, timeout = 0)[0]
    plt.close(fig) 

    n_a = float(int(input("How many latice Spacings?")))
    
    v0 = [(float(point2[0]) - float(point1[0]))/n_a, (float(point2[1]) - float(point1[1]))/n_a] 
    x_cen = centroid_data[0, 0] 
    y_cen = centroid_data[0, 1] 
    
    
    centroid_data_temp = np.copy(centroid_data) 
    for n in range(0, len(centroid_data[:, 0])): 
        centroid_data_temp[n, 0] = centroid_data[n, 0] - x_cen 
        centroid_data_temp[n, 1] = centroid_data[n, 1] - y_cen 
    window = 0.2 
    window_N = 101 
    
    for scale in range(0, 5): 
        print(v0) 
        scale0 = scale 
        
        v0_mag = (np.sqrt(v0[0] * v0[0] + v0[1] * v0[1]))
        a0_mag = 1 / v0_mag
        a0_hat = v0 / v0_mag 
        a0 = a0_hat * a0_mag
    
        u_vals = np.linspace(2 * a0[0] - window * a0_mag * 0.5 ** scale, 2 * a0[0] + window * a0_mag * 0.5 ** scale, window_N) 
        v_vals = np.linspace(2 * a0[1] - window * a0_mag * 0.5 ** scale, 2 * a0[1] + window * a0_mag * 0.5 ** scale, window_N) 
        
        um, ym = np.meshgrid(u_vals, v_vals) 
        zm = np.copy(um) 
        for i in range(0, len(u_vals)): 
            u_i = u_vals[i] 
            # print(i / len(u_vals) * 100) 
            for j in range(0, len(v_vals)): 
                v_j = v_vals[j] 
                zm[i, j] = np.abs(fourier_points(centroid_data_temp, u_i, v_j))
        # print(v0 / v0_mag * (1 / v0_mag)) 
        # print(np.argmax(zm, axis = 0))
        # print(zm.shape) 
        # print(np.argmax(zm))
        index0 = np.argmax(zm) 
        # print(index0%window_N)
        # print((index0 - index0%window_N) / window_N) 
        # print(u_vals[int((index0 - index0%window_N) / window_N)])
        # print(v_vals[int(index0%window_N)])
        a_new = [u_vals[int((index0 - index0%window_N) / window_N)], v_vals[int(index0%window_N)]]
        a_new_mag = np.sqrt(a_new[0] * a_new[0] + a_new[1] * a_new[1]) 
        v0 = a_new / (0.5 * a_new_mag * a_new_mag) 
    print(v0) 
        
        
          
    
    print(cost_centroid(centroid_data, v0, x_cen, y_cen)) 
    delta = 0.6
    scale_coef = 0.6
    n_tests = 501   # 1000 
    for scale in range(-5, 20): 
        scale0 = scale
        if scale0 <= 0: 
            scale0 = 1 
        # print(v0, [x_cen, y_cen]) 
        a = (v0[0]**2+v0[1]**2)**(0.5)
        
        delta = 0.1 * delta 
        v0x = np.linspace(v0[0]-a*delta*scale_coef**(scale0), v0[0]+a*delta*scale_coef**(scale0), n_tests)
        print('v0x: ', v0x[0], v0x[-1]) 
        print(2 * v0x[0] / (v0x[0] + v0x[-1]), 2 * v0x[-1] / (v0x[0] + v0x[-1]))
        v0x_cost = np.zeros(shape = np.shape(v0x)) 
        for i in range(0, len(v0x)): 
            v0_temp = [v0x[i], v0[1]]
            # print(v0, v0_temp) 
            cost_i, stdx, stdy = cost_centroid(centroid_data, v0_temp, x_cen, y_cen) 
            v0x_cost[i] = cost_i 
        # plt.scatter(v0x, v0x_cost) 
        # plt.show() 
        v0[0] = v0x[np.argmin(v0x_cost)] 
        # print(v0[0]) 
        
        print(v0, [x_cen, y_cen], np.min(v0x_cost)) 
        
        v0y = np.linspace(v0[1]-a*delta*scale_coef**(scale0), v0[1]+a*delta*scale_coef**(scale0), n_tests)
        # print(2 * v0y[0] / (v0y[0] + v0y[-1]), 2 * v0y[-1] / (v0y[0] + v0y[-1]))
        v0y_cost = np.zeros(shape = np.shape(v0y)) 
        for i in range(0, len(v0y)): 
            v0_temp = [v0[0], v0y[i]]
            cost_i, stdx, stdy = cost_centroid(centroid_data, v0_temp, x_cen, y_cen) 
            v0y_cost[i] = cost_i 
        # plt.scatter(v0y, v0y_cost) 
        # plt.show() 
        v0[1] = v0y[np.argmin(v0y_cost)] 
        
        print(v0, [x_cen, y_cen], np.min(v0y_cost)) 
        
        
        delta = 10 * delta 
        x_cen_vals = np.linspace(x_cen-a*delta*scale_coef**(scale0), x_cen+a*delta*scale_coef**(scale0), n_tests)
        x_cen_costs = np.zeros(shape = np.shape(x_cen_vals)) 
        for i in range(0, len(x_cen_vals)): 
            x_cen_temp = x_cen_vals[i] 
            cost_i, stdx, stdy = cost_centroid(centroid_data, v0, x_cen_temp, y_cen) 
            x_cen_costs[i] = cost_i 
        x_cen = x_cen_vals[np.argmin(x_cen_costs)]
        
        
        print(v0, [x_cen, y_cen], np.min(x_cen_costs)) 
        
        y_cen_vals = np.linspace(y_cen-a*delta*scale_coef**(scale0), y_cen+a*delta*scale_coef**(scale0), n_tests)
        y_cen_costs = np.zeros(shape = np.shape(y_cen_vals)) 
        for i in range(0, len(y_cen_vals)): 
            y_cen_temp = y_cen_vals[i] 
            cost_i, stdx, stdy = cost_centroid(centroid_data, v0, x_cen, y_cen_temp) 
            y_cen_costs[i] = cost_i 
        y_cen = y_cen_vals[np.argmin(y_cen_costs)]
        
        
        print(v0, [x_cen, y_cen], np.min(y_cen_costs)) 
        print(stdx, stdy, np.sqrt(stdx*stdx + stdy*stdy))
        a_final = (v0[0]**2+v0[1]**2)**(0.5)
        stdx_final = stdx
        stdy_final = stdy 
        
        # centroid_hist(centroid_data, v0, x_cen, y_cen) 
             
        
    # at this point x_cen, y_cen, and v0 are optomized. centroid_displacement should be similar to centroid 
    # and save a file with centroid_data's exact order. 
    centroid_displacement(centroid_data, v0, x_cen, y_cen) 
        
        
        
        
    # centroid_hist(centroid_data, v0, x_cen, y_cen) 
    
    # input("Press Enter...") 
    os.chdir(path)
    print("STDX: ", stdx_final, "STDY: ", stdy_final) 
    return a_final, stdx_final, stdy_final, v0, x_cen, y_cen 
    

def centroid_analysisx_v3(filename, v0, x_cen, y_cen): 
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)
    a = np.mgrid[:5, :5][0]
    print(a) 
    print(np.fft.rfft2(a))
    a_final = 0 
    stdx_final = 0 
    stdy_final = 0
     
    
    centroid_data = np.load(filename[:-5] + 'centroids.npy')
    
    print(cost_centroid(centroid_data, v0, x_cen, y_cen)) 
    delta0 = 0.6
    scale_coef = 0.6
    n_tests = 501   # 1000 
    for scale in range(-5, 20): 
        scale0 = scale
        if scale0 <= 0: 
            scale0 = 1 
        # print(v0, [x_cen, y_cen]) 
        a = (v0[0]**2+v0[1]**2)**(0.5)
        
        delta = 0.1 * delta0 
        v0x = np.linspace(v0[0]-a*delta*scale_coef**(scale0), v0[0]+a*delta*scale_coef**(scale0), n_tests)
        print('v0x: ', v0x[0], v0x[-1]) 
        print(2 * v0x[0] / (v0x[0] + v0x[-1]), 2 * v0x[-1] / (v0x[0] + v0x[-1]))
        v0x_cost = np.zeros(shape = np.shape(v0x)) 
        for i in range(0, len(v0x)): 
            v0_temp = [v0x[i], v0[1]]
            # print(v0, v0_temp) 
            cost_i, stdx, stdy = cost_centroidx(centroid_data, v0_temp, x_cen, y_cen) 
            v0x_cost[i] = cost_i 
        # plt.scatter(v0x, v0x_cost) 
        # plt.show() 
        v0[0] = v0x[np.argmin(v0x_cost)] 
        # print(v0[0]) 
        
        print(v0, [x_cen, y_cen], np.min(v0x_cost)) 
        delta = delta0
        v0y = np.linspace(v0[1]-a*delta*scale_coef**(scale0), v0[1]+a*delta*scale_coef**(scale0), n_tests)
        # print(2 * v0y[0] / (v0y[0] + v0y[-1]), 2 * v0y[-1] / (v0y[0] + v0y[-1]))
        v0y_cost = np.zeros(shape = np.shape(v0y)) 
        for i in range(0, len(v0y)): 
            v0_temp = [v0[0], v0y[i]]
            cost_i, stdx, stdy = cost_centroidx(centroid_data, v0_temp, x_cen, y_cen) 
            v0y_cost[i] = cost_i 
        # plt.scatter(v0y, v0y_cost) 
        # plt.show() 
        v0[1] = v0y[np.argmin(v0y_cost)] 
        
        print(v0, [x_cen, y_cen], np.min(v0y_cost)) 
        
        x_cen_vals = np.linspace(x_cen-a*delta*scale_coef**(scale0), x_cen+a*delta*scale_coef**(scale0), n_tests)
        x_cen_costs = np.zeros(shape = np.shape(x_cen_vals)) 
        for i in range(0, len(x_cen_vals)): 
            x_cen_temp = x_cen_vals[i] 
            cost_i, stdx, stdy = cost_centroidx(centroid_data, v0, x_cen_temp, y_cen) 
            x_cen_costs[i] = cost_i 
        x_cen = x_cen_vals[np.argmin(x_cen_costs)]
        
        
        print(v0, [x_cen, y_cen], np.min(x_cen_costs)) 
        
        print(stdx, stdy, np.sqrt(stdx*stdx + stdy*stdy))
        a_final = (v0[0]**2+v0[1]**2)**(0.5)
        stdx_final = stdx
        stdy_final = stdy 
        
        # centroid_hist(centroid_data, v0, x_cen, y_cen) 
             
        
    # at this point x_cen, y_cen, and v0 are optomized. centroid_displacement should be similar to centroid 
    # and save a file with centroid_data's exact order. 
    centroid_displacementx(centroid_data, v0, x_cen, y_cen) 
        
        
        
        
    # centroid_hist(centroid_data, v0, x_cen, y_cen) 
    
    # input("Press Enter...") 
    os.chdir(path)
    print("X optimization:") 
    print("STDX: ", stdx_final, "STDY: ", stdy_final) 
    return a_final, stdx_final, stdy_final, v0, x_cen, y_cen 
    

def centroid_analysisy_v3(filename, v0, x_cen, y_cen): 
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)
    a = np.mgrid[:5, :5][0]
    print(a) 
    print(np.fft.rfft2(a))
    a_final = 0 
    stdx_final = 0 
    stdy_final = 0
     
    
    centroid_data = np.load(filename[:-5] + 'centroids.npy')
    
    print(cost_centroid(centroid_data, v0, x_cen, y_cen)) 
    delta0 = 0.6
    scale_coef = 0.6
    n_tests = 501   # 1000 
    for scale in range(-5, 20): 
        scale0 = scale
        if scale0 <= 0: 
            scale0 = 1 
        # print(v0, [x_cen, y_cen]) 
        a = (v0[0]**2+v0[1]**2)**(0.5)
        
        delta = 0.1 * delta0
        v0x = np.linspace(v0[0]-a*delta*scale_coef**(scale0), v0[0]+a*delta*scale_coef**(scale0), n_tests)
        print('v0x: ', v0x[0], v0x[-1]) 
        print(2 * v0x[0] / (v0x[0] + v0x[-1]), 2 * v0x[-1] / (v0x[0] + v0x[-1]))
        v0x_cost = np.zeros(shape = np.shape(v0x)) 
        for i in range(0, len(v0x)): 
            v0_temp = [v0x[i], v0[1]]
            # print(v0, v0_temp) 
            cost_i, stdx, stdy = cost_centroidy(centroid_data, v0_temp, x_cen, y_cen) 
            v0x_cost[i] = cost_i 
        # plt.scatter(v0x, v0x_cost) 
        # plt.show() 
        v0[0] = v0x[np.argmin(v0x_cost)] 
        # print(v0[0]) 
        
        print(v0, [x_cen, y_cen], np.min(v0x_cost)) 
        delta = delta0
        
        v0y = np.linspace(v0[1]-a*delta*scale_coef**(scale0), v0[1]+a*delta*scale_coef**(scale0), n_tests)
        # print(2 * v0y[0] / (v0y[0] + v0y[-1]), 2 * v0y[-1] / (v0y[0] + v0y[-1]))
        v0y_cost = np.zeros(shape = np.shape(v0y)) 
        for i in range(0, len(v0y)): 
            v0_temp = [v0[0], v0y[i]]
            cost_i, stdx, stdy = cost_centroidy(centroid_data, v0_temp, x_cen, y_cen) 
            v0y_cost[i] = cost_i 
        # plt.scatter(v0y, v0y_cost) 
        # plt.show() 
        v0[1] = v0y[np.argmin(v0y_cost)] 
        
        print(v0, [x_cen, y_cen], np.min(v0y_cost)) 
        
        y_cen_vals = np.linspace(y_cen-a*delta*scale_coef**(scale0), y_cen+a*delta*scale_coef**(scale0), n_tests)
        y_cen_costs = np.zeros(shape = np.shape(y_cen_vals)) 
        for i in range(0, len(y_cen_vals)): 
            y_cen_temp = y_cen_vals[i] 
            cost_i, stdx, stdy = cost_centroidy(centroid_data, v0, x_cen, y_cen_temp) 
            y_cen_costs[i] = cost_i 
        y_cen = y_cen_vals[np.argmin(y_cen_costs)]
        
        
        print(v0, [x_cen, y_cen], np.min(y_cen_costs)) 
        print(stdx, stdy, np.sqrt(stdx*stdx + stdy*stdy))
        a_final = (v0[0]**2+v0[1]**2)**(0.5)
        stdx_final = stdx
        stdy_final = stdy 
        
        # centroid_hist(centroid_data, v0, x_cen, y_cen) 
             
        
    # at this point x_cen, y_cen, and v0 are optomized. centroid_displacement should be similar to centroid 
    # and save a file with centroid_data's exact order. 
    centroid_displacementy(centroid_data, v0, x_cen, y_cen) 
        
        
        
        
    # centroid_hist(centroid_data, v0, x_cen, y_cen) 
    
    # input("Press Enter...") 
    os.chdir(path)
    print("Y optimization:") 
    print("STDX: ", stdx_final, "STDY: ", stdy_final) 
    return a_final, stdx_final, stdy_final, v0, x_cen, y_cen 
    
    

def fourier_points(points, u, v): 
    summation = 0 + 0 * 1j  
    for n in range(0, len(points[:, 0])): 
        summation += np.exp(-2*3.141592653589*1j*(u*points[n, 0] + v*points[n, 1]))
    return summation 
    
def cost_centroid(centroid_data, v0, x0, y0): 
    v1 = [v0[0]*np.cos(np.pi/3) - v0[1]*np.sin(np.pi/3), v0[0]*np.sin(np.pi/3) + v0[1]*np.cos(np.pi/3)] 
    centroid_data_temp = np.copy(centroid_data) 
    r_data_temp = np.zeros(shape = np.shape(centroid_data[:, 0])) 
    for i in range(0, len(centroid_data[:, 0])): 
        pointi = [centroid_data[i, 0] - x0, centroid_data[i, 1] - y0]
        det = 1 / (v0[0]*v1[1] - v1[0]*v0[1])
        n0 = round((v1[1]*pointi[0] - v1[0]*pointi[1])*det) 
        n1 = round((-v0[1]*pointi[0] + v0[0]*pointi[1])*det)  
        centroid_data_temp[i, 0] = pointi[0] - n0 * v0[0] - n1 * v1[0] 
        centroid_data_temp[i, 1] = pointi[1] - n0 * v0[1] - n1 * v1[1]
        r_data_temp[i] = (centroid_data_temp[i, 0]**2 + centroid_data_temp[i, 1]**2)
        
    return np.sum(r_data_temp), np.std(centroid_data_temp[:, 0]), np.std(centroid_data_temp[:, 1]) 
    
    
def cost_centroidx(centroid_data, v0, x0, y0): 
    v1 = [v0[0]*np.cos(np.pi/3) - v0[1]*np.sin(np.pi/3), v0[0]*np.sin(np.pi/3) + v0[1]*np.cos(np.pi/3)] 
    centroid_data_temp = np.copy(centroid_data) 
    r_data_temp = np.zeros(shape = np.shape(centroid_data[:, 0])) 
    for i in range(0, len(centroid_data[:, 0])): 
        pointi = [centroid_data[i, 0] - x0, centroid_data[i, 1] - y0]
        det = 1 / (v0[0]*v1[1] - v1[0]*v0[1])
        n0 = round((v1[1]*pointi[0] - v1[0]*pointi[1])*det) 
        n1 = round((-v0[1]*pointi[0] + v0[0]*pointi[1])*det)  
        centroid_data_temp[i, 0] = pointi[0] - n0 * v0[0] - n1 * v1[0] 
        centroid_data_temp[i, 1] = pointi[1] - n0 * v0[1] - n1 * v1[1]
        r_data_temp[i] = (centroid_data_temp[i, 0]**2)
        
    return np.sum(r_data_temp), np.std(centroid_data_temp[:, 0]), np.std(centroid_data_temp[:, 1])  
    
    
def cost_centroidy(centroid_data, v0, x0, y0): 
    v1 = [v0[0]*np.cos(np.pi/3) - v0[1]*np.sin(np.pi/3), v0[0]*np.sin(np.pi/3) + v0[1]*np.cos(np.pi/3)] 
    centroid_data_temp = np.copy(centroid_data) 
    r_data_temp = np.zeros(shape = np.shape(centroid_data[:, 0])) 
    for i in range(0, len(centroid_data[:, 0])): 
        pointi = [centroid_data[i, 0] - x0, centroid_data[i, 1] - y0]
        det = 1 / (v0[0]*v1[1] - v1[0]*v0[1])
        n0 = round((v1[1]*pointi[0] - v1[0]*pointi[1])*det) 
        n1 = round((-v0[1]*pointi[0] + v0[0]*pointi[1])*det)  
        centroid_data_temp[i, 0] = pointi[0] - n0 * v0[0] - n1 * v1[0] 
        centroid_data_temp[i, 1] = pointi[1] - n0 * v0[1] - n1 * v1[1]
        r_data_temp[i] = (centroid_data_temp[i, 1]**2)
        
    return np.sum(r_data_temp), np.std(centroid_data_temp[:, 0]), np.std(centroid_data_temp[:, 1])  
    

def centroid_displacement(centroid_data, v0, x0, y0): 
    v1 = [v0[0]*np.cos(np.pi/3) - v0[1]*np.sin(np.pi/3), v0[0]*np.sin(np.pi/3) + v0[1]*np.cos(np.pi/3)] 
    centroid_data_temp = np.copy(centroid_data) 
    r_data_temp = np.zeros(shape = np.shape(centroid_data[:, 0])) 
    for i in range(0, len(centroid_data[:, 0])): 
        pointi = [centroid_data[i, 0] - x0, centroid_data[i, 1] - y0]
        det = 1 / (v0[0]*v1[1] - v1[0]*v0[1])
        n0 = round((v1[1]*pointi[0] - v1[0]*pointi[1])*det) 
        n1 = round((-v0[1]*pointi[0] + v0[0]*pointi[1])*det)  
        centroid_data_temp[i, 0] = pointi[0] - n0 * v0[0] - n1 * v1[0] 
        centroid_data_temp[i, 1] = pointi[1] - n0 * v0[1] - n1 * v1[1]
        r_data_temp[i] = (centroid_data_temp[i, 0]**2 + centroid_data_temp[i, 1]**2)
    
    # centroid_data_temp holds all the displacement values with [:, 0] holding x data and [:, 1] holding y data 
    np.save('displacements.npy', centroid_data_temp)
        
    return np.sum(r_data_temp), np.std(centroid_data_temp[:, 0]), np.std(centroid_data_temp[:, 1]) 
    
    
def centroid_displacementx(centroid_data, v0, x0, y0): 
    v1 = [v0[0]*np.cos(np.pi/3) - v0[1]*np.sin(np.pi/3), v0[0]*np.sin(np.pi/3) + v0[1]*np.cos(np.pi/3)] 
    centroid_data_temp = np.copy(centroid_data) 
    r_data_temp = np.zeros(shape = np.shape(centroid_data[:, 0])) 
    for i in range(0, len(centroid_data[:, 0])): 
        pointi = [centroid_data[i, 0] - x0, centroid_data[i, 1] - y0]
        det = 1 / (v0[0]*v1[1] - v1[0]*v0[1])
        n0 = round((v1[1]*pointi[0] - v1[0]*pointi[1])*det) 
        n1 = round((-v0[1]*pointi[0] + v0[0]*pointi[1])*det)  
        centroid_data_temp[i, 0] = pointi[0] - n0 * v0[0] - n1 * v1[0] 
        centroid_data_temp[i, 1] = pointi[1] - n0 * v0[1] - n1 * v1[1]
        r_data_temp[i] = (centroid_data_temp[i, 0]**2 + centroid_data_temp[i, 1]**2)
    
    # centroid_data_temp holds all the displacement values with [:, 0] holding x data and [:, 1] holding y data 
    np.save('displacementsx.npy', centroid_data_temp)
        
    return np.sum(r_data_temp), np.std(centroid_data_temp[:, 0]), np.std(centroid_data_temp[:, 1]) 
    

def centroid_displacementy(centroid_data, v0, x0, y0): 
    v1 = [v0[0]*np.cos(np.pi/3) - v0[1]*np.sin(np.pi/3), v0[0]*np.sin(np.pi/3) + v0[1]*np.cos(np.pi/3)] 
    centroid_data_temp = np.copy(centroid_data) 
    r_data_temp = np.zeros(shape = np.shape(centroid_data[:, 0])) 
    for i in range(0, len(centroid_data[:, 0])): 
        pointi = [centroid_data[i, 0] - x0, centroid_data[i, 1] - y0]
        det = 1 / (v0[0]*v1[1] - v1[0]*v0[1])
        n0 = round((v1[1]*pointi[0] - v1[0]*pointi[1])*det) 
        n1 = round((-v0[1]*pointi[0] + v0[0]*pointi[1])*det)  
        centroid_data_temp[i, 0] = pointi[0] - n0 * v0[0] - n1 * v1[0] 
        centroid_data_temp[i, 1] = pointi[1] - n0 * v0[1] - n1 * v1[1]
        r_data_temp[i] = (centroid_data_temp[i, 0]**2 + centroid_data_temp[i, 1]**2)
    
    # centroid_data_temp holds all the displacement values with [:, 0] holding x data and [:, 1] holding y data 
    np.save('displacementsy.npy', centroid_data_temp)
        
    return np.sum(r_data_temp), np.std(centroid_data_temp[:, 0]), np.std(centroid_data_temp[:, 1])  
        
    
    

    
    
def centroid_hist(centroid_data, v0, x0, y0): 
    v1 = [v0[0]*np.cos(np.pi/3) - v0[1]*np.sin(np.pi/3), v0[0]*np.sin(np.pi/3) + v0[1]*np.cos(np.pi/3)] 
    centroid_data_temp = np.copy(centroid_data) 
    r_data_temp = np.zeros(shape = np.shape(centroid_data[:, 0])) 
    for i in range(0, len(centroid_data[:, 0])): 
        pointi = [centroid_data[i, 0] - x0, centroid_data[i, 1] - y0]

        det = 1 / (v0[0]*v1[1] - v1[0]*v0[1])
        n0 = round((v1[1]*pointi[0] - v1[0]*pointi[1])*det) 
        n1 = round((-v0[1]*pointi[0] + v0[0]*pointi[1])*det)  
        centroid_data_temp[i, 0] = pointi[0] - n0 * v0[0] - n1 * v1[0] 
        centroid_data_temp[i, 1] = pointi[1] - n0 * v0[1] - n1 * v1[1]
        r_data_temp[i] = (centroid_data_temp[i, 0]**2 + centroid_data_temp[i, 1]**2)
        
    plt.plot(centroid_data_temp[:, 0], centroid_data_temp[:, 1], 'bo')
    plt.show() 






