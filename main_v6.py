from Image_Analysis_Functions_v4 import generate_results_directory
from Image_Analysis_Functions_v4 import image_reduced_by_mouse
from Image_Analysis_Functions_v4 import image_smooth
from Image_Analysis_Functions_v4 import image_smooth_1
from Image_Analysis_Functions_v4 import dist_brightness
from Image_Analysis_Functions_v4 import boolean_mapping_v2
from Image_Analysis_Functions_v4 import object_search_v5
from Image_Analysis_Functions_v4 import object_points_list_v2
from Image_Analysis_Functions_v4 import remove_objects
from Image_Analysis_Functions_v4 import tif_image_fix
from Image_Analysis_Functions_v4 import object_boundry_analysis_v2
from Image_Analysis_Functions_v4 import pin_removal
from Image_Analysis_Functions_v4 import gaussian_smooth
from Image_Analysis_Functions_v4 import brightness_cutoff_calc
from Image_Analysis_Functions_v4 import object_boundry_results_v1
from Image_Analysis_Functions_v4 import object_boundry_results_v2
from Image_Analysis_Functions_v4 import centroid_analysis_v0
from Image_Analysis_Functions_v4 import centroid_analysis_v1
from Image_Analysis_Functions_v4 import centroid_analysis_v2
from Image_Analysis_Functions_v4 import centroid_analysis_v3
from Image_Analysis_Functions_v4 import centroid_analysisx_v3
from Image_Analysis_Functions_v4 import centroid_analysisy_v3
from Image_Analysis_Functions_v4 import visualazation_displacement_v1
from Image_Analysis_Functions_v4 import visualazation_displacement_vtemp
import scipy.stats
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from random import gauss
import copy


def analysis0(filename, number_smooth):
    generate_results_directory(filename)
    image_reduced_by_mouse(filename, 0)
    dist_brightness(filename, 0 + 1, 0, 0)
    for i in range(1, number_smooth):
        image_smooth(filename, i, 0)
        dist_brightness(filename, i + 1, 0, 0)
    return filename
 


def analysis1(filename, width, version_number, peaks_skip, percentile):
    dist = dist_brightness(filename, version_number, 0, 0)
    new_dist, brightness_cutoff = brightness_cutoff_calc(dist, percentile, peaks_skip, width, 1)
    boolean_mapping_v2(filename, version_number, brightness_cutoff, True, 0)
    path = os.getcwd()
    directory = path + '/' + filename[:-4] + '_results'
    os.chdir(directory)
    version_number = np.load('version_counter.npy')
    os.chdir(path)
    remove_objects(filename, version_number, 0)



def analysis1v2(filename, version_number, brightness_cutoff):
    boolean_mapping_v2(filename, version_number, brightness_cutoff, True, 0) 
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)
    version_number = np.load('version_counter.npy')
    os.chdir(path)
    remove_objects(filename, version_number, 0)
    
    
def analysis1v3(filename, version_number, brightness_cutoff):
    boolean_mapping_v2(filename, version_number, brightness_cutoff, True, 0) 
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)
    version_number = np.load('version_counter.npy')
    os.chdir(path)
    # remove_objects(filename, version_number, 0)
    print('Finished analysis1v3') 
    

def analysis2(filename, version_number):
    object_search_v5(filename, version_number, 0, 0.2, 0)
    remove_objects(filename, version_number + 1, 0)
    

def analysis2v2(filename, version_number):
    print('Object Search in analysis2v2') 
    object_search_v5(filename, version_number, 3, 0.9, 0)
    print('Manual object removal in analysis2v2') 
    remove_objects(filename, version_number + 1, 0)
    print('Finished analysis2v2') 


def analysis3(filename, version_number):
    pin_removal(filename, version_number, 0)
    version_number += 2
    object_search_v5(filename, version_number, 0, 0.2, 0)
    version_number += 1
    object_points_list_v2(filename, version_number, 0)
    object_boundry_analysis_v2(filename, version_number, 0)


def analysis4(filename, version_number):
    object_search_v5(filename, version_number, 0, 0.2, 0)
    version_number += 1
    object_points_list_v2(filename, version_number, 0)
    results = object_boundry_results_v1(filename, version_number, 0)
    print('Results are as follows: mean cross-section radius, mean cross-section 90th percentil, 10th percentile, circular area radisu')
    for variable in range(0, 3):
        n = len(results[variable, :])
        mean, se = np.mean(results[variable, :]), scipy.stats.sem(results[variable, :])
        h = se * scipy.stats.t.ppf((1 + 0.95) / 2., n - 1)
        print('95percent confidence interval of mean is %.2f to %.2f with mean %.2f for %.0f objects'%(mean - h, mean + h, mean, n))

    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)
    circular_results = np.load(filename[:-5] + '_circular_results.npy')
    n = len(circular_results)
    mean, se = np.mean(circular_results), scipy.stats.sem(circular_results)
    h = se * scipy.stats.t.ppf((1 + 0.95) / 2., n - 1)
    print('95percent confidence interval of mean is %.2f to %.2f with mean %.2f for %.0f objects'%(mean - h, mean + h, mean, n))
    os.chdir(path)
    
    
def analysis4v2(filename, version_number):
    lattice_constant, stdx, stdy, = centroid_analysis_v0(filename) 
    object_search_v5(filename, version_number, 0, 0.2, 0)
    version_number += 1
    object_points_list_v2(filename, version_number, 0)
    results = object_boundry_results_v2(filename, version_number, 0)
    print('Results are as follows: mean cross-section radius, mean cross-section 90th percentil, 10th percentile, circular area radisu')
    for variable in range(0, 3):
        n = len(results[variable, :])
        mean, se = np.mean(results[variable, :]), scipy.stats.sem(results[variable, :])
        h = se * scipy.stats.t.ppf((1 + 0.95) / 2., n - 1)
        print('95percent confidence interval of mean is %.2f to %.2f with mean %.2f for %.0f objects'%(mean - h, mean + h, mean, n))
    std_r0 = np.std(results[3, :]) 
    std_r1 = np.std(results[4, :]) 
    # print('3:', results[3, :]) 
    # print('4:', results[4, :]) 
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)
    circular_results = np.load(filename[:-5] + '_circular_results.npy')
    n = len(circular_results)
    mean, se = np.mean(circular_results), scipy.stats.sem(circular_results)
    h = se * scipy.stats.t.ppf((1 + 0.95) / 2., n - 1)
    print('95percent confidence interval of mean is %.2f to %.2f with mean %.2f for %.0f objects'%(mean - h, mean + h, mean, n))
    print('std r0: ', std_r0) 
    print('std r1: ', std_r1)
    print('lattice constant: ', lattice_constant) 
    print('stdx: ', stdx) 
    print('stdy: ', stdy)  
    
    os.chdir(path)


def analysis4v3(filename, version_number):
    lattice_constant, stdx, stdy, = centroid_analysis_v1(filename) 
    object_search_v5(filename, version_number, 0, 0.2, 0)
    version_number += 1
    object_points_list_v2(filename, version_number, 0)
    results = object_boundry_results_v2(filename, version_number, 0)
    print('Results are as follows: mean cross-section radius, mean cross-section 90th percentil, 10th percentile, circular area radisu')
    for variable in range(0, 3):
        n = len(results[variable, :])
        mean, se = np.mean(results[variable, :]), scipy.stats.sem(results[variable, :])
        h = se * scipy.stats.t.ppf((1 + 0.95) / 2., n - 1)
        print('95percent confidence interval of mean is %.2f to %.2f with mean %.2f for %.0f objects'%(mean - h, mean + h, mean, n))
    std_r0 = np.std(results[3, :]) 
    std_r1 = np.std(results[4, :]) 
    # print('3:', results[3, :]) 
    # print('4:', results[4, :]) 
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)
    np.save('radius_results.npy', results) 
    circular_results = np.load(filename[:-5] + '_circular_results.npy')
    n = len(circular_results)
    mean, se = np.mean(circular_results), scipy.stats.sem(circular_results)
    h = se * scipy.stats.t.ppf((1 + 0.95) / 2., n - 1)
    print('95percent confidence interval of mean is %.2f to %.2f with mean %.2f for %.0f objects'%(mean - h, mean + h, mean, n))
    print('std r0: ', std_r0, ' Mean r0: ', np.mean(results[3, :])) 
    print('std r1: ', std_r1, ' Mean r1: ', np.mean(results[4, :]))
    print('lattice constant: ', lattice_constant) 
    print('stdx: ', stdx) 
    print('stdy: ', stdy)  
    # results holds object 
    os.chdir(path)
    
    
def analysis4v4(filename, version_number):
    lattice_constant, stdx, stdy, v0_temp0, x_cen_temp0, y_cen_temp0 = centroid_analysis_v3(filename) 
    lattice_constantx, stdxx, stdyx, v0_tempx, x_cen_tempx, y_cen_tempx = centroid_analysisx_v3(filename, v0_temp0, x_cen_temp0, y_cen_temp0) 
    lattice_constanty, stdxy, stdyy, v0_tempy, x_cen_tempy, y_cen_tempy = centroid_analysisy_v3(filename, v0_temp0, x_cen_temp0, y_cen_temp0) 
    object_search_v5(filename, version_number, 0, 0.2, 0)
    version_number += 1
    object_points_list_v2(filename, version_number, 0)
    results = object_boundry_results_v2(filename, version_number, 0)
    print('Results are as follows: mean cross-section radius, mean cross-section 90th percentil, 10th percentile, circular area radisu')
    for variable in range(0, 3):
        n = len(results[variable, :])
        mean, se = np.mean(results[variable, :]), scipy.stats.sem(results[variable, :])
        h = se * scipy.stats.t.ppf((1 + 0.95) / 2., n - 1)
        print('95percent confidence interval of mean is %.2f to %.2f with mean %.2f for %.0f objects'%(mean - h, mean + h, mean, n))
    std_r0 = np.std(results[3, :]) 
    std_r1 = np.std(results[4, :]) 
    # print('3:', results[3, :]) 
    # print('4:', results[4, :]) 
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)
    np.save('radius_results.npy', results) 
    circular_results = np.load(filename[:-5] + '_circular_results.npy')
    n = len(circular_results)
    mean, se = np.mean(circular_results), scipy.stats.sem(circular_results)
    h = se * scipy.stats.t.ppf((1 + 0.95) / 2., n - 1)
    print('95percent confidence interval of mean is %.2f to %.2f with mean %.2f for %.0f objects'%(mean - h, mean + h, mean, n))
    print('std r0: ', std_r0, ' Mean r0: ', np.mean(results[3, :])) 
    print('std r1: ', std_r1, ' Mean r1: ', np.mean(results[4, :]))
    print('lattice constant: ', lattice_constant) 
    print('stdx: ', stdx) 
    print('stdy: ', stdy)  
    print('stdxx: ', stdxx) 
    print('stdyx: ', stdyx) 
    print('stdxy: ', stdxy) 
    print('stdyy: ', stdyy) 
    # results holds object 
    os.chdir(path)
    

def analysis5v1(filename, version_number): 
    # load results, circular results and displacements 
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)
    centroids = np.load(filename[:-5] + 'centroids.npy')
    circular_results = np.load(filename[:-5] + '_circular_results.npy')
    radius_results = np.load('radius_results.npy') 
    displacement_results = np.load('displacements.npy') 
    
    visualazation_displacement_v1(filename, circular_results - np.mean(circular_results), version_number, 'r_circularized', 0.15, 0)
    visualazation_displacement_v1(filename, radius_results[0, :] - np.mean(radius_results[0, :]), version_number, 'r_mean', 0.15, 0)
    visualazation_displacement_v1(filename, radius_results[1, :] - np.mean(radius_results[1, :]), version_number, 'r_90percentile', 0.15, 0)
    visualazation_displacement_v1(filename, radius_results[2, :] - np.mean(radius_results[2, :]), version_number, 'r_10percentile', 0.15, 0)
    visualazation_displacement_v1(filename, radius_results[3, :] - np.mean(radius_results[3, :]), version_number, 'r0', 0.15, 0)
    visualazation_displacement_v1(filename, radius_results[4, :] - np.mean(radius_results[4, :]), version_number, 'r1', 0.15, 0)
    visualazation_displacement_v1(filename, displacement_results[:, 0], version_number, 'x_displacement', 0.15, 0)
    visualazation_displacement_v1(filename, displacement_results[:, 1], version_number, 'y_displacement', 0.15, 0)
    os.chdir(path)
    

def analysis5v2(filename, version_number): 
    # load results, circular results and displacements 
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)
    centroids = np.load(filename[:-5] + 'centroids.npy')
    circular_results = np.load(filename[:-5] + '_circular_results.npy')
    radius_results = np.load('radius_results.npy') 
    displacement_results = np.load('displacements.npy') 
    displacement_resultsx = np.load('displacementsx.npy') 
    displacement_resultsy = np.load('displacementsy.npy') 
    print("1") 
    visualazation_displacement_v1(filename, displacement_resultsx[:, 0], version_number, 'x_displacementx', 0.15, 0)
    visualazation_displacement_v1(filename, displacement_resultsx[:, 1], version_number, 'y_displacementx', 0.15, 0)
    visualazation_displacement_v1(filename, displacement_resultsy[:, 0], version_number, 'x_displacementy', 0.15, 0)
    visualazation_displacement_v1(filename, displacement_resultsy[:, 1], version_number, 'y_displacementy', 0.15, 0)
    visualazation_displacement_v1(filename, circular_results - np.mean(circular_results), version_number, 'r_circularized', 0.15, 0)
    visualazation_displacement_v1(filename, radius_results[0, :] - np.mean(radius_results[0, :]), version_number, 'r_mean', 0.15, 0)
    visualazation_displacement_v1(filename, radius_results[1, :] - np.mean(radius_results[1, :]), version_number, 'r_90percentile', 0.15, 0)
    visualazation_displacement_v1(filename, radius_results[2, :] - np.mean(radius_results[2, :]), version_number, 'r_10percentile', 0.15, 0)
    visualazation_displacement_v1(filename, radius_results[3, :] - np.mean(radius_results[3, :]), version_number, 'r0', 0.15, 0)
    visualazation_displacement_v1(filename, radius_results[4, :] - np.mean(radius_results[4, :]), version_number, 'r1', 0.15, 0)
    visualazation_displacement_v1(filename, displacement_results[:, 0], version_number, 'x_displacement', 0.15, 0)
    visualazation_displacement_v1(filename, displacement_results[:, 1], version_number, 'y_displacement', 0.15, 0)
    os.chdir(path)
    

def analysis5vtemp(filename, version_number): 
    # load results, circular results and displacements 
    path = os.getcwd()
    directory = path + '/' + filename[:-5] + '_results'
    os.chdir(directory)
    centroids = np.load(filename[:-5] + 'centroids.npy')
    circular_results = np.load(filename[:-5] + '_circular_results.npy')
    radius_results = np.load('radius_results.npy') 
    displacement_results = np.load('displacements.npy') 
    
    visualazation_displacement_vtemp(filename, displacement_results[:, 1], version_number, 'y_displacement', 0.15, 0, 23.95)
    os.chdir(path)


def transform(Image, angle, scale):
    1    


def mix(n0, n, filename, location): 
    path = os.getcwd() 
    directory = path + '/' + location 
    os.chdir(directory) 
    
    n_0_str_temp = str(n0) 
    n_0_str = n_0_str_temp
    for j in range(0, 3 - len(n_0_str_temp)): 
        n_0_str = '0' + n_0_str 
    print(n_0_str)
    image_0 = plt.imread(filename + n_0_str + 'a_s.tiff') 
    image_0 = image_0[0:2760, :, :]
    
    image_1 = plt.imread(filename + '003' + 'a_s.tiff') 
    image_1 = image_1[0:2760, :, :]
    image_final = copy.deepcopy(image_0) 
    
    d_size = 40 
    dx = np.linspace(8, 12, 5) 
    dy = np.linspace(-10, -14, 5) 
    results = np.zeros((5, 5)) 
    
    for n in range(0, len(results[:, 0])): 
        for m in range(0, len(results[0, :])): 
            dx_n = dx[n] 
            dy_m = dy[m] 
            results[n, m] = dx_n
    print(results) 
    
    # print(results) 
    # print(dx) 
    
    # for n in range(0, len(results[:, 0])):
    #     dx_n = dx[n]
    #     n_percent = n / len(results[:, 0]) * 100 
    #     for m in range(0, len(results[0, :])):
    #         dy_m = dy[m]
    #         sum1 = 0 
    #         m_percent = m / len(results[:, 0]) / len(results[0, :]) * 100 
    #         for i in range(d_size + 1, len(image_final[:, 0, 0]) - d_size - 1) :
    #             i_percent = i / len(results[:, 0]) / len(results[0, :]) / len(image_final[:, 0, 0]) * 100 
    #             print(n_percent + m_percent + i_percent) 
    #             for j in range(d_size + 1, len(image_final[0, :, 0]) - d_size - 1): 
    #                 sum1 += np.abs(image_1[int(i + dx_n), int(j + dy_m), 0] - image_0[i, j, 0]) / 255
    #         results[n, m] = sum1 
    # print(results) 
     
            
     
    # manual adjust works well ################################################
    # image_0 = plt.imread(filename + n_0_str + 'a_s.tiff') 
    # image_0 = image_0[0:2760, :, :]
    
    # image_1 = plt.imread(filename + '003' + 'a_s.tiff') 
    # image_1 = image_1[0:2760, :, :]
    # image_final = copy.deepcopy(image_0) 
    
    # xc = 1500
    # yc = 1100
    # image_0 = np.copy(image_0[xc - 100:xc + 100, yc - 100:yc + 100, :]) 
    # image_1 = np.copy(image_1[xc - 100:xc + 100, yc - 100:yc + 100, :])
    # image_final = np.copy(image_1)
    # dx = 17
    # dy = -13
    # for i in range(1 + np.abs(dx), len(image_final[:, 0, 0]) - np.abs(dx) - 2): 
    #     for j in range(1 + np.abs(dy), len(image_final[0, :, 0]) - np.abs(dy) - 2): 
    #         # image_final[i, j, 0] = image_1[int(i + dx), int(j + dy), 0] * 0.499 + image_0[i, j, 0] * 0.499 
    #         # image_final[i, j, 1] = image_final[i, j, 0] 
    #         # image_final[i, j, 2] = image_final[i, j, 0]
    #         image_final[i, j, 0] = image_1[int(i + dx), int(j + dy), 0] 
    #         image_final[i, j, 1] = image_0[i, j, 0] 
    #         image_final[i, j, 2] = 0
    
    # plt.figure(figsize=(20, 20))
    # plt.imshow(image_final) 
    # plt.show() 
    ###########################################################################
        
     
        
    # Image Combine ###########################################################
    dx = 32
    dy = -23 
    image_0 = plt.imread('combined.tiff') 
    image_0 = image_0[0:2760, :, :]
    
    image_1 = plt.imread(filename + '005' + 'a_s.tiff') 
    image_1 = image_1[0:2760, :, :]
    image_final = copy.deepcopy(image_0) 
     
    for i in range(1 + np.abs(dx), len(image_final[:, 0, 0]) - (1 + np.abs(dx))): 
        print(100 * i / len(image_final[:, 0, 0])) 
        for j in range(1 + np.abs(dy), len(image_final[0, :, 0]) - (1 + np.abs(dy))): 
            image_final[i, j, 0] = int(image_1[i + dx, j + dy, 0]) * 0.2 + int(image_0[i, j, 0]) * 1.0
            image_final[i, j, 1] = image_final[i, j, 0]
            image_final[i, j, 2] = image_final[i, j, 0]
    plt.figure(figsize = (20, 16))
    plt.imshow(image_final) 
    plt.imsave('combined.tiff', image_final) 
    plt.show() 
    ###########################################################################

    
    
    # for i in range(n0, n): 
    #     n_i_str_temp = str(i) 
    #     n_i_str = n_i_str_temp
    #     for j in range(0, 3 - len(n_i_str_temp)):
    #         n_i_str = '0' + n_i_str 
    #     print(filename + n_i_str + 'a.tiff') 
    #     image = plt.imread(filename + n_i_str + 'a.tiff')
    #     filename_full = filename + n_i_str + 'a.tiff' 
    #     image_smooth_1(filename_full, 0)
        
            
    #     if i == 1:
    #         plt.figure(figsize=(20,16)) 
    #         image = plt.imread(filename + n_i_str + '.tif')
    #         image = image[0:2760, :, :] 
    #         plt.imshow(image) 
    #         plt.show() 
    #     print(image.shape) 

    # need to write this in terms of 4 point transform. 2 points on the projects image plane and 2 points on the projecting image plane
    # translate to match up the first two points then rotate.
    # calculate the resulting boarder points of the transformed image. generate a sufficiently large resulting image to capture both static and transformed image.


def combine_images(folder):
	path = os.getcwd() 
	directory = path + '/' + folder
	os.chdir(directory)
	images = os.listdir() 
	imagen = plt.imread(images[0]) 
	
	fig = plt.figure(figsize=(20,20))
	plt.imshow(imagen)
	point1 = fig.ginput(1, timeout = 0) 
	point2 = fig.ginput(1, timeout = 0) 
	plt.close() 
	print(point1) 
	
	shifts = np.zeros((len(images), 2))
	print(shifts)
	for n in range(1, len(images)): 
		image_0 = plt.imread(images[0]) 
		image_n = plt.imread(images[n]) 
		image_final = copy.deepcopy(image_0) 
		
		
		x_bounds = [int(point1[0][1]), int(point2[0][1])] 
		y_bounds = [int(point1[0][0]), int(point2[0][0])] 
		extra_px = 10 
		xll = np.min(x_bounds) - extra_px 
		xur = np.max(x_bounds) + extra_px
		yll = np.min(y_bounds) - extra_px
		yur = np.max(y_bounds) + extra_px
		
		stop = False 
		dx = 0 
		dy = 0 
		
		# Corse alignment 
		image_0_temp = np.copy(image_0[xll:xur, yll:yur, :]) 
		image_n_temp = np.copy(image_n[xll+dx:xur+dx, yll+dy:yur+dy, :]) 
		image_final_temp = np.copy(image_n_temp)
		image_final_temp[:, :, 0] = image_n_temp[:, :, 0] 
		image_final_temp[:, :, 1] = image_0_temp[:, :, 0] 
		image_final_temp[:, :, 2] = 0 * image_final_temp[:, :, 2] 
		fig = plt.figure(figsize=(20, 20)) 
		plt.title('POINT RED CENTER') 
		plt.imshow(image_final_temp) 
		point3 = fig.ginput(1, timeout = 0) 
		fig = plt.figure(figsize=(20, 20)) 
		plt.title('POINT GREEN CENTER') 
		plt.imshow(image_final_temp) 
		point4 = fig.ginput(1, timeout = 0) 
		dx = int(point3[0][1] - point4[0][1])
		dy = int(point3[0][0] - point4[0][0])
		plt.close()
		
		
		
		while stop == False: 
			image_0_temp = np.copy(image_0[xll:xur, yll:yur, :]) 
			image_n_temp = np.copy(image_n[xll+dx:xur+dx, yll+dy:yur+dy, :]) 
			image_final_temp = np.copy(image_n_temp)
			image_final_temp[:, :, 0] = image_n_temp[:, :, 0] 
			image_final_temp[:, :, 1] = image_0_temp[:, :, 0] 
			image_final_temp[:, :, 2] = 0 * image_final_temp[:, :, 2] 
			for i in range(0, extra_px): 
				for j in range(0, len(image_final_temp[0, :, 0])): 
					image_final_temp[i, j, 0] = 0
					image_final_temp[i, j, 1] = 0
					image_final_temp[i, j, 2] = 0
			for i in range(0, len(image_final_temp[:, 0, 0])): 
				for j in range(0, extra_px): 
					image_final_temp[i, j, 0] = 1
					image_final_temp[i, j, 1] = 1
					image_final_temp[i, j, 2] = 1
			for i in range(0, extra_px): 
				for j in range(0, len(image_final_temp[0, :, 0])): 
					image_final_temp[-i-1, j, 0] = 2
					image_final_temp[-i-1, j, 1] = 2
					image_final_temp[-i-1, j, 2] = 2
			for i in range(0, len(image_final_temp[:, 0, 0])): 
				for j in range(0, extra_px): 
					image_final_temp[i, -j-1, 0] = 3
					image_final_temp[i, -j-1, 1] = 3
					image_final_temp[i, -j-1, 2] = 3
			for i in range(0, extra_px): 
				for j in range(0, extra_px): 
					image_final_temp[i, j, 0] = 254
					image_final_temp[i, j, 1] = 254
					image_final_temp[i, j, 2] = 254 
			fig = plt.figure(figsize=(20, 20))
			plt.title('Move Red to green')  
			plt.imshow(image_final_temp)
			point = fig.ginput(1, timeout = 0) 
			plt.clf() 
			plt.close()
			point_val = image_final_temp[int(point[0][1]), int(point[0][0]), 0] 
			if point_val == 0: 
				dx += 1 
			elif point_val == 1: 
				dy += 1 
			elif point_val == 2: 
				dx += -1
			elif point_val == 3: 
				dy += -1 
			elif point_val == 254: 
				stop = True 
				shifts[n, 0] = dx 
				shifts[n, 1] = dy 
			print(point_val) 
			# plt.close()
			print(shifts) 
		plt.close() 
	
	image_0 = plt.imread(images[0]) 
	image_final = copy.deepcopy(image_0) 
	for i in range(0, len(image_final[:, 0, 0])): 
		for j in range(0, len(image_final[0, :, 0])):
			image_final[i, j, 0] = 0 
			image_final[i, j, 1] = 0 
			image_final[i, j, 2] = 0 
	for n in range(0, len(images)): 
		dx_max = int(np.max(np.abs(shifts[:, 0])))
		dy_max = int(np.max(np.abs(shifts[:, 1])))
		dx = int(shifts[n, 0]) 
		dy = int(shifts[n, 1])
		# print(dx_max) 
		# image_n = plt.imread(images[n]) 
		# length_images = len(images) 
		# for i in range(0 + dx_max, len(image_final[:, 0, 0]) - dx_max - 2): 
		# 	for j in range(0 + dy_max, len(image_final[0, :, 0]) - dy_max - 2): 
		# 		image_final[i, j, 0] += image_n[int(i + dx), int(j + dy), 0] / length_images
		# 		image_final[i, j, 1] += image_n[int(i + dx), int(j + dy), 1] / length_images
		# 		image_final[i, j, 2] += image_n[int(i + dx), int(j + dy), 2] / length_images
		# 	print(i / (len(image_final[:, 0, 0]) - dx_max - 2))
		length_images = len(images) 
		image_n = plt.imread(images[n]) 
		image_final_sub = np.copy(image_final[0 + dx_max:len(image_final[:, 0, 0]) - dx_max - 2, 0 + dy_max:len(image_final[0, :, 0]) - dy_max - 2, :]) 
		image_n_sub = np.copy(image_n[0 + dx_max + dx:len(image_final[:, 0, 0]) - dx_max - 2 + dx, 0 + dy_max + dy:len(image_final[0, :, 0]) - dy_max - 2 + dy, :]) / length_images
		image_final_sub_add = image_final_sub + image_n_sub 
		# plt.imshow(image_final_sub)
		# plt.show()  
		# plt.imshow(image_n_sub)
		# plt.show() 
		# plt.imshow(image_final_sub_add) 
		# plt.show()
		image_final[0 + dx_max:len(image_final[:, 0, 0]) - dx_max - 2, 0 + dy_max:len(image_final[0, :, 0]) - dy_max - 2, :] = image_final_sub_add
		
	fig = plt.figure(figsize=(20,20))
	plt.imshow(image_final)
	plt.show() 
	plt.imsave('combined.tiff', image_final) 
	plt.close() 
		
		
	
	    # manual adjust works well ################################################
    # image_0 = plt.imread(filename + n_0_str + 'a_s.tiff') 
    # image_0 = image_0[0:2760, :, :]
    
    # image_1 = plt.imread(filename + '003' + 'a_s.tiff') 
    # image_1 = image_1[0:2760, :, :]
    # image_final = copy.deepcopy(image_0) 
    
    # xc = 1500
    # yc = 1100
    # image_0 = np.copy(image_0[xc - 100:xc + 100, yc - 100:yc + 100, :]) 
    # image_1 = np.copy(image_1[xc - 100:xc + 100, yc - 100:yc + 100, :])
    # image_final = np.copy(image_1)
    # dx = 17
    # dy = -13
    # for i in range(1 + np.abs(dx), len(image_final[:, 0, 0]) - np.abs(dx) - 2): 
    #     for j in range(1 + np.abs(dy), len(image_final[0, :, 0]) - np.abs(dy) - 2): 
    #         # image_final[i, j, 0] = image_1[int(i + dx), int(j + dy), 0] * 0.499 + image_0[i, j, 0] * 0.499 
    #         # image_final[i, j, 1] = image_final[i, j, 0] 
    #         # image_final[i, j, 2] = image_final[i, j, 0]
    #         image_final[i, j, 0] = image_1[int(i + dx), int(j + dy), 0] 
    #         image_final[i, j, 1] = image_0[i, j, 0] 
    #         image_final[i, j, 2] = 0
    
    # plt.figure(figsize=(20, 20))
    # plt.imshow(image_final) 
    # plt.show() 
	
def fix_combine(filename): 
	image = plt.imread(filename) 
	plt.imsave(filename[:-1], image) 


# Combine images with drift compensation
# combine_images('050') 


# Fix image file if needed (likely due to tiff / tif errors 
# Image_File = '50nm_0p1.tiff'
# fix_combine(Image_File) 


# Image analysis: 
# Image_File = '015.tiff' 
# smoothings = 4
# analysis0(Image_File, smoothings + 1)
# analysis1v3(Image_File, 1 + smoothings, 130) 
# analysis2v2(Image_File, smoothings + 2) # Not necessary??? 
# analysis3(Image_File, smoothings + 5) 
# analysis4v4(Image_File, smoothings + 11) 
# analysis5v2(Image_File, smoothings + 14) 




