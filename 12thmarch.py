import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm

# Specify the path to the first folder containing your text files
folder_path_afb = '/Users/livvy/Documents/Pulsars project/B1822-09_afb_dataset/profile_data_afb'
spin_down_data_afb = '/Users/livvy/Documents/Pulsars project/B1822-09_spin_down.txt'

# Specify the path to the second folder containing your text files
folder_path_dfb = '/Users/livvy/Downloads/B1822-09_dataset 2/profile_data'
spin_down_data_dfb = '/Users/livvy/Documents/Pulsars project/mjd_nudot_err.txt'

# Create empty lists to store DataFrames and file order for the first folder
all_dataframes_afb = []
file_order_afb = []
MJD_afb = []

# Loop through each file in the first folder in ascending order
for filename_afb in sorted(os.listdir(folder_path_afb)):
    if filename_afb.endswith('.txt'):
        file_order_afb.append(filename_afb)
        file_path_afb = os.path.join(folder_path_afb, filename_afb)
        df_afb = pd.read_csv(file_path_afb, header=None, delimiter='\t')
        values_afb = df_afb.values.flatten()
        all_dataframes_afb.append(values_afb)

        # get MJD from the filenames
        extracted_chars_afb = filename_afb.replace('B1822-09_', '').replace('.txt', '')
        num_value_afb = float(extracted_chars_afb)
        num_list_afb = num_value_afb
        MJD_afb.append(num_list_afb)

# Read in spin down data for the first folder
columns_afb = ['MJD', 'Spin-down rate (1e-15 Hz/s)', 'Spin-down rate uncertainty (1e-15 Hz/s)']
spin_down_df_afb = pd.read_csv(spin_down_data_afb, delim_whitespace=True, names=columns_afb)

# Create DataFrame for the first folder
raw_df_afb = pd.DataFrame(all_dataframes_afb)

# Create empty lists to store DataFrames and file order for the second folder
all_dataframes_dfb = []
file_order_dfb = []
MJD_dfb = []

# Loop through each file in the second folder in ascending order
for filename_dfb in sorted(os.listdir(folder_path_dfb)):
    if filename_dfb.endswith('.txt'):
        file_order_dfb.append(filename_dfb)
        file_path_dfb = os.path.join(folder_path_dfb, filename_dfb)
        df_dfb = pd.read_csv(file_path_dfb, header=None, delimiter='\t')
        values_dfb = df_dfb.values.flatten()
        all_dataframes_dfb.append(values_dfb)

        # get MJD from the filenames
        extracted_chars_dfb = filename_dfb.replace('B1822-09_', '').replace('.txt', '')
        num_value_dfb = float(extracted_chars_dfb)
        num_list_dfb = num_value_dfb
        MJD_dfb.append(num_list_dfb)

# Read in spin down data for the second folder
columns_dfb = ['MJD', 'Spin-down rate (1e-15 Hz/s)', 'Spin-down rate uncertainty (1e-15 Hz/s)']
spin_down_df_dfb = pd.read_csv(spin_down_data_dfb, delim_whitespace=True, names=columns_afb)

# Create DataFrame for the second folder
raw_df_dfb = pd.DataFrame(all_dataframes_dfb)

#drop rows
#rows_to_drop = [6, 64, 137, 147, 227, 298, 416,417, 432, 433, 440, 442, 446,448, 449, 452, 453, 460,461, 468, 504, 606, 622, 691, 720, 722, 818, 821, 836, 839, 846]
rows_to_drop = [6, 64, 137, 147, 227, 298, 412,414,415,417,421,422,423,424,425,428,429,430, 432, 433, 440, 442, 446,448, 449, 452, 453, 460,461, 468, 504, 606, 622, 691, 720, 722, 818, 821, 836, 839, 846]
#additional_rows_to_drop = list(range(400, 431))
#rows_to_drop.extend(additional_rows_to_drop)
raw_df_dfb = raw_df_dfb.drop(rows_to_drop)

MJD_dfb = pd.DataFrame(MJD_dfb)
MJD_dfb = MJD_dfb.drop(rows_to_drop)




def plotstackedintensities(matrix,title):
    #Display the stacked pca reconstruction
    #n_components = 2
    plt.figure(figsize=(10, 6))
    plt.imshow(((matrix)), cmap='inferno', aspect='auto')
    plt.colorbar()
    plt.title(f'{title}')
    plt.xlabel('Phase bin')
    plt.ylabel('Observation Number')
    plt.show()
    return

def gaussian_func(x, mu, sigma, A):
    return A*norm.pdf(x, loc=mu, scale=sigma)


def normbygaussian_afb(dataframe):
    raw_df = dataframe
    peak_subset_df = dataframe.iloc[:, np.r_[94:120]]
    bin_averages = np.mean(peak_subset_df, axis=0)                #get profile shape
    scaler = []
    ideal_params, covariance = curve_fit(gaussian_func, np.arange(len(bin_averages)), bin_averages)
    mu_fit = ideal_params[0]
    sigma_fit = ideal_params[1]
    A_fit = ideal_params[2]
    for i in range(0,len(raw_df.iloc[:,1])):
        ideal_params_obs, covarianceobs = curve_fit(gaussian_func, np.arange(len(peak_subset_df.iloc[i,:])), peak_subset_df.iloc[i,:], maxfev = 5000)
        A_fit_obs = ideal_params_obs[2]
        scaling = A_fit / A_fit_obs
        scaler.append(scaling)
    normalised_by_fit_df = raw_df.mul(scaler, axis=0)
    return(normalised_by_fit_df)

def normbygaussian_dfb(dataframe):
    raw_df = dataframe
    #peak_subset_df = dataframe.iloc[:, np.r_[94:120]] #FOR AFB!!!!
    peak_subset_df = dataframe.iloc[:, np.r_[240:290]]
    bin_averages = np.mean(peak_subset_df, axis=0)                #get profile shape
    scaler = []

    ideal_params, covariance = curve_fit(gaussian_func, np.arange(len(bin_averages)), bin_averages)
    mu_fit = ideal_params[0]
    sigma_fit = ideal_params[1]
    A_fit = ideal_params[2]

    def new_gaussian_func(x, A):
        return A*norm.pdf(x, loc=mu_fit, scale=sigma_fit)

    for i in range(0,len(raw_df.iloc[:,1])):
        ideal_params_obs, covarianceobs = curve_fit(new_gaussian_func, np.arange(len(peak_subset_df.iloc[i,:])), peak_subset_df.iloc[i,:], maxfev = 5000)
        A_fit_obs = ideal_params_obs[0]
        scaling = A_fit / A_fit_obs
        scaler.append(scaling)

    normalised_by_fit_df = raw_df.mul(scaler, axis=0)
    return(normalised_by_fit_df)

def dopca(dataset,n_components):
    '''use:
    pcaresult = dopca(dfaveragenorm,2)
    plotPCsandshape4(pcaresult[1].iloc[0,:], pcaresult[1].iloc[1,:], pcaresult[0],pcaresult[2],days)
     where: plotPCsandshape4(pc1, pc2, EIGENVALUES, matrix, days)
    '''
    pca = PCA(n_components=n_components) #defines the pca action
    principalComponents = pca.fit_transform(dataset)
    evals = pd.DataFrame(data=principalComponents)
    evecs = pd.DataFrame(data=pca.components_)
    varianceratio = pca.explained_variance_ratio_
    mean = pca.mean_
    reconstructeddf = mean + evals @ evecs
    return(evals,evecs,reconstructeddf,varianceratio)

def smooth_sp_and_days(days, shape_parameters, window_size):
    num_days = int(days[len(days)-1] - days[0]) +1
    overlap = window_size // 4 #step size
    num_windows = num_days - window_size // overlap + 1


    mean_shape_parameters = []
    mean_day_values = []

    for i in range(num_windows):
        start_day = days[0] + i * overlap
        end_day = start_day + window_size

        window_indices = [j for j, day in enumerate(days) if start_day <= day < end_day]

        window_shape_parameters = [shape_parameters[j] for j in window_indices]

        if window_shape_parameters:
            mean_shape_parameter = np.mean(window_shape_parameters)
            mean_shape_parameters.append(mean_shape_parameter)

            mean_day_value = start_day + (window_size / 2)
            mean_day_values.append(mean_day_value)

    return(mean_day_values,mean_shape_parameters)

#need to normalise, get pca reconstructions and then plot against spin down rate

#assign values:
normalised_df_afb = normbygaussian_afb(raw_df_afb)
# days array is called MJD_afb
normalised_df_dfb = normbygaussian_dfb(raw_df_dfb)
# days array is called MJD_dfb

all_shape_parameter_afb = dopca(normalised_df_afb,2)[0] #perform pca on afb
shape_parameter_afb = all_shape_parameter_afb.iloc[:,0]
#plotstackedintensities(dopca(normalised_df_afb,2)[2]) #plot stacked intensities to check pca working as expected


all_shape_parameter_dfb = dopca(normalised_df_dfb,2)[0] #perform pca on dfb
shape_parameter_dfb = all_shape_parameter_dfb.iloc[:,0]
#plotstackedintensities(dopca(normalised_df_dfb,2)[2]) #plot stacked intensities to check pca working as expected




spindown_MJD_afb = spin_down_df_afb.iloc[:,0]
spindown_rate_afb = spin_down_df_afb.iloc[:,1]

spindown_MJD_dfb = spin_down_df_dfb.iloc[:,0]
spindown_rate_dfb = spin_down_df_dfb.iloc[:,1]

#assign smoothed values
window = 100
smoothed_spindown_MJD_afb = smooth_sp_and_days(spindown_MJD_afb,spindown_rate_afb, window)[0]
smoothed_spindown_rate_afb = smooth_sp_and_days(spindown_MJD_afb,spindown_rate_afb, window)[1]
smoothed_MJD_afb = smooth_sp_and_days(MJD_afb,shape_parameter_afb, window)[0]
smoothed_shape_parameter_afb = smooth_sp_and_days(MJD_afb,shape_parameter_afb, window)[1]



plt.figure(figsize=(10, 6))
plt.plot(MJD_dfb, shape_parameter_dfb)
plt.xlabel('MJD')
plt.ylabel('shape parameter')
plt.title(f'shape parameter')
plt.show()


#plotspin down against pca shape param for dfb

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6), frameon=False)
# Plot for Lyne data
#ax1.plot(spindown_MJD_afb, spindown_rate_afb, label='Lyne', color='b')#unsmoothed
ax1.plot(spindown_MJD_dfb.iloc[763:,], spindown_rate_dfb[763:], label='Lyne', color='b')#smoothed
ax1.set_ylabel('Spin Down Rate', color='b')
ax1.tick_params('y', colors='b')

# Plot for PCA data
ax2.plot(MJD_dfb, shape_parameter_dfb, label='PCA', color='r') #smoothed
#ax2.plot(spindown_MJD_afb, shape_parameter_afb, label='PCA', color='r') unsmoothed
ax2.set_ylabel('Shape Parameter', color='r')
ax2.set_xlabel('MJD', color='black')
ax2.tick_params('y', colors='r')



