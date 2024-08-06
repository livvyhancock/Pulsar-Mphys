import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from itertools import chain


folder_path_new = '/Users/livvy/Documents/Pulsars project/B1828-11/stack.txt'
mjd_path_new = '/Users/livvy/Documents/Pulsars project/B1828-11/obsinfo.txt'
spin_down_new = '/Users/livvy/Documents/Pulsars project/B1828-11/nudot.asc'
n_components = 1
rows_to_drop = []
SNr_values = []
SNr_cutoff = 10
subset_new = np.arange(500,530)
window = 100

def read_in_file(folder_path, mjd_path):
    with open(folder_path, 'r') as file:
        lines = file.readlines()[1:]

    # Extract the last column from each line
    last_column_values = [line.strip().split()[-1] for line in lines]

    # Split the last column values into chunks of 1024 elements
    split_last_column = [last_column_values[i:i+1024] for i in range(0, len(last_column_values), 1024)]

    # Create a DataFrame from the split last column
    raw_df = pd.DataFrame(split_last_column).apply(pd.to_numeric)

    with open(mjd_path, 'r') as second_file:
        second_lines = second_file.readlines()[1:]

    second_column_values = [line.strip().split()[1] for line in second_lines]

    MJD = pd.DataFrame(second_column_values, columns=['Second_Column']).apply(pd.to_numeric)
    return raw_df, MJD

raw_df_new, MJD_new = read_in_file(folder_path_new, mjd_path_new)

def plotstackedintensities(matrix):
    #Display the stacked pca reconstruction

    plt.figure(figsize=(10, 6))
    plt.imshow(((matrix)), cmap='inferno', aspect='auto')
    plt.colorbar()
    plt.title(f'Horizontally Stacked Intensity Curves reconstruction with {n_components} PCs')
    plt.xlabel('Phase bin')
    plt.ylabel('Observation Number')
    plt.show()
    return

def gaussian_func(x, mu, sigma, A):
    return A*norm.pdf(x, loc=mu, scale=sigma)



def normbygaussian_new(dataframe):
    raw_df = dataframe
    peak_subset_df = dataframe.iloc[:, np.r_[500:530]]
    bin_averages = np.mean(peak_subset_df, axis=0)  # Get profile shape


    scaler = []

    ideal_params, covariance = curve_fit(gaussian_func, np.arange(len(bin_averages)), bin_averages)
    mu_fit = ideal_params[0]
    sigma_fit = ideal_params[1]
    A_fit = ideal_params[2]

    def new_gaussian_func(x, A):
        return A * norm.pdf(x, loc=mu_fit, scale=sigma_fit)

    for i in range(0, len(raw_df.iloc[:, 1])):
        ideal_params_obs, covarianceobs = curve_fit(new_gaussian_func, np.arange(len(peak_subset_df.iloc[i, :])),
                                                    peak_subset_df.iloc[i, :], maxfev=5000)
        A_fit_obs = ideal_params_obs[0]
        scaling = A_fit / A_fit_obs
        scaler.append(scaling)

    normalised_by_fit_df = raw_df.mul(scaler, axis=0)
    return normalised_by_fit_df



def dopca(dataset,n_components, subset_range):
    '''use:
    pcaresult = dopca(dfaveragenorm,2)
    plotPCsandshape4(pcaresult[1].iloc[0,:], pcaresult[1].iloc[1,:], pcaresult[0],pcaresult[2],days)
     where: plotPCsandshape4(pc1, pc2, EIGENVALUES, matrix, days)
    '''
    dataset = pd.DataFrame(dataset)
    if subset_range is not None and subset_range.size > 0:
        dataset = dataset.iloc[:, subset_range]
    else:
        dataset = dataset

    pca = PCA(n_components=n_components) #defines the pca action
    principalComponents = pca.fit_transform(dataset)
    evals = pd.DataFrame(data=principalComponents)
    evecs = pd.DataFrame(data=pca.components_)
    varianceratio = pca.explained_variance_ratio_
    mean = pca.mean_
    reconstructeddf = mean + evals @ evecs
    return(evals,evecs,reconstructeddf,varianceratio,mean)


def smooth_sp_and_days(days, shape_parameters, window_size):
    # Convert DataFrame columns to 1D arrays
    #days_array = days.iloc[:, 0].values
    days_array = np.squeeze(np.array(days))
    shape_parameters_array = shape_parameters.values

    num_days = int(days_array[-1] - days_array[0]) + 1
    overlap = window_size // 4  # step size
    num_windows = (num_days - window_size) // overlap + 1

    mean_shape_parameters = []
    mean_day_values = []

    for i in range(num_windows):
        start_day = days_array[0] + i * overlap
        end_day = start_day + window_size

        window_indices = [j for j, day in enumerate(days_array) if start_day <= day < end_day]

        window_shape_parameters = shape_parameters_array[window_indices]

        if window_shape_parameters.size > 0:  # Check if the array is not empty
            mean_shape_parameter = np.mean(window_shape_parameters)
            mean_shape_parameters.append(mean_shape_parameter)

            mean_day_value = start_day + (window_size / 2)
            mean_day_values.append(mean_day_value)

    return mean_day_values, mean_shape_parameters

normalised_df_new = normbygaussian_new(raw_df_new)

for i in range(len(normalised_df_new)):
    obs = normalised_df_new.iloc[i, :]

    # Exclude columns within subset_dfb
    noise_region = obs.loc[~obs.index.isin(subset_new)]

    noise = np.std(noise_region)
    signal = np.max(obs)  # Calculate peak value of pulse
    SNr = signal / noise
    SNr_values.append(SNr)

    # Check if any SNr value is less than SNr_cutoff
    if (SNr < SNr_cutoff):
        rows_to_drop.append(i)



normalised_df_new = normalised_df_new.drop(rows_to_drop)
MJD_new = MJD_new.drop(rows_to_drop)
MJD_new = np.array(MJD_new)
MJD_new = list(chain.from_iterable(MJD_new))
MJD_new = np.round(MJD_new).astype(int)


def read_asc_file(file):
    spindown_MJD = []
    spindow_rate = []
    with open(file, 'r') as file:
        for line in file:
            # Split the line by spaces
            values = line.strip().split()
            # Extract the first and fourth values
            if len(values) >= 4:
                spindown_MJD.append(float(values[0]))
                spindow_rate.append(float(values[3]))
    return spindown_MJD , spindow_rate

def get_evenly_spaced_days(mjd):
    mjd = np.array(mjd)
    initial = mjd[0]
    final = mjd[len(mjd)-1]
    #num_values = len(mjd)
    num_values = final - initial
    #evenly_spaced_mjd = np.linspace(initial, final, num=num_values)
    evenly_spaced_mjd = np.round(np.linspace(initial, final, num_values)).astype(int)


    new_mjd = []
    for entry in evenly_spaced_mjd:

        given_value = entry
        absolute_diff = np.abs(given_value -  mjd)
        closest_index = np.argmin(absolute_diff)
        closest_day = mjd[closest_index]
        #print(f'closest_day= {closest_day}')
        new_mjd.append(closest_day)
    return(np.array(new_mjd).flatten())


def createeven_df_andtranspose(days, evendays, normaliseddf):
    normaliseddf = pd.DataFrame(normaliseddf)
    new_matrix = []
    for entry in evendays:
        #find row with nearest
        given_value = entry
        absolute_diff = np.abs(days - given_value)
        index = np.argmin(absolute_diff) #finds the row index with the day nearest to that of even day

        row_to_append = normaliseddf.iloc[index]
        row_to_append = np.array(row_to_append)
        new_matrix.append(row_to_append) #this is the normalised df in the even spacing but not transposed

    transposed_matrix = [[row[i] for row in new_matrix] for i in range(len(new_matrix[0]))]
    transposed_matrix = np.array(transposed_matrix)
    return(np.array(transposed_matrix))

def plotpcs(MJD, evenMJD, data, n, subset_range):
    eigenvalues,eigenvectors,reconstruction,exp_var,mean = dopca(data, n, None)

    data = pd.DataFrame(data)
    if subset_range is not None and subset_range.size > 0:
        data = data.iloc[:, subset_range]
    else:
        data = data
    transposed_matrix = createeven_df_andtranspose(MJD, evenMJD, data)

    fig, axes = plt.subplots(n+1, 2, figsize=(10, 6))
    axes[0, 0].imshow(transposed_matrix, cmap='inferno', aspect='auto')
    axes[0, 0].set_ylabel('reconstruction')
    axes[0, 0].xaxis.set_visible(False)
    axes[0, 0].yaxis.set_ticks([])
    axes[0, 1].plot(np.arange(len(mean)),mean)
    axes[0, 1].set_ylabel('mean')
    axes[0, 1].xaxis.set_visible(False)

    for i in range(n):
        single_eval = pd.DataFrame(eigenvalues.iloc[:, i])
        single_evec = pd.DataFrame(eigenvectors.iloc[i ,:]).T
        single_recon = single_eval @ single_evec
        even_single_recon = createeven_df_andtranspose(MJD, evenMJD,single_recon)

        axes[i+1, 0].imshow(even_single_recon, cmap='inferno', aspect='auto')
        axes[i+1, 0].set_xlabel('MJD')
        axes[i+1, 0].set_ylabel(f'PC{i +1 }')
        axes[i+1, 0].grid(False)
        axes[i+1, 0].xaxis.set_visible(False)
        axes[i+1, 0].yaxis.set_ticks([])

        axes[i+1, 1].plot(np.arange(len(eigenvectors.iloc[i,:])),eigenvectors.iloc[i,:])
        axes[i+1, 1].set_xlabel('phase')
        axes[i+1, 1].grid(False)
        axes[i+1, 1].xaxis.set_visible(False)

    fig.suptitle('Principal components B1822−09 dfb')
    im = axes[0, 0].imshow(transposed_matrix, cmap='inferno', aspect='auto')
    cbar_ax = fig.add_axes([0.05, 1, 0.3, 0.015])
    fig.colorbar(im,cax=cbar_ax, orientation='horizontal')
    plt.subplots_adjust(top=0.9,hspace=0)
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
    return()

def plot_pcs(dataset,n):
    eigenvectors = dopca(dataset, n, None)[1]

    if n == 1:
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(eigenvectors.iloc[0,:])),eigenvectors.iloc[0,:])
        plt.xlabel('phase')
        plt.ylabel(f'PC{n}')
        plt.title('Principal components')

    else:

        fig, axes = plt.subplots(n, 1, figsize=(10, 6))
        for i in range(n):
            axes[i].plot(np.arange(len(eigenvectors.iloc[i,:])),eigenvectors.iloc[i,:])
            axes[i].set_xlabel('phase')
            axes[i].set_ylabel(f'PC{i}')
        fig.suptitle('Principal components')
        plt.subplots_adjust(hspace=0)
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

def go_into_obs(dataset,start,stop):
    for i in range(start,stop):
        plt.plot(np.arange(len(dataset.iloc[i,:])), dataset.iloc[i,:])
        plt.show()


all_shape_parameter_new = dopca(normalised_df_new,n_components, subset_range=subset_new)[0] #perform pca on afb
shape_parameter_new = all_shape_parameter_new.iloc[:,0]
plotstackedintensities(dopca(normalised_df_new,2,subset_range=subset_new)[2])

spindown_MJD, spindown_rate = read_asc_file(spin_down_new)
spindown_MJD_new = pd.DataFrame(spindown_MJD, columns=['MJD'])
spindown_rate_new = pd.DataFrame(spindown_rate, columns=['Rate'])
smoothed_spindown_MJD_new, smoothed_spindown_rate_new = smooth_sp_and_days(spindown_MJD_new, spindown_rate_new, window)
smoothed_MJD_new, smoothed_shape_parameter_new = smooth_sp_and_days(MJD_new, shape_parameter_new, window)


#get horizontal and day spaced intensities
evenly_spaced_mjd_new = get_evenly_spaced_days(MJD_new)
normalised_df_new_subs = normalised_df_new.iloc[:, subset_new]
transposed_matrix_new = createeven_df_andtranspose(MJD_new, evenly_spaced_mjd_new, normalised_df_new_subs)
transposed_matrix_new_zoom = transposed_matrix_new[:, 59:167]



PCA_matrix = dopca(normalised_df_new,2,subset_range=subset_new)[2]
PCA_matrix = np.array(PCA_matrix)
PCA_matrix_transposed = [[row[i] for row in PCA_matrix] for i in range(len(PCA_matrix[0]))]
PCA_matrix_transposed = np.array(PCA_matrix_transposed)
PCA_matrix_transposed_zoom = PCA_matrix_transposed[:, 59:167]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6), frameon=True)
ax1.plot(spindown_MJD_new.iloc[639:,], spindown_rate_new[639:], label='Lyne', color='b')#smoothed
#ax1.plot(spindown_MJD_new, spindown_rate_new, color='b')#smoothed
ax1.set_ylabel('Spin Down Rate', color='b')
ax1.tick_params('y', colors='b')

# Plot for PCA data
ax2.plot(smoothed_MJD_new, smoothed_shape_parameter_new, label='PCA', color='r') #smoothed
#ax2.plot(spindown_MJD_afb, shape_parameter_afb, label='PCA', color='r') unsmoothed
ax2.set_ylabel('Shape Parameter', color='r')
ax2.set_xlabel('MJD', color='black')
ax2.tick_params('y', colors='r')
plt.suptitle(f'PCA smoothed Shape Parameter (n={n_components}) gauss normalisations, window size {window} days, DFB PSR B1828-11')
plt.show()


fig, axes = plt.subplots(5, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [1,0.1,1, 0.5, 0.5]})
# Plot the transposed matrix on the first subplot
axes[0].imshow(transposed_matrix_new, cmap='inferno', aspect='auto')
axes[0].set_xlabel('MJD')
axes[0].set_ylabel('normalised intensities')
axes[0].grid(False)
axes[0].xaxis.set_visible(False)

axes[1].plot()
axes[1].set_xlim(evenly_spaced_mjd_new[0], evenly_spaced_mjd_new[-1])
for x in np.array(MJD_new):
    axes[1].axvline(x)
axes[1].xaxis.set_visible(False)
axes[1].yaxis.set_visible(False)

# Plot the PCA 2nd sublot
axes[2].imshow(PCA_matrix_transposed, cmap='inferno', aspect='auto')
axes[2].set_xlabel('MJD')
axes[2].set_ylabel(f'PCA w/ {n_components} components')
axes[2].grid(False)
axes[2].xaxis.set_visible(False)

# Plot the spindown dfb on the second subplot
axes[3].plot(smoothed_MJD_new, smoothed_shape_parameter_new, color='red')
axes[3].set_ylabel(f'shape param (window= {window})')
axes[3].set_xlim(evenly_spaced_mjd_new[0], evenly_spaced_mjd_new[-1])
axes[3].xaxis.set_visible(False)

# Plot the third panel
axes[4].plot(spindown_MJD_new, spindown_rate_new, color='blue')  # Replace x_data and y_data with your data
axes[4].set_xlabel('Modified Julian Days')  # Replace with appropriate label
axes[4].set_ylabel('Spindown')  # Replace with appropriate label
axes[4].set_xlim(evenly_spaced_mjd_new[0], evenly_spaced_mjd_new[-1])

fig.suptitle('B1828-11 emission modes shown by colourmap, shape parameter and spindown rate against days', fontsize=16)
plt.subplots_adjust(hspace=0)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

#PLOT ZOOMED ONE!
fig, axes = plt.subplots(5, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1,0.1,1, 0.5, 0.5]})
# Plot the transposed matrix on the first subplot
axes[0].imshow(transposed_matrix_new_zoom, cmap='inferno', aspect='auto')
axes[0].set_xlabel('MJD')
axes[0].set_ylabel('normalised intensities')
axes[0].grid(False)
axes[0].xaxis.set_visible(False)


axes[1].plot()
axes[1].set_xlim(evenly_spaced_mjd_new[0], evenly_spaced_mjd_new[-1])
for x in np.array(MJD_new):
    axes[1].axvline(x)
axes[1].xaxis.set_visible(False)
axes[1].yaxis.set_visible(False)
axes[1].set_xlim(evenly_spaced_mjd_new[60], evenly_spaced_mjd_new[168])

# Plot the PCA 2nd sublot
axes[2].imshow(PCA_matrix_transposed_zoom, cmap='inferno', aspect='auto')
axes[2].set_xlabel('MJD')
axes[2].set_ylabel(f'PCA w/ {n_components} components')
axes[2].grid(False)
axes[2].xaxis.set_visible(False)


# Plot the spindown dfb on the second subplot
axes[3].plot(smoothed_MJD_new, smoothed_shape_parameter_new, color='red')
axes[3].set_ylabel(f'shape param (window= {window})')
axes[3].set_xlim(evenly_spaced_mjd_new[60], evenly_spaced_mjd_new[168])
axes[3].xaxis.set_visible(False)

# Plot the third panel
axes[4].plot(spindown_MJD_new, spindown_rate_new, color='blue')
axes[4].set_xlabel('Modified Julian Days')
axes[4].set_ylabel('Spindown')
axes[4].set_xlim(evenly_spaced_mjd_new[60], evenly_spaced_mjd_new[168])

fig.suptitle('B1828-11 emission modes shown by colourmap, shape parameter and spindown rate against days', fontsize=16)
plt.subplots_adjust(hspace=0)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


#PLOT ZOOMED ONE just intensities!
fig, axes = plt.subplots(4, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1,0.1, 1,0.001]})
# Plot the transposed matrix on the first subplot

im = axes[0].imshow(transposed_matrix_new_zoom, cmap='inferno', aspect='auto')
axes[0].imshow(transposed_matrix_new_zoom, cmap='inferno', aspect='auto')
axes[0].set_xlabel('MJD')
axes[0].set_ylabel('normalised intensities', fontsize=14)
axes[0].grid(False)
axes[0].xaxis.set_visible(False)

axes[1].plot()
for x in np.array(MJD_new):
    axes[1].axvline(x)
axes[1].xaxis.set_visible(False)
axes[1].yaxis.set_visible(False)
axes[1].set_xlim(evenly_spaced_mjd_new[60], evenly_spaced_mjd_new[168])
axes[1].axis('off')


# Plot the PCA 2nd sublot
im_pca = axes[2].imshow(PCA_matrix_transposed_zoom, cmap='inferno', aspect='auto')
axes[2].imshow(PCA_matrix_transposed_zoom, cmap='inferno', aspect='auto')
axes[2].set_xlabel('MJD')
axes[2].set_ylabel(f'PCA w/ {n_components} components', fontsize=14)
axes[2].grid(False)
axes[2].xaxis.set_visible(False)

axes[3].plot()
axes[3].xaxis.set_visible(True)
axes[3].yaxis.set_visible(False)
axes[3].set_xlim(evenly_spaced_mjd_new[60], evenly_spaced_mjd_new[168])
axes[3].set_xlabel('Modified Julian Days',fontsize=14)
axes[3].spines['top'].set_color('none')
axes[3].spines['right'].set_color('none')
axes[3].spines['bottom'].set_color('none')
axes[3].spines['left'].set_color('none')

# Create a colorbar for the PCA matrix and position it horizontally at the top
cbar = fig.colorbar(im, ax=axes[0], orientation='horizontal', fraction=0.04, pad=0.04, location='top')
cbar.ax.xaxis.set_ticks_position('none')  # Hide ticks on colorbar


fig.suptitle('B1828-11 emission modes shown by colourmap against days', fontsize=20)
plt.subplots_adjust(hspace=0)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


eigenvalues, eigenvectors, matrix, variance, mean = dopca(normalised_df_new, 2, None)

#now plot the PCAs and shape parameters. against MJD
fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1,1]})
# Plot the transposed matrix on the first subplot

axes[0].plot(np.arange(len(eigenvectors.iloc[0,:])),eigenvectors.iloc[0,:])
axes[0].set_xlabel('phase')
axes[0].set_ylabel('PC1')

axes[1].plot(np.arange(len(eigenvectors.iloc[0,:])),eigenvectors.iloc[1,:])
axes[1].set_xlabel('phase')
axes[1].set_ylabel('PC2')

fig.suptitle('Principal components')
plt.subplots_adjust(hspace=0)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()






