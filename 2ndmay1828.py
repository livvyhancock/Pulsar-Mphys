import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from itertools import chain
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.interpolate import CubicSpline

folder_path_new = '/Users/livvy/Documents/Pulsars project/B1828-11/stack.txt'
mjd_path_new = '/Users/livvy/Documents/Pulsars project/B1828-11/obsinfo.txt'
spin_down_new = '/Users/livvy/Documents/Pulsars project/B1828-11/nudot.asc'

num_comp = 4
rows_to_drop = []
SNr_values = []
SNr_cutoff = 10
subset_new = np.arange(475,543)
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
    plt.title(f'Horizontally Stacked Intensity Curves reconstruction with {num_comp} PCs')
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
    interval = (final - initial) / (num_values - 1)
    print(interval)
    #evenly_spaced_mjd = np.round(np.linspace(initial, final, num_values)).astype(int)
    evenly_spaced_mjd = np.arange(initial, final + 2, interval, dtype=int)
    print(f'{evenly_spaced_mjd} = evenly spaced days')
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
            axes[i].set_ylabel(f'PC{n}')
            axes[i].xaxis.set_visible(False)
        fig.suptitle('Principal components')
        plt.subplots_adjust(hspace=0)
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()


def go_into_obs(dataset,start,stop):
    for i in range(start,stop):
        plt.plot(np.arange(len(dataset.iloc[i,:])), dataset.iloc[i,:])
        plt.show()


def resample_using_cubic_spline(x1, y1, x2):
    """
    Resample the dataset represented by (x1, y1) to match the x-values in x2
    using cubic spline interpolation.

    Parameters:
        x1 (array-like): x-values of the original dataset.
        y1 (array-like): y-values of the original dataset.
        x2 (array-like): x-values of the target dataset.

    Returns:
        array-like: Resampled y-values corresponding to x2.
    """
    x1 = np.squeeze(x1)
    # Create cubic spline interpolation using the original dataset
    cs = CubicSpline(x1, y1)

    # Evaluate the spline function at the x-values of the target dataset
    y2_resampled = cs(x2)

    return y2_resampled

def calc_srcc(array1,array2):
    array1 = np.array(array1)
    array2 = np.array(array2)
    correlation_coefficient1, _ = spearmanr(array1, array2)

    #bootstrapping to find uncertainty
    num_bootstrap_samples = 1000
    correlation_coefficients = []
    for _ in range(num_bootstrap_samples):
        # Resample array1 with replacement
        resampled_array1_indices = np.random.choice(len(array1), size=len(array1), replace=True)
        resampled_array1 = array1[resampled_array1_indices]

        # Calculate correlation coefficient for the resampled arrays
        correlation_coefficient, _ = spearmanr(resampled_array1, array2)
        correlation_coefficients.append(correlation_coefficient)

    correlation_coefficients = np.array(correlation_coefficients)
    uncertainty = np.std(correlation_coefficients)
    mean = np.mean(correlation_coefficients)
    return(correlation_coefficient1,uncertainty,mean)


def calc_pearson(array1, array2):
    array1 = np.array(array1)
    array2 = np.array(array2)
    correlation_coefficient1, _ = pearsonr(array1, array2)

    # Bootstrapping to find uncertainty
    num_bootstrap_samples = 1000
    correlation_coefficients = []
    for _ in range(num_bootstrap_samples):
        # Resample array1 with replacement
        resampled_array1_indices = np.random.choice(len(array1), size=len(array1), replace=True)
        resampled_array1 = array1[resampled_array1_indices]

        # Calculate correlation coefficient for the resampled arrays
        correlation_coefficient, _ = pearsonr(resampled_array1, array2)
        correlation_coefficients.append(correlation_coefficient)

    correlation_coefficients = np.array(correlation_coefficients)
    uncertainty = np.std(correlation_coefficients)
    mean_bootstrap = np.mean(correlation_coefficients)
    return correlation_coefficient1, uncertainty, mean_bootstrap

#do the pca and get all the variables from it
eigenvalues_new,eigenvectors_new,PCA_matrix,variance_ratio,mean_new =  dopca(normalised_df_new,num_comp, subset_range=subset_new)

shape_parameter_new = eigenvalues_new.iloc[:,0]

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




PCA_matrix = np.array(PCA_matrix)
PCA_matrix_transposed = [[row[i] for row in PCA_matrix] for i in range(len(PCA_matrix[0]))]
PCA_matrix_transposed = np.array(PCA_matrix_transposed)
PCA_matrix_transposed_zoom = PCA_matrix_transposed[:, 59:167]



fig, axes = plt.subplots(num_comp+1, 2, figsize=(9, 6), gridspec_kw={'width_ratios': [1,0.15]})
axes[0, 0].imshow(transposed_matrix_new, cmap='inferno', aspect='auto')
axes[0, 0].set_ylabel('Phase')
#axes[0, 0].yaxis.set_ticks([])
axes[0, 0].twinx().yaxis.set_ticks([])
axes[0, 0].twinx().set_ylabel('Original', rotation=270, labelpad=15)
axes[0, 0].xaxis.set_visible(False)


axes[0, 1].plot(mean_new,np.arange(len(mean_new)))
axes[0, 1].invert_yaxis()
#axes[0, 1].set_ylabel('')
axes[0, 1].xaxis.set_visible(False)


for i in range(num_comp):
    single_eval_new = pd.DataFrame(eigenvalues_new.iloc[:, i])
    single_evec_new = pd.DataFrame(eigenvectors_new.iloc[i ,:]).T
    single_recon_new = single_eval_new @ single_evec_new
    even_single_recon_new = createeven_df_andtranspose(MJD_new,evenly_spaced_mjd_new,single_recon_new)

    extent = [evenly_spaced_mjd_new[0], evenly_spaced_mjd_new[-1], 0, 200]
    axes[i+1, 0].imshow(even_single_recon_new, cmap='inferno', aspect='auto',extent=extent)
    axes[i+1, 0].set_ylabel(f'Phase')
    axes[i+1, 0].grid(False)
    axes[i+1, 0].xaxis.set_visible(False)
    axes[i+1, 0].yaxis.set_ticks([])
    axes[i+1, 0].twinx().set_ylabel(f'PC{i +1 }', rotation=270, labelpad=15)

    axes[i+1, 1].plot(eigenvectors_new.iloc[i,:],np.arange(len(eigenvectors_new.iloc[i,:])))
    axes[i+1, 1].invert_yaxis()
    axes[i+1, 1].set_xlabel('phase')
    axes[i+1, 1].grid(False)
    axes[i+1, 1].xaxis.set_visible(False)

axes[i+1, 0].xaxis.set_visible(True)
axes[i+1, 0].set_xlabel('Modified Julian Day')
fig.suptitle('Principal components B1828-11')
im = axes[0, 0].imshow(transposed_matrix_new, cmap='inferno', aspect='auto')
cbar_ax = fig.add_axes([0.05, 1, 0.3, 0.015])
fig.colorbar(im,cax=cbar_ax, orientation='horizontal')
plt.subplots_adjust(top=0.9,hspace=0)
plt.tight_layout()  # Adjust layout to prevent overlap
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

single_eval_new = pd.DataFrame(eigenvalues_new.iloc[:, 0])
single_evec_new = pd.DataFrame(eigenvectors_new.iloc[0 ,:]).T
single_recon_new = single_eval_new @ single_evec_new
even_single_recon_new = createeven_df_andtranspose(MJD_new,evenly_spaced_mjd_new,single_recon_new)

# Plot the PCA 2nd sublot
axes[2].imshow(even_single_recon_new, cmap='inferno', aspect='auto')
axes[2].set_xlabel('MJD')
axes[2].set_ylabel(f'PCA w/ {num_comp} components')
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




shape_parameter_new = eigenvalues_new.iloc[:,0]
shape_parameter_new1 = eigenvalues_new.iloc[:,1]
shape_parameter_new2 = eigenvalues_new.iloc[:,2]
shape_parameter_new3 = eigenvalues_new.iloc[:,3]

nudot_resampled = resample_using_cubic_spline(spindown_MJD_new, spindown_rate_new , MJD_new)

spindown_rate_new = np.array(spindown_rate_new)
spindown_rate_new = spindown_rate_new.reshape(1000)

pearson1,uncertainty1_new, mp1 = calc_pearson(nudot_resampled,shape_parameter_new)
pearson2,uncertainty2_new, mp2 = calc_pearson(nudot_resampled,shape_parameter_new1)
pearson3,uncertainty3_new, mp3  = calc_pearson(nudot_resampled,shape_parameter_new2)
pearson4,uncertainty4_new, mp4  = calc_pearson(nudot_resampled,shape_parameter_new3)

srcc1_new,suncertainty1_new, ms1 = calc_srcc(nudot_resampled,shape_parameter_new)
srcc2_new,suncertainty2_new, ms2 = calc_srcc(nudot_resampled,shape_parameter_new1)
srcc3_new,suncertainty3_new, ms3 = calc_srcc(nudot_resampled,shape_parameter_new2)
srcc4_new,suncertainty4_new, ms4 = calc_srcc(nudot_resampled,shape_parameter_new3)
#SP_1_2_CORR_new,unc12_new = calc_srcc(shape_parameter_new,shape_parameter_new1)
#SP_2_3_CORR_new, unc23_new = calc_srcc(shape_parameter_new1,shape_parameter_new2)



print(f'pearson SP1 {pearson1}±{uncertainty1_new}   (mean = {mp1})')
print(f'pearson SP2 {pearson2}±{uncertainty2_new}   (mean = {mp2})')
print(f'pearson SP3 {pearson3}±{uncertainty3_new}   (mean = {mp3})')
print(f'pearson SP4 {pearson4}±{uncertainty4_new}   (mean = {mp4})')
#print(f'{SP_1_2_CORR}±{unc12}   (mean = {pm1})')
#print(f'{SP_2_3_CORR}±{unc23}   (mean = {pm2})')

print(f'Spearman SP1 {srcc1_new}±{suncertainty1_new}   (mean = {ms1})')
print(f'Spearman SP2 {srcc2_new}±{suncertainty1_new}   (mean = {ms2})')
print(f'Spearman SP3 {srcc3_new}±{suncertainty1_new}   (mean = {ms3})')
print(f'Spearman SP4 {srcc4_new}±{suncertainty1_new}   (mean = {ms4})')
#print(f'{sSP_1_2_CORR}±{sunc12}   (mean = {sm1})')
#print(f'{sSP_2_3_CORR}±{sunc23}   (mean = {sm2})')

fig, axes = plt.subplots(3, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [0.5,0.5,0.5]})
# Plot the spindown dfb on the second subplot
axes[0].plot(MJD_new, shape_parameter_new, color='red')
axes[0].set_ylabel(f'shape parameter')
axes[0].set_xlim(evenly_spaced_mjd_new[0], evenly_spaced_mjd_new[-1])
axes[0].xaxis.set_visible(False)

axes[1].plot(spindown_MJD_new, nudot_resampled, color='red')
axes[1].set_ylabel(f'resampled shape parameter')
axes[1].set_xlim(evenly_spaced_mjd_new[0], evenly_spaced_mjd_new[-1])
axes[1].xaxis.set_visible(False)

# Plot the spindown dfb on the second subplot
axes[2].plot(spindown_MJD_new, nudot_resampled, color='red')
axes[2].set_ylabel(f'spindown rate')
axes[2].set_xlim(evenly_spaced_mjd_new[0], evenly_spaced_mjd_new[-1])
axes[2].xaxis.set_visible(False)


fig.suptitle(f'demonstration of resampling for correlation coeff.', fontsize=16)
plt.subplots_adjust(hspace=0)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()