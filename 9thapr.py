import osimport pandas as pdimport numpy as npfrom sklearn.decomposition import PCAimport matplotlib.pyplot as pltfrom sklearn.preprocessing import StandardScalerimport matplotlib.pyplot as pltfrom scipy.optimize import curve_fitfrom scipy.stats import norm# Specify the path to the first folder containing your text filesfolder_path_afb = '/Users/livvy/Documents/Pulsars project/B1822-09_afb_dataset/profile_data_afb'spin_down_data_afb = '/Users/livvy/Documents/Pulsars project/B1822-09_spin_down.txt'# Specify the path to the second folder containing your text filesfolder_path_dfb = '/Users/livvy/Downloads/B1822-09_dataset 2/profile_data'spin_down_data_dfb = '/Users/livvy/Documents/Pulsars project/mjd_nudot_err.txt'# Create empty lists to store DataFrames and file order for the first folderall_dataframes_afb = []file_order_afb = []MJD_afb = []# Loop through each file in the first folder in ascending orderfor filename_afb in sorted(os.listdir(folder_path_afb)):    if filename_afb.endswith('.txt'):        file_order_afb.append(filename_afb)        file_path_afb = os.path.join(folder_path_afb, filename_afb)        df_afb = pd.read_csv(file_path_afb, header=None, delimiter='\t')        values_afb = df_afb.values.flatten()        all_dataframes_afb.append(values_afb)        # get MJD from the filenames        extracted_chars_afb = filename_afb.replace('B1822-09_', '').replace('.txt', '')        num_value_afb = float(extracted_chars_afb)        num_list_afb = num_value_afb        MJD_afb.append(round(num_list_afb))# Read in spin down data for the first foldercolumns_afb = ['MJD', 'Spin-down rate (1e-15 Hz/s)', 'Spin-down rate uncertainty (1e-15 Hz/s)']spin_down_df_afb = pd.read_csv(spin_down_data_afb, delim_whitespace=True, names=columns_afb)# Create DataFrame for the first folderraw_df_afb = pd.DataFrame(all_dataframes_afb)# Create empty lists to store DataFrames and file order for the second folderall_dataframes_dfb = []file_order_dfb = []MJD_dfb = []# Loop through each file in the second folder in ascending orderfor filename_dfb in sorted(os.listdir(folder_path_dfb)):    if filename_dfb.endswith('.txt'):        file_order_dfb.append(filename_dfb)        file_path_dfb = os.path.join(folder_path_dfb, filename_dfb)        df_dfb = pd.read_csv(file_path_dfb, header=None, delimiter='\t')        values_dfb = df_dfb.values.flatten()        all_dataframes_dfb.append(values_dfb)        # get MJD from the filenames        extracted_chars_dfb = filename_dfb.replace('B1822-09_', '').replace('.txt', '')        num_value_dfb = float(extracted_chars_dfb)        num_list_dfb = num_value_dfb        MJD_dfb.append(round(num_list_dfb))# Read in spin down data for the second foldercolumns_dfb = ['MJD', 'Spin-down rate (1e-15 Hz/s)', 'Spin-down rate uncertainty (1e-15 Hz/s)']spin_down_df_dfb = pd.read_csv(spin_down_data_dfb, delim_whitespace=True, names=columns_afb)# Create DataFrame for the second folderraw_df_dfb = pd.DataFrame(all_dataframes_dfb)#drop rows#rows_to_drop = [6, 64, 137, 147, 227, 298, 432, 433, 440, 442, 446,448, 449, 452, 453, 460,461, 468, 504, 606, 622, 691, 720, 722, 818, 821, 836, 839, 846]#raw_df_dfb = raw_df_dfb.drop(rows_to_drop)#MJD_dfb = MJD_dfb.drop(rows_to_drop)def closest_index(arr, value):    closest_index = 0    min_diff = abs(arr[0] - value)    for i in range(1, len(arr)):        diff = abs(arr[i] - value)        if diff < min_diff:            min_diff = diff            closest_index = i    return closest_indexdef plotstackedintensities(matrix):    #Display the stacked pca reconstruction    n_components = 2    plt.figure(figsize=(10, 6))    plt.imshow(((matrix)), cmap='inferno', aspect='auto')    plt.colorbar()    plt.title(f'Horizontally Stacked Intensity Curves reconstruction with {n_components} PCs')    plt.xlabel('Phase bin')    plt.ylabel('Observation Number')    plt.show()    returndef gaussian_func(x, mu, sigma, A):    return A*norm.pdf(x, loc=mu, scale=sigma)def normbygaussian_afb(dataframe):    raw_df = dataframe    peak_subset_df = dataframe.iloc[:, np.r_[94:120]]    bin_averages = np.mean(peak_subset_df, axis=0)                #get profile shape    scaler = []    ideal_params, covariance = curve_fit(gaussian_func, np.arange(len(bin_averages)), bin_averages)    mu_fit = ideal_params[0]    sigma_fit = ideal_params[1]    A_fit = ideal_params[2]    for i in range(0,len(raw_df.iloc[:,1])):        ideal_params_obs, covarianceobs = curve_fit(gaussian_func, np.arange(len(peak_subset_df.iloc[i,:])), peak_subset_df.iloc[i,:], maxfev = 5000)        A_fit_obs = ideal_params_obs[2]        scaling = A_fit / A_fit_obs        scaler.append(scaling)    normalised_by_fit_df = raw_df.mul(scaler, axis=0)    return(normalised_by_fit_df)def normbygaussian_dfb(dataframe):    raw_df = dataframe    #peak_subset_df = dataframe.iloc[:, np.r_[94:120]] #FOR AFB!!!!    peak_subset_df = dataframe.iloc[:, np.r_[240:290]]    bin_averages = np.mean(peak_subset_df, axis=0)                #get profile shape    scaler = []    ideal_params, covariance = curve_fit(gaussian_func, np.arange(len(bin_averages)), bin_averages)    mu_fit = ideal_params[0]    sigma_fit = ideal_params[1]    A_fit = ideal_params[2]    def new_gaussian_func(x, A):        return A*norm.pdf(x, loc=mu_fit, scale=sigma_fit)    for i in range(0,len(raw_df.iloc[:,1])):        ideal_params_obs, covarianceobs = curve_fit(new_gaussian_func, np.arange(len(peak_subset_df.iloc[i,:])), peak_subset_df.iloc[i,:], maxfev = 5000)        A_fit_obs = ideal_params_obs[0]        scaling = A_fit / A_fit_obs        scaler.append(scaling)    normalised_by_fit_df = raw_df.mul(scaler, axis=0)    return(normalised_by_fit_df)def dopca(dataset,n_components, subset_range):    '''use:    pcaresult = dopca(dfaveragenorm,2)    plotPCsandshape4(pcaresult[1].iloc[0,:], pcaresult[1].iloc[1,:], pcaresult[0],pcaresult[2],days)     where: plotPCsandshape4(pc1, pc2, EIGENVALUES, matrix, days)    '''    dataset = pd.DataFrame(dataset)    if subset_range is not None and subset_range.size > 0:        dataset = dataset.iloc[:, subset_range]    else:        dataset = dataset    pca = PCA(n_components=n_components) #defines the pca action    principalComponents = pca.fit_transform(dataset)    evals = pd.DataFrame(data=principalComponents)    evecs = pd.DataFrame(data=pca.components_)    varianceratio = pca.explained_variance_ratio_    mean = pca.mean_    reconstructeddf = mean + evals @ evecs    return(evals,evecs,reconstructeddf,varianceratio)def smooth_sp_and_days(days, shape_parameters, window_size):    days = np.array(days)    num_days = int(days[len(days)-1] - days[0]) +1    overlap = window_size // 4 #step size    num_windows = (num_days - window_size) // overlap + 1    mean_shape_parameters = []    mean_day_values = []    for i in range(num_windows):        start_day = days[0] + i * overlap        end_day = start_day + window_size        window_indices = [j for j, day in enumerate(days) if start_day <= day < end_day]        window_shape_parameters = [shape_parameters[j] for j in window_indices]        if window_shape_parameters:            mean_shape_parameter = np.mean(window_shape_parameters)            mean_shape_parameters.append(mean_shape_parameter)            mean_day_value = start_day + (window_size / 2)            mean_day_values.append(mean_day_value)    return(mean_day_values,mean_shape_parameters)def get_evenly_spaced_days(mjd):    mjd = np.array(mjd)    initial = mjd[0]    final = mjd[len(mjd)-1]    #num_values = len(mjd)    num_values = final - initial    #evenly_spaced_mjd = np.linspace(initial, final, num=num_values)    evenly_spaced_mjd = np.round(np.linspace(initial, final, num_values)).astype(int)    print(f'even space = {evenly_spaced_mjd}')    new_mjd = []    for entry in evenly_spaced_mjd:        given_value = entry        absolute_diff = np.abs(given_value -  mjd)        closest_index = np.argmin(absolute_diff)        closest_day = mjd[closest_index]        #print(f'closest_day= {closest_day}')        new_mjd.append(closest_day)    return(np.array(new_mjd).flatten())def createeven_df_andtranspose(days, evendays, normaliseddf):    normaliseddf = pd.DataFrame(normaliseddf)    new_matrix = []    for entry in evendays:        #find row with nearest        given_value = entry        absolute_diff = np.abs(days - given_value)        index = np.argmin(absolute_diff) #finds the row index with the day nearest to that of even day        row_to_append = normaliseddf.iloc[index]        row_to_append = np.array(row_to_append)        new_matrix.append(row_to_append) #this is the normalised df in the even spacing but not transposed    transposed_matrix = [[row[i] for row in new_matrix] for i in range(len(new_matrix[0]))]    transposed_matrix = np.array(transposed_matrix)    return(np.array(transposed_matrix))def plot_pcs(dataset, n, subset):    eigenvalues, eigenvectors, matrix, variance = dopca(dataset, n, subset)    print(variance)    if n == 1:        plt.figure(figsize=(10, 6))        plt.plot(np.arange(len(eigenvectors.iloc[0,:])),eigenvectors.iloc[0,:])        plt.text(0,0,f'exp var = {variance}')        plt.xlabel('phase')        plt.ylabel(f'PC{n}')        plt.title('Principal components')    else:        fig, axes = plt.subplots(n, 1, figsize=(10, 6))        for i in range(n):            axes[i].plot(np.arange(len(eigenvectors.iloc[i,:])),eigenvectors.iloc[i,:])            axes[i].set_xlabel('phase')            axes[i].set_ylabel(f'PC{i}')        fig.suptitle('Principal components')        plt.subplots_adjust(hspace=0)        plt.tight_layout()  # Adjust layout to prevent overlap        plt.show()window = 100num_comp = 1SNr_cutoff = 15#need to normalise, get pca reconstructions and then plot against spin down ratenormalised_df_afb0 = normbygaussian_afb(raw_df_afb)subset_range_afb = np.concatenate((np.arange(60, 110), np.arange(290, 320)))normalised_df_afb = normalised_df_afb0.iloc[:, subset_range_afb]plotstackedintensities(normalised_df_afb)# days array is called MJD_afbnormalised_df_dfb0 = normbygaussian_dfb(raw_df_dfb)rows_to_drop = []SNr_arr = []for i in range(len(normalised_df_dfb0.iloc[:,1])):    obs = normalised_df_dfb0.iloc[i,:]    noise_reigon = obs[400:600] #get off pulse region    noise = np.std(noise_reigon)    signal = np.max(obs) #calc peak value of pulse    SNr = signal / noise    SNr_arr.append(SNr)    if SNr < SNr_cutoff:        rows_to_drop.append(i)normalised_df_dfb = normalised_df_dfb0.drop(rows_to_drop)subset_range_dfb = np.concatenate((np.arange(160, 291), np.arange(750, 801)))subset_range_dfb1 = (np.arange(750, 800))normalised_df_dfb = normalised_df_dfb.iloc[:, subset_range_dfb]#plotstackedintensities(normalised_df_dfb)#assign values:94:120MJD_dfb = pd.DataFrame(MJD_dfb)MJD_dfb = MJD_dfb.drop(rows_to_drop)MJD_dfb = np.array(MJD_dfb)from itertools import chainMJD_dfb = list(chain.from_iterable(MJD_dfb))all_shape_parameter_afb = dopca(normalised_df_afb, num_comp, None)[0] #perform pca on afbshape_parameter_afb = all_shape_parameter_afb.iloc[:,0]#plotstackedintensities(dopca(normalised_df_afb,2)[2]) #plot stacked intensities to check pca working as expectedall_shape_parameter_dfb = dopca(normalised_df_dfb, num_comp, None)[0] #perform pca on dfbshape_parameter_dfb = all_shape_parameter_dfb.iloc[:,0]#plotstackedintensities(dopca(normalised_df_dfb, num_comp)[2]) #plot stacked intensities to check pca working as expectedspindown_MJD_afb = spin_down_df_afb.iloc[:,0]spindown_rate_afb = spin_down_df_afb.iloc[:,1]spindown_MJD_dfb = spin_down_df_dfb.iloc[:,0]spindown_rate_dfb = spin_down_df_dfb.iloc[:,1]#assign smoothed valuessmoothed_spindown_MJD_afb, smoothed_spindown_rate_afb = smooth_sp_and_days(spindown_MJD_afb,spindown_rate_afb, window)smoothed_MJD_afb, smoothed_shape_parameter_afb = smooth_sp_and_days(MJD_afb,shape_parameter_afb, window)smoothed_spindown_MJD_dfb, smoothed_spindown_rate_dfb = smooth_sp_and_days(spindown_MJD_dfb,spindown_rate_dfb, window)smoothed_MJD_dfb, smoothed_shape_parameter_dfb = smooth_sp_and_days(MJD_dfb,shape_parameter_dfb, window)evenly_spaced_mjd_dfb = get_evenly_spaced_days(MJD_dfb)transposed_matrix_dfb = createeven_df_andtranspose(MJD_dfb, evenly_spaced_mjd_dfb, normalised_df_dfb)evenly_spaced_mjd_afb = get_evenly_spaced_days(MJD_afb)transposed_matrix_afb = createeven_df_andtranspose(MJD_afb, evenly_spaced_mjd_afb, normalised_df_afb)#make horizontal version of the uneven day dataset to checknormalised_df_afb = np.array(normalised_df_afb)transposed_matrix_afb_1 = [[row[i] for row in normalised_df_afb] for i in range(len(normalised_df_afb[0]))]transposed_matrix_afb_1 = np.array(transposed_matrix_afb_1)normalised_df_dfb = np.array(normalised_df_dfb)transposed_matrix_dfb_1 = [[row[i] for row in normalised_df_dfb] for i in range(len(normalised_df_dfb[0]))]transposed_matrix_dfb_1 = np.array(transposed_matrix_dfb_1)n_comp = 2PCA_afb = dopca(transposed_matrix_afb,n_comp, None)[2]PCA_dfb = dopca(transposed_matrix_dfb,n_comp, None)[2]#DFB!fig, axes = plt.subplots(5, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [1,0.1,1, 0.5, 0.5]})# Plot the transposed matrix on the first subplotaxes[0].imshow(transposed_matrix_dfb, cmap='inferno', aspect='auto')axes[0].set_xlabel('MJD')axes[0].set_ylabel('normalised intensities')axes[0].grid(False)axes[0].xaxis.set_visible(False)axes[1].plot()axes[1].set_xlim(evenly_spaced_mjd_dfb[0], evenly_spaced_mjd_dfb[-1])for x in np.array(MJD_dfb):    axes[1].axvline(x)axes[1].xaxis.set_visible(False)# Plot the PCA 2nd sublotaxes[2].imshow(PCA_dfb, cmap='inferno', aspect='auto')axes[2].set_xlabel('MJD')axes[2].set_ylabel(f'PCA w/ {n_comp} components')axes[2].grid(False)axes[2].xaxis.set_visible(False)# Plot the spindown dfb on the second subplotaxes[3].plot(smoothed_MJD_dfb, smoothed_shape_parameter_dfb, color='red')axes[3].set_ylabel(f'shape param (window= {window})')axes[3].set_xlim(evenly_spaced_mjd_dfb[0], evenly_spaced_mjd_dfb[-1])axes[3].xaxis.set_visible(False)# Plot the third panelaxes[4].plot(spindown_MJD_dfb, spindown_rate_dfb, color='blue')  # Replace x_data and y_data with your dataaxes[4].set_xlabel('Modified Julian Days')  # Replace with appropriate labelaxes[4].set_ylabel('Spindown')  # Replace with appropriate labelaxes[4].set_xlim(evenly_spaced_mjd_dfb[0], evenly_spaced_mjd_dfb[-1])fig.suptitle('dfb emission modes shown by colourmap, shape parameter and spindown rate against days', fontsize=16)plt.subplots_adjust(hspace=0)plt.tight_layout()  # Adjust layout to prevent overlapplt.show()#AFB!fig, axes = plt.subplots(5, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [1,0.1,1, 0.5, 0.5]})# Plot the transposed matrix on the first subplotaxes[0].imshow(transposed_matrix_afb, cmap='inferno', aspect='auto')axes[0].set_xlabel('MJD')axes[0].set_ylabel('normalised intensities')axes[0].grid(False)axes[0].xaxis.set_visible(False)axes[1].plot()axes[1].set_xlim(evenly_spaced_mjd_afb[0], evenly_spaced_mjd_afb[-1])for x in np.array(MJD_afb):    axes[1].axvline(x)axes[1].xaxis.set_visible(False)axes[1].yaxis.set_visible(False)# Plot the PCA 2nd sublotaxes[2].imshow(PCA_afb, cmap='inferno', aspect='auto')axes[2].set_xlabel('MJD')axes[2].set_ylabel(f'PCA w/ {n_comp} components')axes[2].grid(False)axes[2].xaxis.set_visible(False)# Plot the spindown dfb on the second subplotaxes[3].plot(smoothed_MJD_afb, smoothed_shape_parameter_afb, color='red')axes[3].set_ylabel(f'shape param (window= {window})')axes[3].set_xlim(evenly_spaced_mjd_afb[0], evenly_spaced_mjd_afb[-1])axes[3].xaxis.set_visible(False)# Plot the third panelaxes[4].plot(spindown_MJD_afb, spindown_rate_afb, color='blue')  # Replace x_data and y_data with your dataaxes[4].set_xlabel('Modified Julian Days')  # Replace with appropriate labelaxes[4].set_ylabel('Spindown')  # Replace with appropriate labelaxes[4].set_xlim(evenly_spaced_mjd_afb[0], evenly_spaced_mjd_afb[-1])fig.suptitle('AFB emission modes shown by colourmap, shape parameter and spindown rate against days', fontsize=16)plt.subplots_adjust(hspace=0)plt.tight_layout()  # Adjust layout to prevent overlapplt.show()#AFB !'''fig, axes = plt.subplots(4, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [1,1, 0.5, 0.5]})# Plot the transposed matrix on the first subplotaxes[0].imshow(transposed_matrix_afb, cmap='inferno', aspect='auto')axes[0].set_xlabel('MJD')axes[0].set_ylabel('normalised')axes[0].grid(False)axes[0].xaxis.set_visible(False)# Plot the PCA 2nd sublotaxes[1].imshow(PCA_afb, cmap='inferno', aspect='auto')axes[1].set_xlabel('MJD')axes[1].set_ylabel(f'PCA  {n_comp} components')axes[1].grid(False)axes[1].xaxis.set_visible(False)# Plot the spindown dfb on the second subplotaxes[2].plot(smoothed_MJD_afb, smoothed_shape_parameter_afb, color='red')axes[2].set_ylabel(f'shape param (window= {window})')axes[2].set_xlim(evenly_spaced_mjd_afb[0], evenly_spaced_mjd_afb[-1])axes[2].xaxis.set_visible(False)# Plot the third panelaxes[3].plot(spindown_MJD_afb, spindown_rate_afb, color='blue')  # Replace x_data and y_data with your dataaxes[3].set_xlabel('Modified Julian Days')  # Replace with appropriate labelaxes[3].set_ylabel('Spindown')  # Replace with appropriate labelaxes[3].set_xlim(evenly_spaced_mjd_afb[0], evenly_spaced_mjd_afb[-1])fig.suptitle('afb emission modes shown by colourmap, shape parameter and spindown rate against days', fontsize=16)plt.subplots_adjust(hspace=0)plt.tight_layout()  # Adjust layout to prevent overlapplt.show()'''