#all graphs

#new 4 pcs and variation
fig, axes = plt.subplots(num_comp, 2, figsize=(9, 8), gridspec_kw={'width_ratios': [1,0.15]})
for i in range(num_comp):

    single_eval_new = pd.DataFrame(eigenvalues_new.iloc[:, i])
    single_evec_new = pd.DataFrame(eigenvectors_new.iloc[i ,:]).T
    single_recon_new = single_eval_new @ single_evec_new
    even_single_recon_new = createeven_df_andtranspose(MJD_new,evenly_spaced_mjd_new,single_recon_new)
    print(np.shape(even_single_recon_new))
    extent = [evenly_spaced_mjd_new[0], evenly_spaced_mjd_new[-1], 68, 0]

    axes[i, 0].imshow(even_single_recon_new, cmap='inferno', aspect='auto',extent=extent)
    axes[i, 0].set_ylabel(f' PC{i+1} phase')
    axes[i, 0].grid(False)
    axes[i, 0].xaxis.set_visible(True)

    axes[i, 1].invert_yaxis()
    axes[i, 1].yaxis.set_visible(False)
    axes[i, 1].plot(eigenvectors_new.iloc[i,:],np.arange(len(eigenvectors_new.iloc[i,:])))
    axes[i, 1].grid(False)
    axes[i, 1].xaxis.set_visible(False)

#axes[i, 0].xaxis.set_visible(True)
axes[i, 0].set_xlabel('Modified Julian Day')
axes[i, 1].xaxis.set_visible(True)
axes[i, 1].set_xlabel('Intensity')
fig.suptitle('')
#im = axes[0, 0].imshow(transposed_matrix_new, cmap='inferno', aspect='auto')
cbar_ax = fig.add_axes([0.05, 1, 0.3, 0.015])
fig.colorbar(im,cax=cbar_ax, orientation='horizontal')
plt.subplots_adjust(top=0.9,hspace=0)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

#afb 4 pcs and variation
fig, axes = plt.subplots(num_comp, 2, figsize=(9, 8), gridspec_kw={'width_ratios': [1,0.15]})
for i in range(num_comp):

    single_eval_afb = pd.DataFrame(eigenvalues_afb.iloc[:, i])
    single_evec_afb = pd.DataFrame(eigenvectors_afb.iloc[i ,:]).T
    single_recon_afb = single_eval_afb @ single_evec_afb
    even_single_recon_afb = createeven_df_andtranspose(MJD_afb,evenly_spaced_mjd_afb,single_recon_afb)
    print(np.shape(even_single_recon_afb))
    extent = [evenly_spaced_mjd_afb[0], evenly_spaced_mjd_afb[-1], 68, 0]

    axes[i, 0].imshow(even_single_recon_afb, cmap='inferno', aspect='auto',extent=extent)
    axes[i, 0].set_ylabel(f' PC{i+1} phase')
    axes[i, 0].grid(False)
    axes[i, 0].xaxis.set_visible(True)

    axes[i, 1].invert_yaxis()
    axes[i, 1].yaxis.set_visible(False)
    axes[i, 1].plot(eigenvectors_afb.iloc[i,:],np.arange(len(eigenvectors_afb.iloc[i,:])))
    axes[i, 1].grid(False)
    axes[i, 1].xaxis.set_visible(False)

#axes[i, 0].xaxis.set_visible(True)
axes[i, 0].set_xlabel('Modified Julian Day')
#axes[i, 1].set_xticks([])
axes[i, 1].xaxis.set_visible(True)
axes[i, 1].set_xlabel('Intensity')
fig.suptitle('')
#im = axes[0, 0].imshow(transposed_matrix_afb, cmap='inferno', aspect='auto')
cbar_ax = fig.add_axes([0.05, 1, 0.3, 0.015])
fig.colorbar(im,cax=cbar_ax, orientation='horizontal')
plt.subplots_adjust(top=0.9,hspace=0)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

#dfb 4 pcs and variation
fig, axes = plt.subplots(num_comp, 2, figsize=(9, 8), gridspec_kw={'width_ratios': [1,0.15]})
for i in range(num_comp):

    single_eval_dfb = pd.DataFrame(eigenvalues_dfb.iloc[:, i])
    single_evec_dfb = pd.DataFrame(eigenvectors_dfb.iloc[i ,:]).T
    single_recon_dfb = single_eval_dfb @ single_evec_dfb
    even_single_recon_dfb = createeven_df_andtranspose(MJD_dfb,evenly_spaced_mjd_dfb,single_recon_dfb)
    print(np.shape(even_single_recon_dfb))
    extent = [evenly_spaced_mjd_dfb[0], evenly_spaced_mjd_dfb[-1], 68, 0]

    axes[i, 0].imshow(even_single_recon_dfb, cmap='inferno', aspect='auto',extent=extent)
    axes[i, 0].set_ylabel(f' PC{i+1} phase')
    axes[i, 0].grid(False)
    axes[i, 0].xaxis.set_visible(True)


    axes[i, 1].invert_yaxis()
    axes[i, 1].yaxis.set_visible(False)
    axes[i, 1].plot(eigenvectors_dfb.iloc[i,:],np.arange(len(eigenvectors_dfb.iloc[i,:])))
    axes[i, 1].grid(False)
    axes[i, 1].xaxis.set_visible(False)

axes[0, 1].invert_xaxis()
#axes[i, 0].xaxis.set_visible(True)
axes[i, 0].set_xlabel('Modified Julian Day')
#axes[i, 1].set_xticks([])
axes[i, 1].xaxis.set_visible(True)
axes[i, 1].set_xlabel('Intensity')
fig.suptitle('')
#im = axes[0, 0].imshow(transposed_matrix_afb, cmap='inferno', aspect='auto')
cbar_ax = fig.add_axes([0.05, 1, 0.3, 0.015])
fig.colorbar(im,cax=cbar_ax, orientation='horizontal')
plt.subplots_adjust(top=0.9,hspace=0)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

#smoothing

fig, axes = plt.subplots(3, 1, figsize=(6, 4), gridspec_kw={'height_ratios': [0.1, 1, 1]})
axes[0].plot()
axes[0].set_xlim(evenly_spaced_mjd_new[0], evenly_spaced_mjd_new[-1])
for x in np.array(MJD_new):
    axes[0].axvline(x)
axes[0].xaxis.set_visible(False)
axes[0].yaxis.set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)
axes[0].spines['left'].set_visible(False)

axes[1].plot(MJD_new, shape_parameter_new, color='black')
axes[1].set_ylabel(f'SP')
axes[1].set_xlim(evenly_spaced_mjd_new[0], evenly_spaced_mjd_new[-1])
axes[1].xaxis.set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)

axes[2].plot(smoothed_MJD_new, smoothed_shape_parameter_new, color='black')
axes[2].set_ylabel(f'smoothed SP ')
axes[2].set_xlim(evenly_spaced_mjd_new[0], evenly_spaced_mjd_new[-1])
axes[2].set_xlabel(f'Modified Julian Day')  # Add x-axis label
axes[2].xaxis.set_visible(True)
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)

fig.suptitle(f'')
plt.subplots_adjust(hspace=0)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

#SPS 1828

shape_parameter_new = eigenvalues_new.iloc[:,0]
shape_parameter_new1 = eigenvalues_new.iloc[:,1]
shape_parameter_new2 = eigenvalues_new.iloc[:,2]
shape_parameter_new3 = eigenvalues_new.iloc[:,3]
smoothed_shape_parameter_new = smooth_sp_and_days(MJD_new,shape_parameter_new, window)[1]
smoothed_shape_parameter_new1 = smooth_sp_and_days(MJD_new,shape_parameter_new1, window)[1]
smoothed_shape_parameter_new2 = smooth_sp_and_days(MJD_new,shape_parameter_new2, window)[1]
smoothed_shape_parameter_new3 = smooth_sp_and_days(MJD_new,shape_parameter_new3, window)[1]

fig, axes = plt.subplots(5, 1, figsize=(6, 4), gridspec_kw={'height_ratios': [0.5, 0.5, 0.5, 0.5, 0.5]})
axes[0].plot(smoothed_MJD_new, smoothed_shape_parameter_new, color='black')
axes[0].set_ylabel(f'SP 1')
axes[0].set_xlim(evenly_spaced_mjd_new[0], evenly_spaced_mjd_new[-1])
axes[0].xaxis.set_visible(False)
for spine in axes[0].spines.values():
    if spine == axes[0].spines['top'] or spine == axes[0].spines['right'] or spine == axes[0].spines['bottom']:
        spine.set_visible(False)
axes[0].tick_params(axis='both', which='both', bottom=False, top=False, left=True, right=False)

axes[1].plot(smoothed_MJD_new, smoothed_shape_parameter_new1, color='black')
axes[1].set_ylabel(f'SP 2')
axes[1].set_xlim(evenly_spaced_mjd_new[0], evenly_spaced_mjd_new[-1])
axes[1].xaxis.set_visible(False)
for spine in axes[1].spines.values():
    if spine == axes[1].spines['top'] or spine == axes[1].spines['right'] or spine == axes[1].spines['bottom']:
        spine.set_visible(False)
axes[1].tick_params(axis='both', which='both', bottom=False, top=False, left=True, right=False)

axes[2].plot(smoothed_MJD_new, smoothed_shape_parameter_new2, color='black')
axes[2].set_ylabel(f'SP 3')
axes[2].set_xlim(evenly_spaced_mjd_new[0], evenly_spaced_mjd_new[-1])
axes[2].xaxis.set_visible(False)
for spine in axes[2].spines.values():
    if spine == axes[2].spines['top'] or spine == axes[2].spines['right'] or spine == axes[2].spines['bottom']:
        spine.set_visible(False)
axes[2].tick_params(axis='both', which='both', bottom=False, top=False, left=True, right=False)

axes[3].plot(smoothed_MJD_new, smoothed_shape_parameter_new3, color='black')
axes[3].set_ylabel(f'SP 4')
axes[3].set_xlim(evenly_spaced_mjd_new[0], evenly_spaced_mjd_new[-1])
axes[3].xaxis.set_visible(False)
for spine in axes[3].spines.values():
    if spine == axes[3].spines['top'] or spine == axes[3].spines['right'] or spine == axes[3].spines['bottom']:
        spine.set_visible(False)
axes[3].tick_params(axis='both', which='both', bottom=False, top=False, left=True, right=False)

axes[4].plot(spindown_MJD_new, spindown_rate_new, color='red')  # Replace x_data and y_data with your data
axes[4].set_xlabel('Modified Julian Days')  # Replace with appropriate label
axes[4].set_ylabel(r'$\dot{\nu}$')  # Replace with appropriate label
axes[4].set_xlim(evenly_spaced_mjd_new[0], evenly_spaced_mjd_new[-1])
for spine in axes[4].spines.values():
    if spine == axes[4].spines['top'] or spine == axes[4].spines['right']:
        spine.set_visible(False)
axes[4].tick_params(axis='both', which='both', bottom=True, top=False, left=True, right=False)

fig.suptitle(f'', fontsize=16)
plt.subplots_adjust(hspace=0)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

#sp for afb

shape_parameter_afb = eigenvalues_afb.iloc[:,0]
shape_parameter_afb1 = eigenvalues_afb.iloc[:,1]
shape_parameter_afb2 = eigenvalues_afb.iloc[:,2]
shape_parameter_afb3 = eigenvalues_afb.iloc[:,3]
smoothed_shape_parameter_afb = smooth_sp_and_days(MJD_afb, shape_parameter_afb, window)[1]
smoothed_shape_parameter_afb1 = smooth_sp_and_days(MJD_afb, shape_parameter_afb1, window)[1]
smoothed_shape_parameter_afb2 = smooth_sp_and_days(MJD_afb, shape_parameter_afb2, window)[1]
smoothed_shape_parameter_afb3 = smooth_sp_and_days(MJD_afb, shape_parameter_afb3, window)[1]
fig, axes = plt.subplots(6, 1, figsize=(6, 4), gridspec_kw={'height_ratios': [0.15,0.5, 0.5, 0.5, 0.5, 0.5]})
axes[0].plot()
axes[0].set_xlim(evenly_spaced_mjd_afb[0], evenly_spaced_mjd_afb[-1])
for x in np.array(MJD_afb):
    axes[0].axvline(x)
axes[0].xaxis.set_visible(False)
axes[0].yaxis.set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)
axes[0].spines['left'].set_visible(False)

axes[1].plot(smoothed_MJD_afb, smoothed_shape_parameter_afb, color='black')
axes[1].set_ylabel(f'SP 1')
axes[1].set_xlim(evenly_spaced_mjd_afb[0], evenly_spaced_mjd_afb[-1])
axes[1].xaxis.set_visible(False)
for spine in axes[1].spines.values():
    if spine == axes[1].spines['top'] or spine == axes[1].spines['right'] or spine == axes[1].spines['bottom']:
        spine.set_visible(False)
axes[1].tick_params(axis='both', which='both', bottom=False, top=False, left=True, right=False)

axes[2].plot(smoothed_MJD_afb, smoothed_shape_parameter_afb1, color='black')
axes[2].set_ylabel(f'SP 2')
axes[2].set_xlim(evenly_spaced_mjd_afb[0], evenly_spaced_mjd_afb[-1])
axes[2].xaxis.set_visible(False)
for spine in axes[2].spines.values():
    if spine == axes[2].spines['top'] or spine == axes[2].spines['right'] or spine == axes[2].spines['bottom']:
        spine.set_visible(False)
axes[2].tick_params(axis='both', which='both', bottom=False, top=False, left=True, right=False)

axes[3].plot(smoothed_MJD_afb, smoothed_shape_parameter_afb2, color='black')
axes[3].set_ylabel(f'SP 3')
axes[3].set_xlim(evenly_spaced_mjd_afb[0], evenly_spaced_mjd_afb[-1])
axes[3].xaxis.set_visible(False)
for spine in axes[3].spines.values():
    if spine == axes[3].spines['top'] or spine == axes[3].spines['right'] or spine == axes[3].spines['bottom']:
        spine.set_visible(False)
axes[3].tick_params(axis='both', which='both', bottom=False, top=False, left=True, right=False)

axes[4].plot(smoothed_MJD_afb, smoothed_shape_parameter_afb3, color='black')
axes[4].set_ylabel(f'SP 4')
axes[4].set_xlim(evenly_spaced_mjd_afb[0], evenly_spaced_mjd_afb[-1])
axes[4].xaxis.set_visible(False)
for spine in axes[4].spines.values():
    if spine == axes[4].spines['top'] or spine == axes[4].spines['right'] or spine == axes[4].spines['bottom']:
        spine.set_visible(False)
axes[4].tick_params(axis='both', which='both', bottom=False, top=False, left=True, right=False)

axes[5].plot(spindown_MJD_afb, spindown_rate_afb, color='red')  # Replace x_data and y_data with your data
axes[5].set_xlabel('Modified Julian Days')  # Replace with appropriate label
axes[5].set_ylabel(r'$\dot{\nu}$')  # Replace with appropriate label
axes[5].set_xlim(evenly_spaced_mjd_afb[0], evenly_spaced_mjd_afb[-1])
for spine in axes[5].spines.values():
    if spine == axes[5].spines['top'] or spine == axes[5].spines['right']:
        spine.set_visible(False)
axes[5].tick_params(axis='both', which='both', bottom=True, top=False, left=True, right=False)

vertical_mjd_values = [49954.7, 51107.864779, 51970.496756, 52813.1993571, 53751.6380477]
for mjd_value in vertical_mjd_values:  # Loop starts from the 4th element
    for ax in axes[1:]:  # Loop through the axes starting from the 4th
        ax.axvline(x=mjd_value, color='blue', linestyle='--')

fig.suptitle(f'', fontsize=16)
plt.subplots_adjust(hspace=0)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

#SPs dfb
shape_parameter_dfb = eigenvalues_dfb.iloc[:,0]
shape_parameter_dfb1 = eigenvalues_dfb.iloc[:,1]
shape_parameter_dfb2 = eigenvalues_dfb.iloc[:,2]
shape_parameter_dfb3 = eigenvalues_dfb.iloc[:,3]
smoothed_shape_parameter_dfb = smooth_sp_and_days(MJD_dfb, shape_parameter_dfb, window)[1]
smoothed_shape_parameter_dfb1 = smooth_sp_and_days(MJD_dfb, shape_parameter_dfb1, window)[1]
smoothed_shape_parameter_dfb2 = smooth_sp_and_days(MJD_dfb, shape_parameter_dfb2, window)[1]
smoothed_shape_parameter_dfb3 = smooth_sp_and_days(MJD_dfb, shape_parameter_dfb3, window)[1]

fig, axes = plt.subplots(6, 1, figsize=(6, 4), gridspec_kw={'height_ratios': [0.15,0.5, 0.5, 0.5, 0.5, 0.5]})
axes[0].plot()
axes[0].set_xlim(evenly_spaced_mjd_dfb[0], evenly_spaced_mjd_dfb[-1])
for x in np.array(MJD_dfb):
    axes[0].axvline(x)
axes[0].xaxis.set_visible(False)
axes[0].yaxis.set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)
axes[0].spines['left'].set_visible(False)

axes[1].plot(smoothed_MJD_dfb, smoothed_shape_parameter_dfb, color='black')
axes[1].set_ylabel(f'SP 1')
axes[1].set_xlim(evenly_spaced_mjd_dfb[0], evenly_spaced_mjd_dfb[-1])
axes[1].xaxis.set_visible(False)
for spine in axes[1].spines.values():
    if spine == axes[1].spines['top'] or spine == axes[1].spines['right'] or spine == axes[1].spines['bottom']:
        spine.set_visible(False)
axes[1].tick_params(axis='both', which='both', bottom=False, top=False, left=True, right=False)

axes[2].plot(smoothed_MJD_dfb, smoothed_shape_parameter_dfb1, color='black')
axes[2].set_ylabel(f'SP 2')
axes[2].set_xlim(evenly_spaced_mjd_dfb[0], evenly_spaced_mjd_dfb[-1])
axes[2].xaxis.set_visible(False)
for spine in axes[2].spines.values():
    if spine == axes[2].spines['top'] or spine == axes[2].spines['right'] or spine == axes[2].spines['bottom']:
        spine.set_visible(False)
axes[2].tick_params(axis='both', which='both', bottom=False, top=False, left=True, right=False)

axes[3].plot(smoothed_MJD_dfb, smoothed_shape_parameter_dfb2, color='black')
axes[3].set_ylabel(f'SP 3')
axes[3].set_xlim(evenly_spaced_mjd_dfb[0], evenly_spaced_mjd_dfb[-1])
axes[3].xaxis.set_visible(False)
for spine in axes[3].spines.values():
    if spine == axes[3].spines['top'] or spine == axes[3].spines['right'] or spine == axes[3].spines['bottom']:
        spine.set_visible(False)
axes[3].tick_params(axis='both', which='both', bottom=False, top=False, left=True, right=False)

axes[4].plot(smoothed_MJD_dfb, smoothed_shape_parameter_dfb3, color='black')
axes[4].set_ylabel(f'SP 4')
axes[4].set_xlim(evenly_spaced_mjd_dfb[0], evenly_spaced_mjd_dfb[-1])
axes[4].xaxis.set_visible(False)
for spine in axes[4].spines.values():
    if spine == axes[4].spines['top'] or spine == axes[4].spines['right'] or spine == axes[4].spines['bottom']:
        spine.set_visible(False)
axes[4].tick_params(axis='both', which='both', bottom=False, top=False, left=True, right=False)

axes[5].plot(spindown_MJD_dfb, spindown_rate_dfb, color='red')  # Replace x_data and y_data with your data
axes[5].set_xlabel('Modified Julian Days')  # Replace with appropriate label
axes[5].set_ylabel(r'$\dot{\nu}$')  # Replace with appropriate label
axes[5].set_xlim(evenly_spaced_mjd_dfb[0], evenly_spaced_mjd_dfb[-1])
for spine in axes[5].spines.values():
    if spine == axes[5].spines['top'] or spine == axes[5].spines['right']:
        spine.set_visible(False)
axes[5].tick_params(axis='both', which='both', bottom=True, top=False, left=True, right=False)

fig.suptitle(f'', fontsize=16)
plt.subplots_adjust(hspace=0)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


##code to compare specific PCs and SPs
shape_parameter_dfb = eigenvalues_dfb.iloc[:,0]
shape_parameter_dfb1 = eigenvalues_dfb.iloc[:,1]
shape_parameter_dfb2 = eigenvalues_dfb.iloc[:,2]
shape_parameter_dfb3 = eigenvalues_dfb.iloc[:,3]
smoothed_shape_parameter_dfb = smooth_sp_and_days(MJD_dfb, shape_parameter_dfb, window)[1]
smoothed_shape_parameter_dfb1 = smooth_sp_and_days(MJD_dfb, shape_parameter_dfb1, window)[1]
smoothed_shape_parameter_dfb2 = smooth_sp_and_days(MJD_dfb, shape_parameter_dfb2, window)[1]
smoothed_shape_parameter_dfb3 = smooth_sp_and_days(MJD_dfb, shape_parameter_dfb3, window)[1]

i=1
single_eval_dfb = pd.DataFrame(eigenvalues_dfb.iloc[:, i])
single_evec_dfb = pd.DataFrame(eigenvectors_dfb.iloc[i ,:]).T
single_recon_dfb = single_eval_dfb @ single_evec_dfb
even_single_recon_dfb = createeven_df_andtranspose(MJD_dfb,evenly_spaced_mjd_dfb,single_recon_dfb)

fig, axes = plt.subplots(4, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1,1,0.5,0.5]})
extent = [evenly_spaced_mjd_dfb[0], evenly_spaced_mjd_dfb[-1], len(eigenvectors_dfb.iloc[0,:]), 0]
axes[0].imshow(transposed_matrix_dfb, cmap='inferno', aspect='auto', extent=extent)
axes[0].set_xlabel('MJD')
axes[0].set_ylabel('pulse phase')
axes[0].grid(False)
axes[0].xaxis.set_visible(False)

axes[1].imshow(even_single_recon_dfb, cmap='inferno', aspect='auto', extent=extent)
axes[1].set_xlabel('MJD')
axes[1].set_ylabel('PC2 phase')
axes[1].grid(False)
axes[1].xaxis.set_visible(False)

axes[2].plot(smoothed_MJD_dfb, smoothed_shape_parameter_dfb1, color='black')
axes[2].set_ylabel(f'SP 2')
axes[2].set_xlim(evenly_spaced_mjd_dfb[0], evenly_spaced_mjd_dfb[-1])
axes[2].xaxis.set_visible(False)
for spine in axes[2].spines.values():
    if spine == axes[2].spines['top'] or spine == axes[2].spines['right'] or spine == axes[2].spines['bottom']:
        spine.set_visible(False)
axes[2].tick_params(axis='both', which='both', bottom=False, top=False, left=True, right=False)

axes[3].plot(spindown_MJD_dfb, spindown_rate_dfb, color='red')  # Replace x_data and y_data with your data
axes[3].set_xlabel('Modified Julian Days')  # Replace with appropriate label
axes[3].set_ylabel(r'$\dot{\nu}$')  # Replace with appropriate label
axes[3].set_xlim(evenly_spaced_mjd_dfb[0], evenly_spaced_mjd_dfb[-1])
for spine in axes[3].spines.values():
    if spine == axes[3].spines['top'] or spine == axes[3].spines['right']:
        spine.set_visible(False)
axes[3].tick_params(axis='both', which='both', bottom=True, top=False, left=True, right=False)


fig.suptitle('', fontsize=16)
plt.subplots_adjust(hspace=0)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

##code to compare specific PCs and SPs - new

i=0
single_eval_new = pd.DataFrame(eigenvalues_new.iloc[:, i])
single_evec_new = pd.DataFrame(eigenvectors_new.iloc[i ,:]).T
single_recon_new = single_eval_new @ single_evec_new
even_single_recon_new = createeven_df_andtranspose(MJD_new,evenly_spaced_mjd_new,single_recon_new)

fig, axes = plt.subplots(5, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1,1,0.15,0.5,0.5]})
extent = [evenly_spaced_mjd_new[0], evenly_spaced_mjd_new[-1], len(eigenvectors_new.iloc[0,:]), 0]
axes[0].imshow(transposed_matrix_new, cmap='inferno', aspect='auto', extent=extent)
axes[0].set_xlabel('MJD')
axes[0].set_ylabel('pulse phase')
axes[0].grid(False)
axes[0].xaxis.set_visible(False)

axes[1].imshow(even_single_recon_new, cmap='inferno', aspect='auto', extent=extent)
axes[1].set_xlabel('MJD')
axes[1].set_ylabel('PC1 phase')
axes[1].grid(False)
axes[1].xaxis.set_visible(False)

axes[2].plot()
axes[2].set_xlim(evenly_spaced_mjd_new[0], evenly_spaced_mjd_new[-1])
for x in np.array(MJD_new):
    axes[2].axvline(x)
axes[2].xaxis.set_visible(False)
axes[2].yaxis.set_visible(False)
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)
axes[2].spines['bottom'].set_visible(False)
axes[2].spines['left'].set_visible(False)

axes[3].plot(smoothed_MJD_new, smoothed_shape_parameter_new, color='black')
axes[3].set_ylabel(f'SP 1')
axes[3].set_xlim(evenly_spaced_mjd_new[0], evenly_spaced_mjd_new[-1])
axes[3].xaxis.set_visible(False)
for spine in axes[3].spines.values():
    if spine == axes[3].spines['top'] or spine == axes[3].spines['right'] or spine == axes[3].spines['bottom']:
        spine.set_visible(False)
axes[3].tick_params(axis='both', which='both', bottom=False, top=False, left=True, right=False)

axes[4].plot(spindown_MJD_new, spindown_rate_new, color='red')  # Replace x_data and y_data with your data
axes[4].set_xlabel('Modified Julian Days')  # Replace with appropriate label
axes[4].set_ylabel(r'$\dot{\nu}$')  # Replace with appropriate label
axes[4].set_xlim(evenly_spaced_mjd_new[0], evenly_spaced_mjd_new[-1])
for spine in axes[4].spines.values():
    if spine == axes[4].spines['top'] or spine == axes[4].spines['right']:
        spine.set_visible(False)
axes[4].tick_params(axis='both', which='both', bottom=True, top=False, left=True, right=False)

