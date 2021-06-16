# Source code for the following paper:

Insect inspired visual-based velocity estimation through spatial pooling of optic flow during linear motion

# Data

You can download the raw data from datadryad here: link

Update the `data_directory` location at the top of each of the notebooks before running them. 

# Figure 3

To run the analysis shown in figure 3, run the following jupyter notebooks from start to finish, in order. 

1. Run `insert_flownet_and_alpha_images.ipynb`
2. Run `noglobalalphadotavg_estimate_velocity_for_receptive_fields_from_bryson_data_alphadots.ipynb`
3. Run `make_velocity_estimation_figure.ipynb`

# Figure 4

To run the analysis shown in figure 4, run the following jupyter notebooks from start to finish, in order. 

1. Run `noglobalalphadotavg_estimate_velocity_for_receptive_fields_from_bryson_data_alphadots.ipynb`
2. Run `analyze_bryson_data_nomatchedfilters_from_corrected_v_over_d.ipynb`