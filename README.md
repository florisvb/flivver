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
3. To make Figure C, run the notebooks listed under Sup. Fig. 1

# Supp. Figure 1

To run the analysis shown in figure 4C, and supplemental figure 1, first run: `Figure_S1_resolution/replot_fig_4/replot_data_from_fig_4.ipynb`

Then run the following three notebooks in the directories ending in p1deg, 1deg and 2deg:

1. `Figure_S1_resolution/2021_velocity_estimation_figure_4_1deg/reformat_bryson_data_to_hdf5.ipynb`
2. `Figure_S1_resolution/2021_velocity_estimation_figure_4_1deg/noglobalalphadotavg_estimate_velocity_for_receptive_fields_from_bryson_data_alphadots.ipynb`
3. `Figure_S1_resolution/2021_velocity_estimation_figure_4_1deg/analyze_bryson_data_nomatchedfilters_from_corrected_v_over_d.ipynb`

# Figures 5-6

Run the python file `Figure_5-6/make_depth_figures.py`
