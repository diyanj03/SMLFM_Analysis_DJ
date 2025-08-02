import os
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_3Dlocs(filepath):
    """
    Plots light field data from a CSV file.
    """

    locs_file = pd.read_csv(filepath)

    coords_df = locs_file.iloc[:, 2:5]

    # Check that all values in selected columns are numeric
    if not coords_df.applymap(np.isreal).all().all():
        raise TypeError("Coordinate columns contain non-numeric values. Likely due to a bug in 3D fitting. \n Try re-running 3D fitting for this dataset by slightly changing the mla X, Y offset in the 3D_config json file.")

    coordinates = locs_file.iloc[:, 2:5].values / 1000 #conversion to microns

    fig = plt.figure(figsize=(16, 8))  

    # Subplot 1: Z vs Y (2D)
    ax1 = fig.add_subplot(221)  # 2 rows, 2 columns, plot 1
    ax1.scatter(coordinates[:, 1], coordinates[:, 2], c=coordinates[:, 2], marker='.', s=1)
    ax1.set_xlabel('Y (μm)', fontsize=12, color='black')
    ax1.set_ylabel('Z (μm)', fontsize=12, color='black')
    ax1.axis('equal')
    ax1.set_title('Z vs Y (2D)')

    # Subplot 2: Y vs X (2D)
    ax2 = fig.add_subplot(222)  # 2 rows, 2 columns, plot 2
    ax2.scatter(coordinates[:, 0], coordinates[:, 1], c=coordinates[:, 2], marker='.', s=1)
    ax2.set_xlabel('X (μm)', fontsize=12, color='black')
    ax2.set_ylabel('Y (μm)', fontsize=12, color='black')
    ax2.axis('equal')
    ax2.set_title('Y vs X (2D)')

    # Subplot 3: Z vs X (2D)
    ax3 = fig.add_subplot(223)  # 2 rows, 2 columns, plot 3
    ax3.scatter(coordinates[:, 0], coordinates[:, 2], c=coordinates[:, 2], marker='.', s=1)
    ax3.set_xlabel('X (μm)', fontsize=12, color='black')
    ax3.set_ylabel('Z (μm)', fontsize=12, color='black')
    ax3.axis('equal')
    ax3.set_title('Z vs X (2D)')

    # Subplot 4: 3D Interactive Plot
    ax4 = fig.add_subplot(224, projection='3d')  # 2 rows, 2 columns, plot 4
    ax4.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c=coordinates[:, 2], marker='.', s=1)
    ax4.set_xlabel('X (μm)', fontsize=12, color='black')
    ax4.set_ylabel('Y (μm)', fontsize=12, color='black')
    ax4.set_zlabel('Z (μm)', fontsize=12, color='black')
    ax4.axis('equal')
    ax4.xaxis.pane.fill = False
    ax4.yaxis.pane.fill = False
    ax4.zaxis.pane.fill = False
    ax4.xaxis.line.set_color("black")
    ax4.yaxis.line.set_color("black")
    ax4.zaxis.line.set_color("black")
    ax4.set_title('3D Plot')

    plt.tight_layout() #prevents overlapping of titles.
    plt.show()

     # Calculate and return min/max of scatter points 
    xmin, xmax = coordinates[:, 0].min(), coordinates[:, 0].max()
    ymin, ymax = coordinates[:, 1].min(), coordinates[:, 1].max()
    zmin, zmax = coordinates[:, 2].min(), coordinates[:, 2].max()

    data_bounds = [xmin, xmax, ymin, ymax, zmin, zmax]

    return data_bounds, filepath



def crop_3Dlocs(root_dir, in_filepath, data_bounds, user_axis_limits):
    """
    Crops light field data and generates plots with axis limits set to cropping bounds (default is min max of the original so no cropping).

    Args: root_dir, filepath, data_bounds, user_axis_limits

    """
    final_cropping_bounds = np.zeros(6)
    for idx in range(6):
        if user_axis_limits[idx] is not None:
             final_cropping_bounds[idx] = user_axis_limits[idx]
        else:
             final_cropping_bounds[idx] = data_bounds[idx]

    filename = os.path.basename(in_filepath)


    axis_limits = final_cropping_bounds*1000 # conversion back to nm for cropping the localisations from csv.
    
    fm_df = pd.read_csv(in_filepath)
    coordinates = fm_df.iloc[:, 2:5].values

    # get coordinates within axis limits
    keep_idx = (
        (axis_limits[0] <= coordinates[:, 0]) &
        (axis_limits[1] >= coordinates[:, 0]) &
        (axis_limits[2] <= coordinates[:, 1]) &
        (axis_limits[3] >= coordinates[:, 1]) &
        (axis_limits[4] <= coordinates[:, 2]) &
        (axis_limits[5] >= coordinates[:, 2])
    )

    cropped_data = fm_df[keep_idx]
    cropped_dir = os.path.join(root_dir, 'data', 'cropped_3Dlocs') 
    os.makedirs(cropped_dir, exist_ok=True)
    
    figure_dir = os.path.join(cropped_dir, 'figures')
    os.makedirs(figure_dir, exist_ok=True)


    crop_filename = os.path.basename(in_filepath).replace('_formatted.csv','_cropped.csv') 
    cropped_path = os.path.join(cropped_dir, crop_filename)

    cropped_data.to_csv(cropped_path, index=False)

    print(
    f"Applied cropping to '{filename}' using the following following axis limits (μm):\n"
    f"X [{final_cropping_bounds[0]:.2f}, {final_cropping_bounds[1]:.2f}]\n"
    f"Y [{final_cropping_bounds[2]:.2f}, {final_cropping_bounds[3]:.2f}]\n"
    f"Z [{final_cropping_bounds[4]:.2f}, {final_cropping_bounds[5]:.2f}]\n"
    "Cropped locs3D csv saved in the '/data/cropped_3Dlocs' directory.\n"
    f"Number of 3D localisations before cropping: {fm_df.shape[0]}\n"
    f"Number of 3D localisations after cropping: {cropped_data.shape[0]}" 
    )

    # Plotting
    fig = plt.figure(figsize=(16, 8))

    um_cropped_y = cropped_data['y (nm)'] / 1000
    um_cropped_z = cropped_data['z (nm)'] /1000
    um_cropped_x = cropped_data['x (nm)'] /1000
    # Subplot 1: Z vs Y (2D)
    ax1 = fig.add_subplot(221)
    ax1.scatter(um_cropped_y, um_cropped_z, c=um_cropped_z, marker='.', s=1)
    ax1.set_xlabel('Y (μm)', fontsize=12, color='black')
    ax1.set_ylabel('Z (μm)', fontsize=12, color='black')
    ax1.set_title('Z vs Y (2D)')
    ax1.set_xlim(final_cropping_bounds[2], final_cropping_bounds[3])  # Set xlim to ymin, ymax
    ax1.set_ylim(final_cropping_bounds[4], final_cropping_bounds[5])  # Set ylim to zmin, zmax
    ax1.axis('equal')

    # Subplot 2: Y vs X (2D)
    ax2 = fig.add_subplot(222)
    ax2.scatter(um_cropped_x, um_cropped_y, c=um_cropped_z, marker='.', s=1)
    ax2.set_xlabel('X (μm)', fontsize=12, color='black')
    ax2.set_ylabel('Y (μm)', fontsize=12, color='black')
    ax2.set_title('Y vs X (2D)')
    ax2.set_xlim(final_cropping_bounds[0], final_cropping_bounds[1])  # Set xlim to xmin, xmax
    ax2.set_ylim(final_cropping_bounds[2], final_cropping_bounds[3])  # Set ylim to ymin, ymax
    ax2.axis('equal')

    # Subplot 3: Z vs X (2D)
    ax3 = fig.add_subplot(223)
    ax3.scatter(um_cropped_x, um_cropped_z, c=um_cropped_z, marker='.', s=1)
    ax3.set_xlabel('X (μm)', fontsize=12, color='black')
    ax3.set_ylabel('Z (μm)', fontsize=12, color='black')
    ax3.set_title('Z vs X (2D)')
    ax3.set_xlim(final_cropping_bounds[0], final_cropping_bounds[1])  # Set xlim to xmin, xmax
    ax3.set_ylim(final_cropping_bounds[4], final_cropping_bounds[5])  # Set ylim to zmin, zmax
    ax3.axis('equal')

    # Subplot 4: 3D Interactive Plot
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.scatter(um_cropped_x, um_cropped_y, um_cropped_z, c=um_cropped_z, marker='.', s=2)
    ax4.set_xlabel('X (μm)', fontsize=12, color='black')
    ax4.set_ylabel('Y (μm)', fontsize=12, color='black')
    ax4.set_zlabel('Z (μm)', fontsize=12, color='black')
    ax4.xaxis.pane.fill = False
    ax4.yaxis.pane.fill = False
    ax4.zaxis.pane.fill = False
    ax4.xaxis.line.set_color("black")
    ax4.yaxis.line.set_color("black")
    ax4.zaxis.line.set_color("black")
    ax4.set_title('3D Plot')
    ax4.set_xlim(final_cropping_bounds[0], final_cropping_bounds[1])  # Set xlim to xmin, xmax
    ax4.set_ylim(final_cropping_bounds[2], final_cropping_bounds[3])  # Set ylim to ymin, ymax
    ax4.set_zlim(final_cropping_bounds[4], final_cropping_bounds[5])  # Set zlim to zmin, zmax
    ax4.axis('equal')

    plt.tight_layout()
    plt.show()

    # Save figures
    file_name_prefix = os.path.splitext(crop_filename)[0]
    fig.savefig(os.path.join(figure_dir, f'{file_name_prefix}_scatter.png'), dpi = 400)
