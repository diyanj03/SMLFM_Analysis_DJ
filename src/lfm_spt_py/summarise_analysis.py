import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import numpy as np
from matplotlib import cm
import smlfm.graphs as graphs



def numFrames(input_dir):
    dict_numFrames = {}
    if not os.path.exists(input_dir):
        print(f"[FAIL] 2D locs dir not found: {input_dir}")
        return None
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_dir, file), skiprows=8)
            numFrames = df['#Frame'].max()
            dataset_name = file.split('_locs2D.csv')[0]
            dict_numFrames[dataset_name] = numFrames
    return dict_numFrames

def num2D_locs(input_dir):
    df_2D_locs = {}
    if not os.path.exists(input_dir):
        print(f"[FAIL] 2D locs dir not found: {input_dir}")
        return df_2D_locs

    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_dir, file), skiprows=8)
            num2dlocs = len(df)
            dataset_name = file.split('_locs2D.csv')[0]
            df_2D_locs[dataset_name] = num2dlocs

    return df_2D_locs


def num3D_raw_locs(input_dir):
    df_3d_locs = {}
    if not os.path.exists(input_dir):
        print(f"[FAIL] Raw 3D locs dir not found: {input_dir}")
        return df_3d_locs

    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_dir, file))
            num3dlocs = len(df)
            dataset_name = file.split('_locs3D_formatted.csv')[0]
            df_3d_locs[dataset_name] = num3dlocs

    return df_3d_locs


def num3D_cropped_locs(input_dir):
    df_3d_locs = {}
    if not os.path.exists(input_dir):
        print(f"[FAIL] Cropped 3D locs dir not found: {input_dir}")
        return df_3d_locs

    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_dir, file))
            num3dlocs = len(df)
            dataset_name = file.split('_locs3D_cropped.csv')[0]
            df_3d_locs[dataset_name] = num3dlocs

    return df_3d_locs


def mean_precision(input_dir):
    def get_precision(results_folder, max_lat_err=200, min_views=3):
        locs_3d_path = os.path.join(results_folder, 'locs3D.csv')
        locs_3d_data = np.genfromtxt(locs_3d_path, delimiter=',')
        mean_lat_err, mean_ax_err = graphs.get_precision(
            locs_3d_data,
            max_lat_err,
            min_views
        )
        return mean_lat_err, mean_ax_err

    df_latPrecision = {}
    df_axPrecision = {}
    if not os.path.exists(input_dir):
        print(f"[Warning] 3D_fitting_results dir not found. cannot output precisions.")
        return None
    
    for foldername in os.listdir(input_dir):
        datasetName = foldername
        results_folder = os.path.join(input_dir, foldername)
        lat_err, ax_err = get_precision(results_folder)
        df_latPrecision[datasetName] = lat_err*1000
        df_axPrecision[datasetName] = ax_err*1000

    return df_latPrecision, df_axPrecision



def numTracks(input_dir):
    df_tracks = {}
    if not os.path.exists(input_dir):
        print(f"[FAIL] Tracks dir not found: {input_dir}")
        return df_tracks

    for file in os.listdir(input_dir):
        if file.endswith('positionsFramesIntensity.csv'):
            df = pd.read_csv(os.path.join(input_dir, file), low_memory=False)
            numtracks = len(df)
            dataset_name = file.split('_positionsFramesIntensity.csv')[0]
            df_tracks[dataset_name] = numtracks

    return df_tracks


def perfConf(matlab_results_dir):
    perfovstats_filepath = os.path.join(matlab_results_dir, 'AllTrajectories', 'PerFOVstats.csv')
    if not os.path.exists(perfovstats_filepath):
        print(f"[FAIL] File not found: {perfovstats_filepath}")
        raise FileNotFoundError(f"File not found: {perfovstats_filepath}")

    
    df = pd.read_csv(perfovstats_filepath)
    confPerc_list = df['confPerc'].fillna(0).where(df['confPerc'] >= 0, 0)
    sample_list_editted = df['originDataset'].str.replace('_trackPositions.csv', '', regex=False)
    confPerc_list = confPerc_list[confPerc_list.notna() & (confPerc_list >= 0)].tolist()
    
    dict_confPerc = dict(zip(sample_list_editted, confPerc_list))

    return dict_confPerc



def diffConst(matlab_results_dir):
    csv_path = os.path.join(matlab_results_dir, 'AllTrajectories', 'diffusionConst.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    df = pd.read_csv(csv_path)

    avg_diffConst_dict = {}

    for x in (df['originDataset']).unique():
        df_perfov = df[df['originDataset'] == x]
        dataset_name = x.replace('_trackPositions.csv', '')
        avg_diffConst_dict[dataset_name] = np.mean(df_perfov['diffusionConst'])

    return avg_diffConst_dict


def anomalous_exp(matlab_results_dir):
    csv_path = os.path.join(matlab_results_dir, 'AllTrajectories', 'alpha.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    df = pd.read_csv(csv_path)

    avg_alpha_dict = {}

    for x in (df['originDataset']).unique():
        df_perfov = df[df['originDataset'] == x]
        dataset_name = x.replace('_trackPositions.csv', '')
        avg_alpha_dict[dataset_name] = np.mean(df_perfov['alpha'])

    return avg_alpha_dict


def compile_stats(sample_name, root_directory, matlab_results_dir=None, custom_dir=None, only_matlab=False):
    """ 
    Compiles statistics (number of 2D locs, 3D locs, tracks, confPerc) per dataset into a summary CSV file.
    Args:
        sample_name (str): Name of the sample for labeling the summary.
        root_directory (str): Path to root_directory to read the data folders and saving the summary stats.
        matlab_results_dir (str, optional): Directory containing MATLAB results to include % trajs confined in summary
        custom_data_dir (str, optional): replaces 'root_dir/data' with custom directory path containing the '2D_locs_csv', '3D_fitting_results', 'formatted_3Dlocs', 'cropped_3Dlocs', 'tracks' folders for the given sample.  
        only_matlab (bool, optional): set to True if no data dir input, just confPerc, diffusionConst and alpha.
    Returns:
        str: Path to the generated summary CSV file.
    """
    if not only_matlab:
        if custom_dir:
            print(f'Reading subfolders, "2D_locs_csv", "3D_fitting_results", "formatted_3Dlocs","cropped_3Dlocs","tracks", within the custom directory as provided to generate analysis summary stats...\n(note: ensure the mentioned subfolders all contain datasets only corresponding to "{sample_name}")')
            locs2D_dir = os.path.join(custom_dir,'2D_locs_csv')
            fm_3Dlocs_dir = os.path.join(custom_dir,'formatted_3Dlocs')
            cropped_3Dlocs_dir = os.path.join(custom_dir, 'cropped_3Dlocs')
            tracks_dir = os.path.join(custom_dir, 'tracks')
            fitting3D_dir = os.path.join(custom_dir, '3D_fitting_results')
        else:
            print(f'Reading subfolders, "2D_locs_csv","3D_fitting_results", "formatted_3Dlocs","cropped_3Dlocs","tracks", within the "../data" directory to generate analysis summary stats...\n(note: ensure the mentioned subfolders all contain datasets only corresponding to "{sample_name}")')
            locs2D_dir = os.path.join(root_directory, 'data', '2D_locs_csv')
            fm_3Dlocs_dir = os.path.join(root_directory, 'data', 'formatted_3Dlocs')
            cropped_3Dlocs_dir = os.path.join(root_directory, 'data', 'cropped_3Dlocs')
            tracks_dir = os.path.join(root_directory, 'data', 'tracks')
            fitting3D_dir = os.path.join(root_directory, 'data', '3D_fitting_results')



        if matlab_results_dir:
            df = pd.DataFrame({
                'numFrames': numFrames(locs2D_dir),
                'num_2Dlocs': num2D_locs(locs2D_dir),
                'mean_lateralPrecision': mean_precision(fitting3D_dir)[0],
                'num_raw3Dlocs': num3D_raw_locs(fm_3Dlocs_dir),
                'num_cropped3Dlocs': num3D_cropped_locs(cropped_3Dlocs_dir),
                'mean_axialPrecision': mean_precision(fitting3D_dir)[1],
                'numTracks': numTracks(tracks_dir),
                'confPerc': perfConf(matlab_results_dir),
                'diffusionConst (all)': diffConst(matlab_results_dir),
                'alpha (all)': anomalous_exp(matlab_results_dir)
            }).T
        else: 
            df = pd.DataFrame({
                'numFrames': numFrames(locs2D_dir),
                'num_2Dlocs': num2D_locs(locs2D_dir),
                'mean_lateralPrecision': mean_precision(fitting3D_dir)[0],
                'num_raw3Dlocs': num3D_raw_locs(fm_3Dlocs_dir),
                'mean_axialPrecision': mean_precision(fitting3D_dir)[1],
                'num_cropped3Dlocs': num3D_cropped_locs(cropped_3Dlocs_dir),
                'numTracks': numTracks(tracks_dir),
            }).T

        df.index.name = 'Property'
        df = df.T
        df.index.name = 'Sample'

        # calculate per-frame values and add them as new columns
        df['num_2Dlocs_perframe'] = df['num_2Dlocs'] / df['numFrames']
        df['num_raw3Dlocs_perframe'] = df['num_raw3Dlocs'] / df['numFrames']
        df['num_cropped3Dlocs_perframe'] = df['num_cropped3Dlocs'] / df['numFrames']
        df['numTracks_perframe'] = df['numTracks'] / df['numFrames']

        if matlab_results_dir:
            destination_dir = os.path.join(root_directory, 'results', sample_name + '_perFOV_results', 'analysisSummary')
        else:
            destination_dir = os.path.join(root_directory, 'results', sample_name + '_analysisSummary')
        os.makedirs(destination_dir, exist_ok=True)
        csv_destination_path = os.path.join(destination_dir, f'{sample_name}_stats.csv')
        df.to_csv(csv_destination_path)

        mean_lat_precision = np.mean(df['mean_lateralPrecision'])
        mean_ax_precision = np.mean(df['mean_axialPrecision'])

        # Per-frame and filtered stats
        valid_2D = df.dropna(subset=['numFrames', 'num_2Dlocs'])
        mean_2Dlocs = valid_2D['num_2Dlocs'].mean()
        std_2Dlocs = valid_2D['num_2Dlocs'].std()
        mean_2Dlocs_pf = (valid_2D['num_2Dlocs'] / valid_2D['numFrames']).mean()
        std_2Dlocs_pf = (valid_2D['num_2Dlocs'] / valid_2D['numFrames']).std()

        valid_raw3D = df.dropna(subset=['numFrames', 'num_raw3Dlocs'])
        mean_raw3D = valid_raw3D['num_raw3Dlocs'].mean()
        std_raw3D = valid_raw3D['num_raw3Dlocs'].std()
        mean_raw3D_pf = (valid_raw3D['num_raw3Dlocs'] / valid_raw3D['numFrames']).mean()
        std_raw3D_pf = (valid_raw3D['num_raw3Dlocs'] / valid_raw3D['numFrames']).std()

        valid_crop3D = df.dropna(subset=['numFrames', 'num_cropped3Dlocs'])
        mean_crop3D = valid_crop3D['num_cropped3Dlocs'].mean()
        std_crop3D = valid_crop3D['num_cropped3Dlocs'].std()
        mean_crop3D_pf = (valid_crop3D['num_cropped3Dlocs'] / valid_crop3D['numFrames']).mean()
        std_crop3D_pf = (valid_crop3D['num_cropped3Dlocs'] / valid_crop3D['numFrames']).std()

        valid_tracks = df.dropna(subset=['numFrames', 'numTracks'])
        mean_tracks = valid_tracks['numTracks'].mean()
        std_tracks = valid_tracks['numTracks'].std()
        mean_tracks_pf = (valid_tracks['numTracks'] / valid_tracks['numFrames']).mean()
        std_tracks_pf = (valid_tracks['numTracks'] / valid_tracks['numFrames']).std()

        summary_txt_path = os.path.join(destination_dir, f'{sample_name}_stats_summary.txt')
        with open(summary_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"=== {sample_name} - Summary Statistics ===\n\n")

            f.write(f">> Number of 2D localisations:\n")
            f.write(f"Mean: {mean_2Dlocs:.2f}\n")
            f.write(f"Standard Deviation: {std_2Dlocs:.2f}\n")
            f.write(f"Per frame mean: {mean_2Dlocs_pf:.2f}\nPer frame Std: {std_2Dlocs_pf:.2f}\n")
            f.write(f"Mean lateral precision (nm): {mean_lat_precision}\n\n")

            f.write(f">> Number of raw 3D localisations:\n")
            f.write(f"Mean: {mean_raw3D:.2f}\n")
            f.write(f"Standard Deviation: {std_raw3D:.2f}\n")
            f.write(f"Per frame mean: {mean_raw3D_pf:.2f}\nPer frame Std: {std_raw3D_pf:.2f}\n")
            f.write(f"Mean axial precision (nm): {mean_ax_precision}\n\n")

            f.write(f">> Number of cropped 3D localisations:\n")
            f.write(f"Mean: {mean_crop3D:.2f}\n")
            f.write(f"Standard Deviation: {std_crop3D:.2f}\n")
            f.write(f"Per frame mean: {mean_crop3D_pf:.2f}\nPer frame Std: {std_crop3D_pf:.2f}\n\n")

            f.write(f">> Number of tracks:\n")
            f.write(f"Mean: {mean_tracks:.2f}\n")
            f.write(f"Standard Deviation: {std_tracks:.2f}\n")
            f.write(f"Per frame mean: {mean_tracks_pf:.2f}\nPer frame Std: {std_tracks_pf:.2f}\n\n")
        if matlab_results_dir:
            print(f'Successfully saved "{sample_name}" summary stats to the "../results/{sample_name}_perFOV_results" folder.\n')
        else:
            print(f'Successfully saved "{sample_name}" summary stats to the "../results/{sample_name}_analysisSummary" folder.\n')

    else:
        if matlab_results_dir:
            df = pd.DataFrame({
                'confPerc': perfConf(matlab_results_dir),
                'diffusionConst (all)': diffConst(matlab_results_dir),
                'alpha (all)': anomalous_exp(matlab_results_dir)
            }).T
        else: 
            return

        df.index.name = 'Property'
        df = df.T
        df.index.name = 'Sample'

        destination_dir = os.path.join(root_directory, 'results', sample_name + '_perFOV_results', 'paramValues')
        os.makedirs(destination_dir, exist_ok=True)
        csv_destination_path = os.path.join(destination_dir, f'{sample_name}_stats.csv')
        df.to_csv(csv_destination_path)


def correlation_analysis(stats_csv_path, sample_name, root_directory, rgb=(0.5, 0.5, 0.5)):
    """
    Correlations analysis between number of trajectories and percentage of trajectories bound to chromatin per dataset.
    Args:
        root_directory (str): path to SMLFM_analysis.
        stats_csv_path (str): Path to the CSV file containing summary statistics generated by compile_stats.
        sample_name (str): Name of the sample for labeling the plot.
        rgb (tuple, optional): RGB color values for the plot. 
    """
    df = pd.read_csv(stats_csv_path)

    x = df['numTracks']
    y = df['confPerc']
    mask = x.notna() & y.notna()

    # Compute Pearson correlation
    r, p = pearsonr(x[mask], y[mask])

    # Scatter plot with regression line
    sns.regplot(x=x[mask], y=y[mask], color=rgb)
    plt.title(f'Pearson r = {r:.3f}, p = {p:.3e}')
    plt.xlabel('Number of Trajectories')
    plt.ylabel('Percentage of Trajectories bound to Chromatin (%)')
    plt.tight_layout()
    destination = os.path.join(root_directory, 'analysis_summary', 'correlation_analysis')
    os.makedirs(destination, exist_ok=True)
    plt.savefig(os.path.join(destination, f'{sample_name}_numTracks_vs_confPerc.pdf'))
    plt.show()


def jitter_pipelineStats(csv_list, sample_labels, root_directory, rgb_list=None, ylim_min=None):
    """
    Args:
    - csv_list (list of str): list of paths to the summary stats file generated by compile stats.
    - sample_labels (list of str): labels for each sample in the same order as list_csv_paths.
    - root_directory: path to root_dir
    - rgb_list (list of tuples): optional list of RGB color codes for each dataset in the same order as list_csv_paths.
    - ylim_min (float, optional): to set yminimun on plots
    """
    
    print(f'Generating perFOV distributions of data analysis properties: 2Dlocs, 3Dlocs, numTracks...')
    for column_heading in ['num_2Dlocs_perframe', 'num_raw3Dlocs_perframe', 'num_cropped3Dlocs_perframe', 'numTracks_perframe', 'mean_axialPrecision', 'mean_lateralPrecision']:
        distribution_lists = []
        skip_column = False

        for path in csv_list:
            df = pd.read_csv(path)
            if column_heading not in df.columns:
                print(f"Column '{column_heading}' not found in {path}, skipping this parameter.")
                skip_column = True
                break
            values = df[column_heading].dropna().tolist()
            distribution_lists.append(values)

        if skip_column:
            continue

        if all(len(lst) == 0 for lst in distribution_lists):
            print(f"No data found for column '{column_heading}', skipping.")
            continue

        means = {label: np.mean(dist) for label, dist in zip(sample_labels, distribution_lists)}
        stds = {label: np.std(dist) for label, dist in zip(sample_labels, distribution_lists)}
        xs = [np.random.normal(i + 1, 0.04, len(group)) for i, group in enumerate(distribution_lists)]

        width_per_dataset = 6.4 / 3
        fig_width = width_per_dataset * len(distribution_lists)

        plt.figure(figsize=(fig_width, 4.8))

        plt.boxplot(
            distribution_lists,
            tick_labels=sample_labels,
            widths=0.28 if len(distribution_lists) == 2 else 0.5,
            medianprops=dict(color='black')
        )

        plt.ylabel(f'{column_heading}', fontsize=12)

        if rgb_list and len(rgb_list) == len(distribution_lists):
            for x, val, rgb in zip(xs, distribution_lists, rgb_list):
                plt.scatter(x, val, color=rgb, alpha=0.4)
        else:
            clevels = np.linspace(0., 1., len(distribution_lists))
            for x, val, clevel in zip(xs, distribution_lists, clevels):
                plt.scatter(x, val, color=cm.prism(clevel), alpha=0.4)

        all_values = np.concatenate(distribution_lists)
        padding = 0.1 * (np.max(all_values) - np.min(all_values)) if len(all_values) > 0 else 1
        lower_limit = ylim_min if ylim_min is not None else np.min(all_values) - padding
        upper_limit = np.max(all_values) + padding

        plt.ylim(lower_limit, upper_limit)

        destination_dir = os.path.join(root_directory, 'results', '_'.join(sample_labels) + '_perFOV_results')
        os.makedirs(destination_dir, exist_ok=True)
        destination_subdir = os.path.join(destination_dir, 'analysisParams', column_heading)
        os.makedirs(destination_subdir, exist_ok=True)

        plot_path = os.path.join(destination_subdir, '_'.join(sample_labels) + f'_{column_heading}.pdf')
        plt.savefig(plot_path)
        plt.show()

        txt_path = os.path.join(destination_subdir, '_'.join(sample_labels) + f'_{column_heading}_stats.txt')
        with open(txt_path, 'w', encoding='utf-8') as fileHandle:
            fileHandle.write(f"=== {column_heading} - Stats Summary ===\n\n")
            fileHandle.write(f">> Means:\n")
            for label in sample_labels:
                fileHandle.write(f"{label}: {means[label]:.5f}\n")
            fileHandle.write("\n>> Standard Deviations:\n")
            for label in sample_labels:
                fileHandle.write(f"{label}: {stds[label]:.5f}\n")
            fileHandle.write("\n")

    print(f'Successfully saved perFOV analysis params to "../results/{'_'.join(sample_labels)}_perFOV_results/analysisParams" folder.\n')
