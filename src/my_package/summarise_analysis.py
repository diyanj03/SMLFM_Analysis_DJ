import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import numpy as np





def num2D_locs(input_dir):
    df_2D_locs = {}
    if not os.path.exists(input_dir):
        print(f"[FAIL] 2D locs dir not found: {input_dir}")
        return df_2D_locs
    print(f"[OK] Found 2D locs dir: {input_dir}")
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_dir, file), skiprows=8)
            num2dlocs = len(df)
            dataset_name = file.split('_locs2D.csv')[0]
            df_2D_locs[dataset_name] = num2dlocs
    print("[DONE] 2D locs loaded")
    return df_2D_locs


def num3D_raw_locs(input_dir):
    df_3d_locs = {}
    if not os.path.exists(input_dir):
        print(f"[FAIL] Raw 3D locs dir not found: {input_dir}")
        return df_3d_locs
    print(f"[OK] Found raw 3D locs dir: {input_dir}")
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_dir, file))
            num3dlocs = len(df)
            dataset_name = file.split('_locs3D_formatted.csv')[0]
            df_3d_locs[dataset_name] = num3dlocs
    print("[DONE] Raw 3D locs loaded")
    return df_3d_locs


def num3D_cropped_locs(input_dir):
    df_3d_locs = {}
    if not os.path.exists(input_dir):
        print(f"[FAIL] Cropped 3D locs dir not found: {input_dir}")
        return df_3d_locs
    print(f"[OK] Found cropped 3D locs dir: {input_dir}")
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_dir, file))
            num3dlocs = len(df)
            dataset_name = file.split('_locs3D_cropped.csv')[0]
            df_3d_locs[dataset_name] = num3dlocs
    print("[DONE] Cropped 3D locs loaded")
    return df_3d_locs


def numTracks(input_dir):
    df_tracks = {}
    if not os.path.exists(input_dir):
        print(f"[FAIL] Tracks dir not found: {input_dir}")
        return df_tracks
    print(f"[OK] Found tracks dir: {input_dir}")
    for file in os.listdir(input_dir):
        if file.endswith('positionsFramesIntensity.csv'):
            df = pd.read_csv(os.path.join(input_dir, file), low_memory=False)
            numtracks = len(df)
            dataset_name = file.split('_locs3D_cropped_positionsFramesIntensity.csv')[0]
            df_tracks[dataset_name] = numtracks
    print("[DONE] Tracks loaded")
    return df_tracks


def perfConf(matlab_results_dir):
    perfovstats_filepath = os.path.join(matlab_results_dir, 'AllTrajectories', 'PerFOVstats.csv')
    if not os.path.exists(perfovstats_filepath):
        print(f"[FAIL] File not found: {perfovstats_filepath}")
        raise FileNotFoundError(f"File not found: {perfovstats_filepath}")
    print(f"[OK] Found PerFOVstats file: {perfovstats_filepath}")
    
    df = pd.read_csv(perfovstats_filepath)
    confPerc_list = df['confPerc'].fillna(0).where(df['confPerc'] >= 0, 0)
    sample_list_editted = df['originDataset'].str.replace('_locs3D_cropped_trackPositions.csv', '', regex=False)
    confPerc_list = confPerc_list[confPerc_list.notna() & (confPerc_list >= 0)].tolist()
    
    dict_confPerc = dict(zip(sample_list_editted, confPerc_list))
    print("[DONE] Confidence values loaded")
    return dict_confPerc


def compile_stats(root_directory, matlab_results_dir, sample_name):
    """ 
    Compiles statistics (number of 2D locs, 3D locs, tracks, confPerc) per dataset into a summary CSV file.
    Args:
        root_directory (str): Root directory containing the analysis results.  
        matlab_results_dir (str): Directory containing MATLAB results.
        sample_name (str): Name of the sample for labeling the summary.

    Returns:
        str: Path to the generated summary CSV file.
    """

    locs2D_dir = os.path.join(root_directory, data, '2D_locs_csv')
    fm_3Dlocs_dir = os.path.join(root_directory, data, 'formatted_3Dlocs')
    cropped_3Dlocs_dir = os.path.join(root_directory, data, 'cropped_3Dlocs')
    tracks_dir = os.path.join(root_directory, data, 'tracks')
    

    df = pd.DataFrame({
        'num_2Dlocs': num2D_locs(locs2D_dir),
        'num_raw3Dlocs': num3D_raw_locs(fm_3Dlocs_dir),
        'num_cropped3Dlocs': num3D_cropped_locs(cropped_3Dlocs_dir),
        'numTracks': numTracks(tracks_dir),
        'confPerc': perfConf(results_dir)
    }).T

    df.index.name = 'Property'
    df = df.T
    df.index.name = 'Sample'

    destination_dir = os.path.join(root_directory, 'analysis_summary')
    os.makedirs(destination_dir, exist_ok=True)
    destination_path = os.path.join(destination_dir, f'{sample_name}_stats.csv')
    df.to_csv(destination_path)

    # computing_summary_stats:
    mean_2Dlocs = np.mean([v for v in df['num_2Dlocs'] if isinstance(v, (int, float))]) if any(isinstance(v, (int, float)) for v in df['num_2Dlocs']) else 0
    std_2Dlocs = np.std([v for v in df['num_2Dlocs'] if isinstance(v, (int, float))]) if any(isinstance(v, (int, float)) for v in df['num_2Dlocs']) else 0

    mean_raw_3Dlocs = np.mean([v for v in df['num_raw3Dlocs'] if isinstance(v, (int, float))]) if any(isinstance(v, (int, float)) for v in df['num_raw3Dlocs']) else 0
    std_raw_3Dlocs = np.std([v for v in df['num_raw3Dlocs'] if isinstance(v, (int, float))]) if any(isinstance(v, (int, float)) for v in df['num_raw3Dlocs']) else 0

    mean_cropped_3Dlocs = np.mean([v for v in df['num_cropped3Dlocs'] if isinstance(v, (int, float))]) if any(isinstance(v, (int, float)) for v in df['num_cropped3Dlocs']) else 0
    std_cropped_3Dlocs = np.std([v for v in df['num_cropped3Dlocs'] if isinstance(v, (int, float))]) if any(isinstance(v, (int, float)) for v in df['num_cropped3Dlocs']) else 0

    mean_tracks = np.mean([v for v in df['numTracks'] if isinstance(v, (int, float))]) if any(isinstance(v, (int, float)) for v in df['numTracks']) else 0
    std_tracks = np.std([v for v in df['numTracks'] if isinstance(v, (int, float))]) if any(isinstance(v, (int, float)) for v in df['numTracks']) else 0

    summary_txt_path = os.path.join(destination_dir, f'{sample_name}_stats_summary.txt')
    
    with open(summary_txt_path, 'w', encoding='utf-8') as fileHandle:
        fileHandle.write(f"=== {sample_name} - Summary Statistics ===\n\n")
        fileHandle.write(f">> Number of 2D localisations:\n")
        fileHandle.write(f"Mean: {mean_2Dlocs:.2f}\n")
        fileHandle.write(f"Standard Deviation: {std_2Dlocs:.2f}\n\n")
        fileHandle.write(f">> Number of raw 3D localisations:\n")
        fileHandle.write(f"Mean: {mean_raw_3Dlocs:.2f}\n")
        fileHandle.write(f"Standard Deviation: {std_raw_3Dlocs:.2f}\n\n")
        fileHandle.write(f">> Number of cropped 3D localisations:\n")
        fileHandle.write(f"Mean: {mean_cropped_3Dlocs:.2f}\n")
        fileHandle.write(f"Standard Deviation: {std_cropped_3Dlocs:.2f}\n\n")
        fileHandle.write(f">> Number of tracks:\n")
        fileHandle.write(f"Mean: {mean_tracks:.2f}\n")
        fileHandle.write(f"Standard Deviation: {std_tracks:.2f}\n\n")

    return destination_path


def correlation_analysis(root_directory, stats_csv_path, sample_name, rgb=(0.5, 0.5, 0.5)):
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


def jitter_pipelineStats(list_matlab_paths, sample_labels, destination_dir, rgb_list=None):
    """
    Args:
    - list_matlab_paths (list of str): list of paths to matlab output dirs to be compared.
    - sample_labels (list of str): labels for each sample in the same order as list_csv_paths.
    - destination_dir (str): directory where the plot and stats summary will be saved.
    - rgb_list (list of tuples): optional list of RGB color codes for each dataset in the same order as list_csv_paths.
    """

    for column_heading in ['num_2Dlocs', 'num_raw3Dlocs', 'num_cropped3Dlocs', 'numTracks']:
        distribution_lists = []
        for csv_path in list_csv_paths:
            distribution_lists.append(pd.read_csv(csv_path)[column_heading].dropna().tolist())

        means = {}
        stds = {}
        for label, dist_list in zip(sample_labels, distribution_lists):
            means[label] = np.mean(dist_list)
            stds[label] = np.std(dist_list)

        xs = [np.random.normal(i + 1, 0.04, len(group)) for i, group in enumerate(distribution_lists)]

        width_per_dataset = 6.4 / 3
        fig_width = width_per_dataset * len(distribution_lists)

        plt.figure(figsize=(fig_width, 4.8))

        if len(distribution_lists) == 2:
            plt.boxplot(
                distribution_lists,
                tick_labels=sample_labels,
                widths=0.28,
                medianprops=dict(color='black'))
        else:
            plt.boxplot(
                distribution_lists,
                tick_labels=sample_labels,
                medianprops=dict(color='black'))

        plt.ylabel(f'{column_heading}', fontsize=12)

        if rgb_list:
            if len(rgb_list) != len(distribution_lists):
                print("number of rgb codes provided doesn't match number of datasets to compare")
            for x, val, rgb in zip(xs, distribution_lists, rgb_list):
                plt.scatter(x, val, color=rgb, alpha=0.4)
        else:
            clevels = np.linspace(0., 1., len(distribution_lists))
            for x, val, clevel in zip(xs, distribution_lists, clevels):
                plt.scatter(x, val, color=cm.prism(clevel), alpha=0.4)
                
        os.makedirs(destination_dir, exist_ok=True)
        destination_subdir = os.path.join(destination_dir, 'pipeline_stats')
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
            fileHandle.write("\n")

            fileHandle.write(f">> Standard Deviations:\n")
            for label in sample_labels:
                fileHandle.write(f"{label}: {stds[label]:.5f}\n")
            fileHandle.write("\n")



