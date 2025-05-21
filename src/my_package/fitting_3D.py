import dataclasses
import pkgutil
import time
from datetime import datetime
from pathlib import Path
import os 
import shutil

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import smlfm
import smlfm.graphs


def load_config(root_dir, config_name = None):
    if config_name:
        config_path = os.path.join(root_dir, 'configs', config_name)
        if os.path.exists(config_path):
            with open(config_path, 'rt', encoding='utf-8') as file:
                cfg_dump = file.read()
                cfg = smlfm.Config.from_json(cfg_dump)
            print(f"Configuration file successfully loaded from {config_path}. Modify if needed.")
            return cfg

        print(f"Configuration file not found at {config_path}. Falling back to default.")

    default_path = os.path.join(root_dir, 'configs', 'default_configs', 'fitting3D_configDefault.json')
    with open(default_path, 'rt', encoding='utf-8') as file:
        cfg_dump = file.read()
        cfg = smlfm.Config.from_json(cfg_dump)
    print(f"Default configuration file loaded from {default_path}. WARNING: default settings.")
    return cfg




def fittingAxialLocs(csv_file_path, root_dir, config_name=None, cfg=None):

    tic_total = time.time()
    user_interaction_time = 0
    timestamp = datetime.now()

    if cfg is None:
        try: 
            cfg = load_config(root_dir, config_name)
        except Exception as e:
            print(f"Error loading configuration: {e}.")

    cfg.csv_file = Path(csv_file_path)

    if not cfg.csv_file.exists():
        print(f'ERROR: The CSV file "{cfg.csv_file}" does not exist')
        return None, None

    lfm = smlfm.FourierMicroscope(
        cfg.num_aperture, cfg.mla_lens_pitch,
        cfg.focal_length_mla, cfg.focal_length_obj_lens,
        cfg.focal_length_tube_lens, cfg.focal_length_fourier_lens,
        cfg.pixel_size_camera, cfg.ref_idx_immersion,
        cfg.ref_idx_medium)


    tic = time.time()

    csv = smlfm.LocalisationFile(cfg.csv_file, cfg.csv_format)
    csv.read()

    print(f'Loaded {csv.data.shape[0]} localisations from'
          f' {np.unique(csv.data[:, 0]).shape[0]} unique frames')

    if cfg.show_graphs and cfg.show_all_debug_graphs:
        fig = smlfm.graphs.draw_locs_csv(plt.figure(), csv.data[:, 1:3])
        fig.canvas.manager.set_window_title('Raw localisations')

    csv.scale_peakfit_data(lfm.pixel_size_sample)
    locs_2d_csv = csv.data.copy()

    locs_2d_csv[:, 1] -= locs_2d_csv[:, 1].mean()
    locs_2d_csv[:, 2] -= locs_2d_csv[:, 2].mean()

    if cfg.log_timing:
        print(f'Loading {repr(cfg.csv_file.name)} took {1e3 * (time.time() - tic):.3f} ms')

    # Prepare MLA and rotate it to match the CSV data
    mla = smlfm.MicroLensArray(
        cfg.mla_type, cfg.focal_length_mla, cfg.mla_lens_pitch,
        cfg.mla_optic_size, np.array(cfg.mla_centre))

    if cfg.show_graphs and cfg.show_all_debug_graphs:
        fig = smlfm.graphs.draw_mla(plt.figure(), mla.lens_centres, mla.centre)
        fig.canvas.manager.set_window_title('Micro-lens array centres')

    mla.rotate_lattice(np.deg2rad(cfg.mla_rotation))
    mla.offset_lattice(np.array(cfg.mla_offset) / lfm.mla_to_xy_scale)

    if cfg.show_graphs and cfg.show_all_debug_graphs:
        fig = smlfm.graphs.draw_mla(plt.figure(), mla.lens_centres, mla.centre)
        fig.canvas.manager.set_window_title('Micro-lens array centres rotated')

    # Map localisations to lenses
    tic = time.time()

    lfl = smlfm.Localisations(locs_2d_csv)
    lfl.assign_to_lenses(mla, lfm)

    if cfg.log_timing:
        print(f'Mapping points to lenses took {1e3 * (time.time() - tic):.3f} ms')

    if cfg.show_graphs and cfg.show_mla_alignment_graph:
        fig = smlfm.graphs.draw_locs(
            plt.figure(),
            xy=lfl.locs_2d[:, 3:5],
            lens_idx=lfl.locs_2d[:, 12],
            lens_centres=(mla.lens_centres - mla.centre) * lfm.mla_to_xy_scale,
            mla_centre=np.array([0.0, 0.0]))
        fig.canvas.manager.set_window_title('Localisations with lens centers')

        if cfg.confirm_mla_alignment:
            tic = time.time()
            print('\nVerify that the lenses are properly aligned with the data in the figure.' 
                  'If alignment is correct, type "yes" in the input prompt to confirm.')
            print('If not, type "no", then adjust the MLA rotation and/or offset in the configuration and rerun this cell.')
            plt.show()
            while True:
                data = input('\nAre the lens centres aligned with the data? [yes/no]: ')
                data = 'y' if data == '' else data[0].casefold()
                if data not in ('yes', 'no'):
                    print('Not an appropriate choice.')
                else:
                    if data == 'no':
                        return None, None
                    break
            print('')
            user_interaction_time = time.time() - tic

    # 5. Filter localisations and set alpha model
    tic = time.time()

    if cfg.filter_lenses:
        lfl.filter_lenses(mla, lfm)
    if cfg.filter_rhos is not None:
        lfl.filter_rhos(cfg.filter_rhos)
    if cfg.filter_spot_sizes is not None:
        lfl.filter_spot_sizes(cfg.filter_spot_sizes)
    if cfg.filter_photons is not None:
        lfl.filter_photons(cfg.filter_photons)

    lfl.init_alpha_uv(cfg.alpha_model, lfm, worker_count=cfg.max_workers)

    if cfg.log_timing:
        print(f'Filtering and setting alpha model took {1e3 * (time.time() - tic):.3f} ms')

    # 6. Find system aberrations
    tic = time.time()

    fit_params_cor = dataclasses.replace(
        cfg.fit_params_aberration,
        frame_min=(cfg.fit_params_aberration.frame_min
                   if cfg.fit_params_aberration.frame_min > 0
                   else lfl.min_frame),
        frame_max=(cfg.fit_params_aberration.frame_max
                   if cfg.fit_params_aberration.frame_max > 0
                   else min(1000, lfl.max_frame)),
    )

    print(f'Fitting frames'
          f' {fit_params_cor.frame_min}-{fit_params_cor.frame_max}'
          f' for aberration correction...')
    _, fit_data = smlfm.Fitting.light_field_fit(
        lfl.filtered_locs_2d, lfm.rho_scaling, fit_params_cor,
        worker_count=cfg.max_workers)

    correction = smlfm.Fitting.calculate_view_error(
        lfl.filtered_locs_2d, lfm.rho_scaling, fit_data, cfg.aberration_params)

    lfl.correct_xy(correction)

    if cfg.show_graphs and cfg.show_all_debug_graphs:
        fig = smlfm.graphs.draw_locs(
            plt.figure(),
            xy=lfl.corrected_locs_2d[:, 3:5],
            lens_idx=lfl.corrected_locs_2d[:, 12],
            lens_centres=(mla.lens_centres - mla.centre) * lfm.mla_to_xy_scale,
            mla_centre=np.array([0.0, 0.0]))
        fig.canvas.manager.set_window_title('Corrected localisations')

    if cfg.log_timing:
        print(f'Aberration correction took {1e3 * (time.time() - tic):.3f} ms')

    # Fit full data set on corrected localisations
    tic = time.time()

    fit_params_all = dataclasses.replace(
        cfg.fit_params_full,
        frame_min=(cfg.fit_params_full.frame_min
                   if cfg.fit_params_full.frame_min > 0
                   else lfl.min_frame),
        frame_max=(cfg.fit_params_full.frame_max
                   if cfg.fit_params_full.frame_max > 0
                   else lfl.max_frame),
    )

    print(f'Fitting frames'
          f' {fit_params_all.frame_min}-{fit_params_all.frame_max}...')
    locs_3d, _ = smlfm.Fitting.light_field_fit(
        lfl.corrected_locs_2d, lfm.rho_scaling, fit_params_all,
        worker_count=cfg.max_workers,
        progress_func=lambda frame, min_frame, max_frame:
            print(f'Processing frame'
                  f' {frame - min_frame + 1}/{max_frame - min_frame + 1}...'))

    print(f'Total number of frames used for fitting:'
          f' {np.unique(locs_3d[:, 7]).shape[0]}')
    print(f'Total number of 2D localisations used for fitting:'
          f' {int(np.sum(locs_3d[:, 5]))}')
    print(f'Total number of 3D localisations: {locs_3d.shape[0]}')

    if cfg.log_timing:
        print(f'Complete fitting took {1e3 * (time.time() - tic):.3f} ms')

    # Write the results

    if cfg.save_dir is not None and cfg.save_dir:

        timestamp_str = timestamp.strftime('%Y%m%d-%H%M%S')
        subdir_name = Path(f'{cfg.csv_file.name}')

        results = smlfm.OutputFiles(cfg, subdir_name)
        print(f"Saving results to folder: '{results.folder.name}' within the 'data/3D_fitting_results' directory.")

        try:
            results.mkdir()
        except Exception as ex:
            print(f'ERROR: Failed to create target folder ({repr(ex)})')
        else:
            try:
                results.save_config()
            except Exception as ex:
                print(f'ERROR: Failed to save configuration file ({repr(ex)})')
            try:
                results.save_csv(locs_3d)
            except Exception as ex:
                print(f'ERROR: Failed to save CSV file ({repr(ex)})')
            try:
                results.save_visp(locs_3d)
            except Exception as ex:
                print(f'ERROR: Failed to save ViSP file ({repr(ex)})')
            try:
                results.save_figures()
            except Exception as ex:
                print(f'ERROR: Failed to save figures file ({repr(ex)})')

    # 9. Plotting results

    if cfg.show_graphs and cfg.show_result_graphs:
        fig1, fig2, fig3 = smlfm.graphs.reconstruct_results(
            plt.figure(), plt.figure(), plt.figure(),
            locs_3d, cfg.show_max_lateral_err, cfg.show_min_view_count)
        fig1.canvas.manager.set_window_title('Occurrences')
        fig2.canvas.manager.set_window_title('Histogram')
        fig3.canvas.manager.set_window_title('3D')

    # End

    if cfg.log_timing:
        total_time = time.time() - tic_total - user_interaction_time
        print(f'Total time: {1e3 * total_time:.3f} ms')

    if cfg.show_graphs:
        plt.show()

    if locs_3d is not None:
        print("SMLFM data processed successfully!")

    else:
        print("SMLFM data processing failed.")

    print("Formatting the locs3D csv output to a suitable format for downstream processing...")
    folder_name = f'{cfg.csv_file.name}'
    folder_path = os.path.join(root_dir, 'data', '3D_fitting_results', folder_name)
    locs3d_path = os.path.join(folder_path, 'locs3D.csv')

    raw_arr = pd.read_csv(locs3d_path, header=None).values
    n_data = raw_arr.shape[0]
    fm_arr = np.concatenate([raw_arr[:,-1].reshape(-1,1),
                            np.zeros((n_data, 1)),
                            (raw_arr[:,0:3].reshape(-1,3))*1000,
                            np.full((n_data, 3), np.nan),
                            raw_arr[:,-2].reshape(-1,1),
                            np.zeros((n_data,1))],
                            axis = 1) 
    
    columns = ['frame num', 'molecule num', 'x (nm)', 'y (nm)', 'z (nm)',
            'x fid-corrected (nm)', 'y fid-corrected (nm)',
            'z fid-corrected (nm)', 'photons detected',
            'mean background photons']
    
    fm_df = pd.DataFrame(fm_arr, columns=columns)
    
    for col_idx in [0,1,9]:
            fm_df[columns[col_idx]] = fm_df[columns[col_idx]].astype(np.int64)




    destination_dir = os.path.join(root_dir, 'data', 'formatted_3Dlocs')
    os.makedirs(destination_dir, exist_ok=True)

    destination_file_name = folder_name.replace('_locs2D.csv', '_locs3D_formatted.csv')
    destination_path = os.path.join(destination_dir, destination_file_name)

    fm_df.to_csv(destination_path, index=False)
    
    num3Dlocs = locs_3d.shape[0]    

    print(f"Formatting successful! '{destination_file_name}' saved to the 'data/formatted_3Dlocs' directory.\n")

    return (destination_path, num3Dlocs, [cfg.mla_rotation, cfg.mla_offset])




def batchFittingAxialLocs(root_dir, config_name=None):
    input_dir = os.path.join(root_dir, 'data', '2D_locs_csv')
    cfg = load_config(root_dir, config_name)
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            print(f"\nProcessing file: {file}")
            csv_path = os.path.join(input_dir, file)
            fittingAxialLocs(csv_path, root_dir, cfg=cfg)

    





def load_config_opt(root_dir, config_name = None):
    if config_name:
        config_path = os.path.join(root_dir, 'configs', config_name)
        if os.path.exists(config_path):
            with open(config_path, 'rt', encoding='utf-8') as file:
                cfg_dump = file.read()
                cfg = smlfm.Config.from_json(cfg_dump)
            print(f"Running iteration for the following config {config_path} ...")
            return cfg

        print(f"Error saving editted config file")

    print(f"stopping the run")
    return None





def fittingAxialLocs_opt(csv_file_path, root_dir, config_name=None, cfg=None):

    tic_total = time.time()
    user_interaction_time = 0
    timestamp = datetime.now()

    if cfg is None:
        try: 
            cfg = load_config_opt(root_dir, config_name)
        except Exception as e:
            print(f"Error loading configuration: {e}.")

    cfg.csv_file = Path(csv_file_path)

    if not cfg.csv_file.exists():
        print(f'ERROR: The CSV file "{cfg.csv_file}" does not exist')
        return None, None

    lfm = smlfm.FourierMicroscope(
        cfg.num_aperture, cfg.mla_lens_pitch,
        cfg.focal_length_mla, cfg.focal_length_obj_lens,
        cfg.focal_length_tube_lens, cfg.focal_length_fourier_lens,
        cfg.pixel_size_camera, cfg.ref_idx_immersion,
        cfg.ref_idx_medium)

    # 2. Read localisation file

    tic = time.time()

    csv = smlfm.LocalisationFile(cfg.csv_file, cfg.csv_format)
    csv.read()

    # print(f'Loaded {csv.data.shape[0]} localisations from'
    #       f' {np.unique(csv.data[:, 0]).shape[0]} unique frames')

    if cfg.show_graphs and cfg.show_all_debug_graphs:
        fig = smlfm.graphs.draw_locs_csv(plt.figure(), csv.data[:, 1:3])
        fig.canvas.manager.set_window_title('Raw localisations')

    csv.scale_peakfit_data(lfm.pixel_size_sample)
    locs_2d_csv = csv.data.copy()

    locs_2d_csv[:, 1] -= locs_2d_csv[:, 1].mean()
    locs_2d_csv[:, 2] -= locs_2d_csv[:, 2].mean()

    # if cfg.log_timing:
    #     print(f'Loading {repr(cfg.csv_file.name)} took {1e3 * (time.time() - tic):.3f} ms')

    # 3. Prepare MLA and rotate it to match the CSV data

    mla = smlfm.MicroLensArray(
        cfg.mla_type, cfg.focal_length_mla, cfg.mla_lens_pitch,
        cfg.mla_optic_size, np.array(cfg.mla_centre))

    if cfg.show_graphs and cfg.show_all_debug_graphs:
        fig = smlfm.graphs.draw_mla(plt.figure(), mla.lens_centres, mla.centre)
        fig.canvas.manager.set_window_title('Micro-lens array centres')

    mla.rotate_lattice(np.deg2rad(cfg.mla_rotation))
    mla.offset_lattice(np.array(cfg.mla_offset) / lfm.mla_to_xy_scale)

    if cfg.show_graphs and cfg.show_all_debug_graphs:
        fig = smlfm.graphs.draw_mla(plt.figure(), mla.lens_centres, mla.centre)
        fig.canvas.manager.set_window_title('Micro-lens array centres rotated')

    # 4. Map localisations to lenses

    tic = time.time()

    lfl = smlfm.Localisations(locs_2d_csv)
    lfl.assign_to_lenses(mla, lfm)

    # if cfg.log_timing:
    #     print(f'Mapping points to lenses took {1e3 * (time.time() - tic):.3f} ms')

    if cfg.show_graphs and cfg.show_mla_alignment_graph:
        fig = smlfm.graphs.draw_locs(
            plt.figure(),
            xy=lfl.locs_2d[:, 3:5],
            lens_idx=lfl.locs_2d[:, 12],
            lens_centres=(mla.lens_centres - mla.centre) * lfm.mla_to_xy_scale,
            mla_centre=np.array([0.0, 0.0]))
        fig.canvas.manager.set_window_title('Localisations with lens centers')

        if cfg.confirm_mla_alignment:
            tic = time.time()
            print('\nCheck on the figure that the lenses are well aligned with the'
                  ' data. Then close the window(s) to continue.')
            print('If the alignment is incorrect, adjust MLA rotation and/or offset in the configuration,'
                  ' and run this application again.')
            plt.show()
            while True:
                data = input('\nAre the lens centres aligned with the data? [Y/n]: ')
                data = 'y' if data == '' else data[0].casefold()
                if data not in ('y', 'n'):
                    print('Not an appropriate choice.')
                else:
                    if data == 'n':
                        return None, None
                    break
            print('')
            user_interaction_time = time.time() - tic

    # 5. Filter localisations and set alpha model

    tic = time.time()

    if cfg.filter_lenses:
        lfl.filter_lenses(mla, lfm)
    if cfg.filter_rhos is not None:
        lfl.filter_rhos(cfg.filter_rhos)
    if cfg.filter_spot_sizes is not None:
        lfl.filter_spot_sizes(cfg.filter_spot_sizes)
    if cfg.filter_photons is not None:
        lfl.filter_photons(cfg.filter_photons)

    lfl.init_alpha_uv(cfg.alpha_model, lfm, worker_count=cfg.max_workers)

    # if cfg.log_timing:
    #     print(f'Filtering and setting alpha model took {1e3 * (time.time() - tic):.3f} ms')

    # 6. Find system aberrations

    tic = time.time()

    fit_params_cor = dataclasses.replace(
        cfg.fit_params_aberration,
        frame_min=(cfg.fit_params_aberration.frame_min
                   if cfg.fit_params_aberration.frame_min > 0
                   else lfl.min_frame),
        frame_max=(cfg.fit_params_aberration.frame_max
                   if cfg.fit_params_aberration.frame_max > 0
                   else min(1000, lfl.max_frame)),
    )

    # print(f'Fitting frames'
    #       f' {fit_params_cor.frame_min}-{fit_params_cor.frame_max}'
    #       f' for aberration correction...')
    _, fit_data = smlfm.Fitting.light_field_fit(
        lfl.filtered_locs_2d, lfm.rho_scaling, fit_params_cor,
        worker_count=cfg.max_workers)

    correction = smlfm.Fitting.calculate_view_error(
        lfl.filtered_locs_2d, lfm.rho_scaling, fit_data, cfg.aberration_params)

    lfl.correct_xy(correction)

    if cfg.show_graphs and cfg.show_all_debug_graphs:
        fig = smlfm.graphs.draw_locs(
            plt.figure(),
            xy=lfl.corrected_locs_2d[:, 3:5],
            lens_idx=lfl.corrected_locs_2d[:, 12],
            lens_centres=(mla.lens_centres - mla.centre) * lfm.mla_to_xy_scale,
            mla_centre=np.array([0.0, 0.0]))
        fig.canvas.manager.set_window_title('Corrected localisations')

    # if cfg.log_timing:
    #     print(f'Aberration correction took {1e3 * (time.time() - tic):.3f} ms')

    # 7. Fit full data set on corrected localisations

    tic = time.time()

    fit_params_all = dataclasses.replace(
        cfg.fit_params_full,
        frame_min=(cfg.fit_params_full.frame_min
                   if cfg.fit_params_full.frame_min > 0
                   else lfl.min_frame),
        frame_max=(cfg.fit_params_full.frame_max
                   if cfg.fit_params_full.frame_max > 0
                   else lfl.max_frame),
    )

    # print(f'Fitting frames'
    #       f' {fit_params_all.frame_min}-{fit_params_all.frame_max}...')
    locs_3d, _ = smlfm.Fitting.light_field_fit(
        lfl.corrected_locs_2d, lfm.rho_scaling, fit_params_all,
        worker_count=cfg.max_workers)

  

    if locs_3d is not None:
        print("3D fitting iteration completed successfully!")
        print(f'Number frames used for fitting: {np.unique(locs_3d[:, 7]).shape[0]}, num2Dlocs used for fitting: {int(np.sum(locs_3d[:, 5]))}, num3Dlocs fitted: {locs_3d.shape[0]}\n')


    else:
        print("SMLFM data processing failed.")

    

    num3Dlocs = locs_3d.shape[0]    

    return (None, num3Dlocs, [cfg.mla_rotation, cfg.mla_offset])