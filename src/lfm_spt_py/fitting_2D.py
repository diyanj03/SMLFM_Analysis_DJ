import imagej
import numpy as np
import pandas as pd
import scyjava as sj
import os
import tifffile as tiff
import xarray as xr
import logging
import sys
import time
import json
import textwrap
import time

def check_tif_compatibility(file_path):
    """
    Args:
        file_path (str): The path to the TIFF file.
    Returns:
        numpy.ndarray: The TIFF data as a NumPy array.
    """
    tic = time.time()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.lower().endswith(('.tif', '.tiff')):
        raise ValueError("Invalid file type. File must be a .tif or .tiff file.")

    logging.info('Checking compaitibility of input .tif file...')
    try:
        np_tif = tiff.imread(file_path)
        if len(np_tif.shape) > 3:
            with tiff.TiffFile(file_path) as tif:
                actual_num_pages = len(tif.pages)
                if actual_num_pages <= 0:
                    raise ValueError("TIFF file has no pages.")
                
                logging.info(f'Warning: Input dataset {os.path.basename(file_path)} has greater than three dimensions with a shape of {np_tif.shape}.'
                       '\nCorrecting the shape by concatenenating the .tif file pages along the non time dimension (interrupt kernel to stop if you wish to)...')
                first_page_shape = tif.pages[0].shape
                dtype = tif.pages[0].dtype
                valid_data = np.zeros((actual_num_pages,) + first_page_shape, dtype=dtype)
                for i in range(actual_num_pages):
                    valid_data[i] = tif.pages[i].asarray()
                logging.info(f'Image successfully corrected into an np array with shape {valid_data.shape}. Took {(time.time() - tic):.3f} seconds.')
                return valid_data, file_path
        
        elif len(np_tif.shape) < 3:
            logging.error("Dimensions too small for peak fit. Please check you images.")
            sys.exit()
            return None

        else:
            with tiff.TiffFile(file_path) as tif:
                actual_num_pages = len(tif.pages)
                if np_tif.shape[0] == actual_num_pages:
                    logging.info(f'No issue with tifffile.imread so image file has compatible dimensions and shape.\nImage sucessfully loaded as an np array with shape {np_tif.shape}. Took {(time.time() - tic):.3f} seconds.')
                    return np_tif, file_path
                
                else:
                    print('Number of pages do not match the number of timeframes assigned by .tif file metadata.\nHence, creating a new object by iterating through each successful page...')
                    first_page_shape = tif.pages[0].shape
                    dtype = tif.pages[0].dtype
                    valid_data = np.zeros((actual_num_pages,) + first_page_shape, dtype=dtype)
                    for i in range(actual_num_pages):
                        valid_data[i] = tif.pages[i].asarray()
                    logging.info(f'Image successfully loaded as an np array with shape {valid_data.shape}. Took {(time.time() - tic):.3f} seconds.')
                    return valid_data, file_path

    except ValueError as ve:
        raise ValueError(f"Error processing TIFF file: {ve}")
    except FileNotFoundError as fnfe:
        raise FileNotFoundError(f"File not found: {fnfe}")
    except MemoryError as me:
        raise MemoryError(f'Memory error. Invest in a better computer.: {me}')
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


def _load_image(source_filepath, np_image, my_dims=('t', 'row', 'col')):
    """Loads image files using tifffile and xarray
    Args: source_filepath, np loaded image, dimensions
    Returns: xarray loaded image and basename of the filepath to set the image name in downstream functions  
    """
    tic = time.time()
    try:
        logging.info(f"Now loading the image as an xarray and labelling dimensions as {my_dims}...")
        xr_timeseries = xr.DataArray(np_image, dims=my_dims)
        logging.info(f"Image successfully loaded as an xarray (took {(time.time() - tic):.3f} seconds). Image dimensions: {xr_timeseries.shape}.\n")
        datasetName = os.path.splitext(os.path.basename(source_filepath))[0]

        return xr_timeseries, datasetName

    except FileNotFoundError as fnf_error:
        logging.error(f"File error: {fnf_error}")
        return None

    except ValueError as ve:
        logging.error(f"Value error: {ve}")
        return None

    except Exception as e:
        logging.error(f"Failed to load image as an xarray: {e}")
        return None


def check_tifDir_compatibility(file_dir):
    """
    Args:
        file_dir (str): The path to the directory containing different image stack files of the same fov.
    Returns:
        A list of numpy arrays, each corresponding to a np loaded stack in order (sorted numerically). 
    """
    tic = time.time()
        # valid_extensions = ('.tif', '.tiff', '.czi')
        # stacks_list = sorted([file for file in os.listdir(source_dir) if file.endswith(valid_extensions)])
        # if not stacks_list:
        #     raise FileNotFoundError(f"No valid image files found in {source_dir}")
        # else:
        #     logging.info(f"Loading image files as xarrays and labelling dimensions as {my_dims}...")

        # first_image_data = tiff.imread(os.path.join(source_dir, stacks_list[0]))
        # if first_image_data.ndim != len(my_dims):
        #     raise ValueError(f"Dimension mismatch for file {stacks_list[0]}: Image shape {first_image_data.shape} doesn't match dims {my_dims}.")

        # img_np_list = [tiff.imread(os.path.join(source_dir, file)) for file in stacks_list]

    if not os.path.exists(file_dir):
        raise FileNotFoundError(f"Directory not found: {file_dir}")
    
    valid_extensions = ('.tif', '.tiff', '.czi')
    
    stacksName_list = [file for file in os.listdir(file_dir) if file.endswith(valid_extensions)]
    if not stacksName_list:
        raise FileNotFoundError(f"No files with extensions {valid_extensions} found in {file_dir}")
    stacksName_list.sort()

    filepath_list = [os.path.join(file_dir, file) for file in stacksName_list]

    logging.info('Verifying compatibility of the first .tif file in the directory...')
    try:
        np_tif_stack1 = tiff.imread(filepath_list[0])
        if len(np_tif_stack1.shape) > 3:
            logging.info(f'\nWarning: Stacks within input dataset have greater than three dimensions. E.g. {stacksName_list[0]} has a shape of {np_tif_stack1.shape}.'
                       '\nCorrecting the shape of each stack by concatenenating the .tif file pages along the non time dimension (interrupt kernel to stop if you wish to)...')
            np_img_list = []
            for file in filepath_list:
                with tiff.TiffFile(file) as tif:
                    actual_num_pages = len(tif.pages)
                    if actual_num_pages <= 0:
                        raise ValueError(f"TIFF file {file_dir} has no pages.")
                    
                    logging.info(f'Processing dataset {os.path.basename(file)}...')
                    first_page_shape = tif.pages[0].shape
                    dtype = tif.pages[0].dtype
                    valid_data = np.zeros((actual_num_pages,) + first_page_shape, dtype=dtype)
                    
                    for i in range(actual_num_pages):
                        valid_data[i] = tif.pages[i].asarray()
                    
                    np_img_list.append(valid_data)

            logging.info(f'Images successfully corrected into np arrays with correct dimensions. Took {(time.time() - tic):.3f} seconds.')

            return np_img_list, file_dir
       
        elif len(np_tif_stack1.shape) < 3:
            logging.error("Dimensions too small for peak fit. Please check your image files.")
            sys.exit()
            return None
          
        else:
            with tiff.TiffFile(filepath_list[0]) as tif_stack1:
                actual_num_pages = len(tif_stack1.pages)
                
                if np_tif_stack1.shape[0] == actual_num_pages:
                    logging.info(f'No issue with tifffile.imread so image files have compatible dimensions and shape.')
                    np_img_list = [np_tif_stack1,]
                    for file in filepath_list[1:]:
                        img = tiff.imread(file)
                        np_img_list.append(img)
                    logging.info(f'Image stacks sucessfully loaded as np arrays. Took {(time.time() - tic):.3f} seconds.')
                    return np_img_list, file_dir
                
                else:
                    print('Number of pages do not match the number of timeframes assigned by .tif file metadata.\nHence, creating a new object for each stack by iterating through each successful page...')
                    np_img_list = []
                    for file in filepath_list:
                        with tiff.TiffFile(file) as tif:
                            actual_num_pages = len(tif.pages)
                            if actual_num_pages <= 0:
                                raise ValueError(f"TIFF file {file_dir} has no pages.")
                            
                            logging.info(f'Processing dataset {os.path.basename(file)}...')
                            first_page_shape = tif.pages[0].shape
                            dtype = tif.pages[0].dtype
                            valid_data = np.zeros((actual_num_pages,) + first_page_shape, dtype=dtype)
                            
                            for i in range(actual_num_pages):
                                valid_data[i] = tif.pages[i].asarray()
                            
                            np_img_list.append(valid_data)

                    logging.info(f'Images successfully corrected into np arrays with correct dimensions. Took {(time.time() - tic):.3f} seconds.')
                    return np_img_list, file_dir

    except ValueError as ve:
        raise ValueError(f"Error processing TIFF file: {ve}")
    except FileNotFoundError as fnfe:
        raise FileNotFoundError(f"File not found: {fnfe}")
    except MemoryError as me:
        raise MemoryError(f'Memory error. Invest in a better computer.: {me}')
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")

def _load_and_concat_images(source_dir, img_np_list, my_dims=('t', 'row', 'col')):
    """Loads and concatenates image files using tifffile and xarray.
    Args: source_directory of image stacks, list of numpy arrays of each image in order, dimensions.
    Returns: xarray concatenated image, and name for the image.    
    """
    tic = time.time()
    try:
        logging.info(f"Now loading images as xarrays and labelling dimensions as {my_dims}...")

        image_arrays = [xr.DataArray(file, dims=my_dims) for file in img_np_list]
        logging.info(f'Images successfully loaded as xarrays. Took {(time.time() - tic):.3f} seconds.\n')

        logging.info(f'Concatenating the {len(image_arrays)} image stacks along the time axis...')
        xr_timeseries = xr.concat(image_arrays, dim='t') if len(img_np_list) > 1 else image_arrays[0]
        logging.info(f"Image concatenation successful (took {(time.time() - tic):.3f} seconds). Final dataset has {len(xr_timeseries.t)} frames.\n")
        datasetName = os.path.basename(source_dir)
        return xr_timeseries, datasetName

    except FileNotFoundError as fnf_error:
        logging.error(f"File error: {fnf_error}")
        return None
    except ValueError as ve:
        logging.error(f"Dimension mismatch error: {ve}")
        return None
    except Exception as e:
        logging.error(f"Failed to load and concatenate images: {e}")
        return None
    
def _save_concatenated_image(xarr_timeseries, datasetName, root_directory):
    """Saves the concatenated image to a file."""
    tic = time.time()
    try:
        output_file_name = f'{datasetName}.tif'
        destination_dir = os.path.join(root_directory, 'data', 'raw_image_datasets')
        os.makedirs(destination_dir, exist_ok=True)
        destination_path = os.path.join(destination_dir, output_file_name)
        
        logging.info(f"Saving the loaded image as {datasetName}.tif in '/data/Raw_Images'...")
        tiff.imwrite(destination_path, xarr_timeseries.values)
        logging.info(f"{datasetName}.tif successfully saved in '/data/Raw_Images' (took {(time.time() - tic):.3f} seconds).\n")
    except Exception as e:
        logging.error(f"Failed to save image: {e}\n")

def _init_imagej(fiji_directory, ram='8'):
    """Initialises ImageJ via PyImageJ."""
    tic = time.time()
    try:
        logging.info("Initialising PyImageJ interface...")
        sj.config.add_options(f'-Xmx{ram}G')
        ij = imagej.init(fiji_directory, mode='interactive')
        version = ij.getVersion()
        logging.info(f"PyImageJ successfully initialised with ImageJ2 version {version} (took {(time.time() - tic):.3f} seconds).\n")

        if str(version).split('/')[1] == "Inactive":
            logging.error("PyImageJ initialised but is inactive. Restart the kernel and try again")
            sys.exit("Stopping execution due to inactive IJ.")
            return None 

        return ij

    except Exception as e:
        logging.error(f"Failed to initialise ImageJ: {e}")
        return None

def _convert_to_java_image(ij, xarr_timeseries, datasetName):
    """Converts xarray image to Java object for ImageJ."""
    try:
        logging.info('Converting the xarray image to a Java object via IJ gateway...')
        java_timeseries = ij.py.to_java(xarr_timeseries)
        java_timeseries.setName(datasetName)
        logging.info('Conversion successful! Here is the summary of your Java object to be ran on Peak Fit:')
        return java_timeseries
    except Exception as e:
        logging.error(f"Failed to convert image to Java object: {e}")
        return None

def _image_info(image):
    """Provides information about the image."""
    try:
        dims_shape_dict = {image.dims[x]: image.shape[x] for x in range(len(image.dims))} if hasattr(image, 'dims') else None
        name = image.getName() if hasattr(image, 'getName') else image.getTitle() if hasattr(image, 'getTitle') else 'N/A'
        
        logging.info(
            f"  - Name: {name}\n"
            f"  - Dimensions & Shape: {dims_shape_dict}")
    except Exception as e:
        logging.warning(f"Error gathering image information: {e}")

def _run_peak_fit(
                  ij, java_timeseries, datasetName, root_directory,
                  run_background_subtraction, rb_radius, run_gaussian_blur, offset_value, sigma, calibration, exposure_time, psf_model, spot_filter_type,
                  spot_filter, smoothing, spot_filter2, smoothing2, search_width, border_width, fitting_width, fit_solver, fail_limit, 
                  pass_rate, neighbour_height, residuals_threshold, duplicate_distance, shift_factor, signal_strength,
                  min_photons, min_width_factor, max_width_factor, precision, camera_bias, gain, read_noise, 
                  psf_parameter_1, precision_method, relative_threshold, absolute_threshold,
                  parameter_relative_threshold, parameter_absolute_threshold, max_iterations, lambdaa,
                  image_scale, image_size, image_pixel_size
                  ):
    """Runs the Peak Fit macro in ImageJ."""
    tic = time.time()
    try:
        ij.ui().show(java_timeseries)
        locs2d_xls_directory = os.path.join(root_directory, 'data', '2D_locs_xls')
        locs2d_csv_directory = os.path.join(root_directory, 'data', '2D_locs_csv')
        os.makedirs(locs2d_xls_directory, exist_ok=True)
        os.makedirs(locs2d_csv_directory, exist_ok=True)
        java_output_dir = locs2d_xls_directory.replace('\\', '/') + '/'
        java_csv_out = locs2d_csv_directory.replace('\\', '/') + '/'
        
        if run_gaussian_blur:
            macro_code = ""
            if run_background_subtraction:
                macro_code += f'selectWindow("{datasetName}")\n'
                macro_code += f'run("Subtract Background...", "rolling={rb_radius} disable stack")'
            
            macro_code += textwrap.dedent(f"""                    
            run("Z Project...", "projection=[Average Intensity]");
            imageCalculator("Subtract create stack", "{datasetName}","AVG_{datasetName}");
            run("Add...", "value={offset_value} stack");
            run("Gaussian Blur...", "sigma={sigma} stack");
            selectWindow("{datasetName}")
            close();
            selectWindow("AVG_{datasetName}")
            close();
            selectWindow("Result of {datasetName}")

            run("Peak Fit", 
                "template=None camera_type=EMCCD calibration={calibration} exposure_time={exposure_time} " + 
                "psf=[{psf_model}] spot_filter_type={spot_filter_type} spot_filter={spot_filter} " +
                "smoothing={smoothing} spot_filter2={spot_filter2} smoothing2={smoothing2} " +
                "search_width={search_width} border_width={border_width} fitting_width={fitting_width} fit_solver=[{fit_solver}] " +
                "fail_limit={fail_limit} pass_rate={pass_rate} neighbour_height={neighbour_height} " +
                "residuals_threshold={residuals_threshold} duplicate_distance={duplicate_distance} shift_factor={shift_factor} signal_strength={signal_strength} " +
                "min_photons={min_photons} min_width_factor={min_width_factor} max_width_factor={max_width_factor} precision={precision} " +
                "show_results_table image=[Localisations (width=precision)] results_format=Text " +
                "results_directory=[{java_output_dir}] " +
                "save_to_memory camera_bias={camera_bias} gain={gain} read_noise={read_noise} psf_parameter_1={psf_parameter_1} " +
                "relative_threshold={relative_threshold} absolute_threshold={absolute_threshold} parameter_relative_threshold={parameter_relative_threshold} " +
                "parameter_absolute_threshold={parameter_absolute_threshold} max_iterations={max_iterations} lambda={lambdaa} " +
                "duplicate_distance_absolute precision_method={precision_method} table_distance_unit=[pixel (px)] " +
                "table_intensity_unit=photon table_angle_unit=[degree (°)] table_show_precision " +
                "table_precision=0 equalised image_size_mode=[Image size] image_scale={image_scale} image_size={image_size} " +
                "image_pixel_size={image_pixel_size} lut=Fire file_distance_unit=[pixel (px)] " +
                "file_intensity_unit=photon file_angle_unit=[degree (°)] file_show_precision");
                
                selectWindow("Result of {datasetName}")
                close();
                selectWindow("Result of {datasetName} (LVM LSE) SuperRes");
                close();

                run("Text File... ", "open=[{java_output_dir}Result of {datasetName}.results.xls]");

                saveAs("Text", "{java_csv_out}{datasetName}_locs2D.csv");

                run("Close All");
                """)
        
        else: 
            macro_code = ""
            if run_background_subtraction:
                macro_code += f'selectWindow("{datasetName}")\n'
                macro_code += f'run("Subtract Background...", "rolling={rb_radius} disable stack")'
            
            macro_code += textwrap.dedent(f"""                    
            selectWindow("{datasetName}")

            run("Peak Fit", 
                "template=None camera_type=EMCCD calibration={calibration} exposure_time={exposure_time} " + 
                "psf=[{psf_model}] spot_filter_type={spot_filter_type} spot_filter={spot_filter} " +
                "smoothing={smoothing} spot_filter2={spot_filter2} smoothing2={smoothing2} " +
                "search_width={search_width} border_width={border_width} fitting_width={fitting_width} fit_solver=[{fit_solver}] " +
                "fail_limit={fail_limit} pass_rate={pass_rate} neighbour_height={neighbour_height} " +
                "residuals_threshold={residuals_threshold} duplicate_distance={duplicate_distance} shift_factor={shift_factor} signal_strength={signal_strength} " +
                "min_photons={min_photons} min_width_factor={min_width_factor} max_width_factor={max_width_factor} precision={precision} " +
                "show_results_table image=[Localisations (width=precision)] results_format=Text " +
                "results_directory=[{java_output_dir}] " +
                "save_to_memory camera_bias={camera_bias} gain={gain} read_noise={read_noise} psf_parameter_1={psf_parameter_1} " +
                "relative_threshold={relative_threshold} absolute_threshold={absolute_threshold} parameter_relative_threshold={parameter_relative_threshold} " +
                "parameter_absolute_threshold={parameter_absolute_threshold} max_iterations={max_iterations} lambda={lambdaa} " +
                "duplicate_distance_absolute precision_method={precision_method} table_distance_unit=[pixel (px)] " +
                "table_intensity_unit=photon table_angle_unit=[degree (°)] table_show_precision " +
                "table_precision=0 equalised image_size_mode=[Image size] image_scale={image_scale} image_size={image_size} " +
                "image_pixel_size={image_pixel_size} lut=Fire file_distance_unit=[pixel (px)] " +
                "file_intensity_unit=photon file_angle_unit=[degree (°)] file_show_precision");
                
                selectWindow("{datasetName}")
                close();
                selectWindow("{datasetName} (LVM LSE) SuperRes");
                close();

                run("Text File... ", "open=[{java_output_dir}{datasetName}.results.xls]");

                saveAs("Text", "{java_csv_out}{datasetName}_locs2D.csv");

                run("Close All");
                """)
        

        pf_time = f'{((time.time() - tic))/60:.3f} mins'
        logging.info('\nExecuting GDSC SMLM2 peak fit function (caution - requires patience!)...')
        ij.py.run_macro(macro_code)
        
        csv_file_path = f'{java_csv_out}{datasetName}_locs2D.csv'
        

        logging.info(f"Peak fit successful (took {pf_time}) :)")
        logging.info(f"2D localisations saved as '{datasetName}_locs2D.csv' in the 'data/2D_locs_csv' directory.")
        df = pd.read_csv(csv_file_path, skiprows=8) 
        locs = len(df)
        logging.info(f"Number of 2D localisations fitted: {locs}.\n")
        return csv_file_path, macro_code

    except Exception as e:
        logging.error(f"Peak fit failed: {e}")
        return None
       

def gdsc_peakFit(source_input, runLoop, fiji_directory, root_directory, config_name=None):
    """
    main execution function
    Args:
        source_input (str): Path to image directory or file.
        fiji_directory (str): Path to Fiji installation.
        root_directory (str): Base directory for output.
        config_name (str): Configuration file name (e.g., 'fitting_2D.json').

    Returns:
        str: Path to the resulting CSV file, or None on failure.
    """
    logging.basicConfig(level=logging.INFO, format='%(message)s')

     # loading .json configs ===
    config_data = {}
    pf_params_config = {}
    try:
        if config_name:
        # fitting_2D.json is in the same directory as this script
            config_path = os.path.join(root_directory, 'configs', config_name)
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                logging.info(f"Loaded 2D fitting configurations from: {config_path}. Modify if needed.\n") # Simple log message
                pre_processing = config_data.get('pre-processing', {})
                camera_settings = config_data.get('camera_settings', {})
                maxima_identification = config_data.get('maxima_identification', {})
                gaussian_fitting = config_data.get('gaussian_fitting', {})
                fit_solver_settings = config_data.get('fit_solver_settings', {})
                peak_filtering = config_data.get('peak_filtering', {})
                other_settings = config_data.get('other_less_relevant_settings', {})

            else:
                logging.info(f"Configuration file not found at {config_path}. Falling back to default!!")
                config_path = None

        if not config_name or not config_path:
                config_path = os.path.join(root_directory, 'configs', 'default_configs', 'fitting2D_configDefault.json')
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                logging.info(f"Loaded 2D fitting configurations from: {config_path}.\n")
                pre_processing = config_data.get('pre-processing', {})
                camera_settings = config_data.get('camera_settings', {})
                maxima_identification = config_data.get('maxima_identification', {})
                gaussian_fitting = config_data.get('gaussian_fitting', {})
                fit_solver_settings = config_data.get('fit_solver_settings', {})
                peak_filtering = config_data.get('peak_filtering', {})
                other_settings = config_data.get('other_less_relevant_settings', {})

    except Exception as e:
        logging.error(f"Error loading configuration: {e}.")
        return None

    # Override defaults with JSON values if they exist
    saveRawImage = config_data.get('saveRawImage')
    # Ensure my_dims from JSON is a tuple if it exists, otherwise use default tuple
    my_dims = tuple(config_data.get('my_dims'))
    ram = config_data.get('ram')

    # Override peak fit parameters
    run_background_subtraction = pre_processing.get('run_background_subtraction')
    rb_radius = pre_processing.get('rb_radius')
    run_gaussian_blur = pre_processing.get('run_gaussian_blur')
    sigma = pre_processing.get('sigma')
    offset_value = pre_processing.get('offset_value')

    calibration = camera_settings.get('calibration')
    exposure_time = camera_settings.get('exposure_time')
    camera_bias = camera_settings.get('camera_bias')
    gain = camera_settings.get('gain')
    read_noise = camera_settings.get('read_noise')

    spot_filter_type = maxima_identification.get('spot_filter_type')
    spot_filter = maxima_identification.get('spot_filter')
    smoothing = maxima_identification.get('smoothing')
    spot_filter2 = maxima_identification.get('spot_filter2')
    smoothing2 = maxima_identification.get('smoothing2')
    search_width = maxima_identification.get('search_width')
    border_width = maxima_identification.get('border_width')
    fitting_width = maxima_identification.get('fitting_width')

    psf_model = gaussian_fitting.get('psf_model')
    psf_parameter_1 = gaussian_fitting.get('psf_parameter_1')
    fit_solver = gaussian_fitting.get('fit_solver')
    fail_limit = gaussian_fitting.get('fail_limit')
    pass_rate = gaussian_fitting.get('pass_rate')

    relative_threshold = fit_solver_settings.get('relative_threshold')
    absolute_threshold = fit_solver_settings.get('absolute_threshold')
    parameter_relative_threshold = fit_solver_settings.get('parameter_relative_threshold')
    parameter_absolute_threshold = fit_solver_settings.get('parameter_absolute_threshold')
    max_iterations = fit_solver_settings.get('max_iterations')
    lambdaa = fit_solver_settings.get('lambda')
    precision_method = fit_solver_settings.get('precision_method')

    shift_factor = peak_filtering.get('shift_factor')
    signal_strength = peak_filtering.get('signal_strength')
    min_photons = peak_filtering.get('min_photons')
    min_width_factor = peak_filtering.get('min_width_factor')
    max_width_factor = peak_filtering.get('max_width_factor')
    precision = peak_filtering.get('precision')

    neighbour_height = other_settings.get('neighbour_height')
    residuals_threshold = other_settings.get('residuals_threshold')
    duplicate_distance = other_settings.get('duplicate_distance')
    image_scale = other_settings.get('image_scale')
    image_size = other_settings.get('image_size')
    image_pixel_size = other_settings.get('image_pixel_size')

    if runLoop == True:
        try:
            ij = _init_imagej(fiji_directory, ram)
            if ij is None:
                return None
        
            if os.path.isdir(source_input):
                for source_name in sorted(os.listdir(source_input)):
                    input_path = os.path.join(source_input, source_name)
                    if os.path.isfile(input_path):
                        logging.info(f"\nPROCESSING FILE: {input_path} ...")
                        np_image, filepath = check_tif_compatibility(input_path)
                        xr_timeseries, datasetName = _load_image(filepath, np_image, my_dims)

                    elif os.path.isdir(input_path):
                        logging.info(f"\nPROCESSING DIRECTORY: {input_path} ...")
                        np_img_list, dirPath = check_tifDir_compatibility(input_path)
                        xr_timeseries, datasetName = _load_and_concat_images(dirPath, np_img_list, my_dims)

                    else:
                        raise ValueError(f"{input_path} is not a valid file or path")

                    if xr_timeseries is None or datasetName is None:
                        return None

                    if saveRawImage:
                        _save_concatenated_image(xr_timeseries, datasetName, root_directory)

                    
                    jv_timeseries = _convert_to_java_image(ij, xr_timeseries, datasetName)
                    if jv_timeseries is None:
                        return None

                    _image_info(jv_timeseries)

                    _run_peak_fit(ij, jv_timeseries, datasetName, root_directory, run_background_subtraction, rb_radius, run_gaussian_blur, offset_value, sigma, calibration, exposure_time, psf_model, spot_filter_type,
                                                                                spot_filter, smoothing, spot_filter2, smoothing2, search_width, border_width, fitting_width, fit_solver, fail_limit, 
                                                                                pass_rate, neighbour_height, residuals_threshold, duplicate_distance, shift_factor, signal_strength,
                                                                                min_photons, min_width_factor, max_width_factor, precision, camera_bias, gain, read_noise, 
                                                                                psf_parameter_1, precision_method, relative_threshold, absolute_threshold,
                                                                                parameter_relative_threshold, parameter_absolute_threshold, max_iterations, lambdaa,
                                                                                image_scale, image_size, image_pixel_size)
            else:
                logging.error(f"cannot loop over a file. ensure runLoop is False or source_input is a directory")
                return None
        except Exception as e:
            logging.error(f"An error occurred in peak fitting pipeline: {e}")
            return None
        finally:
            if 'ij' in locals() and ij is not None:
                ij.dispose()
    
    
    else:
        try:
            if os.path.isfile(source_input):
                np_image, filepath = check_tif_compatibility(source_input)
                xr_timeseries, datasetName = _load_image(filepath, np_image, my_dims)

            elif os.path.isdir(source_input):
                np_img_list, dirPath = check_tifDir_compatibility(source_input)
                xr_timeseries, datasetName = _load_and_concat_images(dirPath, np_img_list, my_dims)

            else:
                raise ValueError(f"{source_input} is not a valid file or path")

            if xr_timeseries is None or datasetName is None:
                return None

            if saveRawImage:
                _save_concatenated_image(xr_timeseries, datasetName, root_directory)

            ij = _init_imagej(fiji_directory, ram)
            if ij is None:
                return None

            jv_timeseries = _convert_to_java_image(ij, xr_timeseries, datasetName)
            if jv_timeseries is None:
                return None

            _image_info(jv_timeseries)

            csv_file_path, macro_code = _run_peak_fit(ij, jv_timeseries, datasetName, root_directory, run_background_subtraction, rb_radius, run_gaussian_blur, offset_value, sigma, calibration, exposure_time, psf_model, spot_filter_type,
                                                                        spot_filter, smoothing, spot_filter2, smoothing2, search_width, border_width, fitting_width, fit_solver, fail_limit, 
                                                                        pass_rate, neighbour_height, residuals_threshold, duplicate_distance, shift_factor, signal_strength,
                                                                        min_photons, min_width_factor, max_width_factor, precision, camera_bias, gain, read_noise, 
                                                                        psf_parameter_1, precision_method, relative_threshold, absolute_threshold,
                                                                        parameter_relative_threshold, parameter_absolute_threshold, max_iterations, lambdaa,
                                                                        image_scale, image_size, image_pixel_size)
            return csv_file_path, macro_code

        except Exception as e:
            logging.error(f"An error occurred in peak fitting pipeline: {e}")
            return None
        finally:
            if 'ij' in locals() and ij is not None:
                ij.dispose()


        
        
