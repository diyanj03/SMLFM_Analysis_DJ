# lfmSPTanalysis

This pipeline quantifies the biophysical parameters of a nuclear transcription factor from live-cell single molecule light field microscopy (SMLFM) data. 
It first performs 2D fitting of raw timestack images, followed by 3D fitting to obtain axial positions. Single particle spatiotemporal trajectories (SPTs) are then obtained and are classified into two populations based on track-wise biophysical parameters. It then computes global parameters of the TF such as DNA-bound fraction, diffusion coefficient, and association rate to DNA. 
Residence time analysis can also be performed using tracks obtained from time-lapse experiments to compute dissocation rate of the TF from DNA.



## 1. Environment Setup

### 1.1. Download the Repository

- Download/clone from GitHub: [diyanj03/lfmSPTanalysis](https://github.com/diyanj03/lfmSPTanalysis)
- Your workspace should contain:
    - `src/`, `configs/`, `tests/`, `README.md`,`requirements.txt`
---

### 1.2. Install Core Software

#### Python (3.13):
- Download from [python.org](https://www.python.org/downloads/)
- During install: ✅ *Add to PATH*  
- Confirm installation on terminal:
  ```bash
  py --version
  pip --version

#### Java JDK 21 (Temurin):
- Download from [adoptium.net](https://adoptium.net/temurin/releases/)
- Extract to a suitable location.

Add bin directory to system `Path`:
- Windows:
    1. Add `JAVA_HOME` as a system variable to the path where you extracted the JDK (e.g. `C:\Program Files\Eclipse Adoptium\jdk-21.x.x`)
    2. Add `%JAVA_HOME%\bin` to your system `Path` variable
 
- macOS/Linux:
    - Add the following to your shell config file (e.g. `~/.bashrc`, `~/.zshrc`): 
    
    ```bash
    export JAVA_HOME=/path/to/your/jdk
    export PATH=$JAVA_HOME/bin:$PATH
    ```


#### Apache Maven (3.9.9):

- Download the "Binary zip archive" from [maven.apache.org](https://maven.apache.org/download.cgi)
- Extract and add bin directory to your system `Path` like you did for JDK 
- Verify installation:
    ```bash
    mvn -version
    ```


#### Fiji ImageJ2:
- Download version 2.16.0/1.54p with GDSC-SMLM2 PeakFit plugin from this [onedrive_link](https://1drv.ms/u/c/4b95c84a5c5eb7e0/EW_BqP-NbElKhGFYc3vsQFcBVx6-iJrRClgTlfgZeFLxwA?e=akZzXw)
- Extract outside `Program Files`
- Install pip package:
    ```
    pip install pyimagej
    ```
- Confirm installation:
    ```
    py -c 'import imagej; print("ImageJ initialised successfully." if imagej.init("your/path/to/Fiji.app") else "ImageJ initialisation failed.")'
    ```

#### C++ Build Tools (Windows only):
- Download from [visualstudio.microsoft.com](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Install: "Desktop dev with C++" + MSVC, Windows SDK, CMake
<br><br>

**MATLAB (R2024b):**
- Download from [mathworks.com](https://www.mathworks.com/downloads/).
- Add `MATLAB/bin` to system `PATH`.
- Verify installation:
    ```
    matlab -batch "disp(['MATLAB version: ' version]); exit"
    ```
- Install the following add-ons:
    -  Curve Fitting Toolbox
    -  Image Processing Toolbox
    -  Statistics and Machine Learning Toolbox

---

### 1.3. Set Up Virtual Environment


1.  Navigate to the `lfmSPTanalysis-main` project directory and create a virtual py environment:
    ```
    cd your/path/to/lfmSPTanalysis-main
    py -m venv .venv
    ```

2. Activate venv by running the following command: 
    * CMD: `.venv\Scripts\activate`
    * PowerShell: `.venv\Scripts\Activate.ps1`
    * Bash: `source .venv/bin/activate`

    
    (Once active, your command prompt will then be prefixed with `(.venv)`)

---
### 1.4. Install Python Dependencies

With an active virtual environment, install dependencies:
```
pip install -r requirements.txt
```
---
### 1.5. Link virtual environment to Jupyter

The `src/main.ipynb` in your Jupyter editor (recommended: [VSCode](https://code.visualstudio.com/download)).

Manually select the `.venv` kernel as python interpreter:
- Windows: `.venv\Scripts\python.exe
- macOS/Linux: `.venv/bin/python`

<br>

## 2. Running the Analysis

Launch `main.ipynb` on Jupyter. Follow the notebook cells step-by-step, adjusting configs/inputs as detailed below. 


### 2.1. Defining root directory and importing packages (cell 1)
- **Purpose:** to import py packages and define the project root directory.
- **Inputs**:
    - **`root_dir`**: absolute path of your lfmSPTanalysis-main folder as noted in step 1.1. 
- Run this cell evertyime you restart the kernel.
---

### 2.2. 2D Fitting using ImageJ's PeakFit (Cell 2)
- **Purpose:** To obtain 2D localisations (XY coordinates) across all frames from raw .tif timestacks.

- **Inputs:**  
    - **`source_path`**: can be set to either of the following:
        1. absolute path to a single .tif file. e.g.: `r"C:/path/to/file.tif"`
        2. absolute path to a directory containing multiple .tif files of the same field of view (FOV) - to be combined as one dataset prior to 2D fitting. e.g.: `r"C:/path/to/single_FOV_folder"`
        3. absolute path to a directory containing multiple .tif files of different FOVs (to be all analysed separately). e.g.: `r"C:/path/to/multiple_FOVs_folder"`
    
    - **`runLoop`**:
        - set as `False` if source_path is either option 1 or 2 (anlaysing a single FOV/dataset). 
        - set as `True` if source_path is option 3 (batch processing multiple datasets/FOVs).

    - **`config_name`**: name of the 2D fitting config file to be used for analysis from the `configs` folder. e.g.: `"fitting2D_config.json"`

    - **`fiji_path`**: absolute path to Fiji app installed from step 1.2. e.g.: `r"C:\Users\username\Applications\Fiji.app"`

- **Configuration for 2D Fitting:**
    - Adjust 2D fitting settings by opening and modifying the `fitting2D_config.json` file under the `configs` folder.
    - Key settings: 
        - `ram` - memory allocated to Fiji. (ensure this is less than the memory capacity of your device, set as "6" if unsure).
        - `saveRawImage` - to save raw timestack in workspace - boolean.
        - `my_dims` - change if dimensions of your image are not [time, row, column].
        - `peak_fit_params` - parameters for GDSC SMLM2 PeakFit function.
      
    - Multiple copies of the config file can be created in the configs folder with custom names. But, ensure config_name is set accordingly when running cell 2.


- **Output:** 2D localisation csv file for each dataset will be saved in the `data` folder, specifically within `data/2D_locs_csv/`

---

### 2.3. 3D Fitting using PySMLFM (Cell 3)
- **Purpose:** To obtain 3D localisations (XYZ coordinates) across all frames from 2D localisation csv files. Uses the PySMLFM package from https://github.com/Photometrics/PySMLFM.

- **Inputs:** 
    - **`runLoop`**
        - set to `False` to run 3D fitting for a single 2D localisation file.
        - set to `True` to batch proccess all 2D localisation files within the `data/2D_locs_csv` folder.


    - **`locs2D_path`:** path to the 2D localisation file to be analysed if runLoop = False.
        - set to `return_locs2D_path` if you ran 2D fitting with a single dataset.
        - set to absolute path to locs2D csv file if the **cell2_locs2D_path** returned from cell 2 is lost. e.g.: `r"C:/path/to/lfmSPTanalysis-main/data/2D_locs_csv/you_file_locs2D.csv"` 

    - **`cfg_name`**: name of the 3D fitting config file to be used for analysis from the `configs` folder. e.g.: `"fitting2D_config.json"`

- **Configuration for 3D Fitting:**
    - Adjust 2D fitting settings by opening and modifying the `fitting2D_config.json` file under the `configs` folder.
    - Key settings include: 
        - `show_graphs` - initially set to True to check MLA alignment of a dataset. set to False when batch proccessing datasets with the same alignment using runLoop.
        - `mla_rotation` and `mla_offset` - adjust to align data with the microlens array (MLA). 
    - Multiple copies of the config file can be created in the configs folder with custom names. But, ensure cfg_name is set accordingly when running cell 3.

- **Output:** 
    - Figures (alignment plot, 3D scatter plot) + config file + localisation csv saved to the 'data/3D_fitting_results/' folder. 
    - Formatted version of the 3D localisations csv file saved to the 'data/formatted_3Dlocs/' folder.

---

## 2.4. Optimising MLA alignment for 3D Fitting (Cell 4)
- **NOTE:** If 3D fitting worked well with a good MLA alignment (obtaining expected number of 3D localisations),  then skip this step and move on to section 2.5 (cell 5).
- **Purpose:** Despite qualitatively plausible MLA alignment, 3D fitting can fail. This cell uses skopt's gp_optimize to find mla_offset and mla_rotation values that optimise alignment to maximise 3D localisations. 

- **Inputs:** 
    - **`locs2D_path`**: absolute path to a 2D localisations file to be used for alignment optimisation.
    - **`base_cfg_filename`**: file name of the 3D fitting config that will be used as a base config file for optimisating the mla alignment.
        - `mla_rotation` and `mla_offset` parameters change during the search so doesn't matter what these are set as.
    - Search bounds:
        - `rotation_search`, `x_offset_search` and `y_offset_search` : tuple (min, max).
        - Set the boundaries to extreme parameter values that would still give a qualitatively plausible MLA alignment. 

    - **`n_calls`**: number of total searches during optimisation. 200 works well.
    - **`n_initial_points`**: number of initial random search prior to optimisation algorithm. 50 works well.

- **Output:** Text file of top combinations of `mla_rotation` and `mla_offset` saved in 'data/3D_fitting_alignmentOpt/'. Apply these optimal alignment parameters in the 3D fitting config file and re-run 3D fitting on cell 3. 
    
---

## 2.5. Plotting 3D Localisations (Cell 5)
- **Purpose:** To produce and display scatter plots of 3D localisations across all frames for qualitative assessment of a dataset.

- **Inputs:**
    - **`fm_locs3D_path`:** path to the formatted 3D localisation file to be plotted.
        - set to `return_locs3D_path` to automatically plot the 3D localisation file returned from the most recent 3D fitting run.
        - set to absolute path to locs2D csv file if the **return_locs2D_path** returned from cell 3 is lost/undefined. e.g.: `r"C:/path/to/lfmSPTanalysis-main/data/formatted_3Dlocs/your_locs3D_formatted.csv"` 

- **Note**: After plotting 3D locs for a given dataset, apply the cropping using Cell 6 (see section 2.6) before moving on to the next dataset.

- **Output:** 2D and 3D scatter plots of all 3D localisations fitted across all frames for the given dataset. Must run next cell after this plot.

---

## 2.6. Cropping 3D Localisations (Cell 6)
- **Purpose:**  To crop out 3D localisations that make up the coverslip and artefacts.

- **Inputs:**
    - **`xmin`, `xmax`, `ymin`, `ymax`, `zmin`, `zmax`**: spatial cropping bounds in microns. 
    - Set bounds based on visual inspection of plots from Cell 5. 
        - e.g.: `zmin = 2` to remove all data below 2μm in the z axis.
    - Set to `None` for axes not requiring cropping.
- **Important**: Run this cell, even if no cropping required (in which case, set all input bounds to **None**). 
- For unsatisfactory crops, can always re-run Cell 5 with uncropped data, then re-run this cell with adjusted bounds.

- **Outputs:**
    - A cropped 3D localisation csv file saved in the **'data/cropped_3Dlocs/**' folder.
    - Displays and saves scatter plots of the cropped data. 

---

## 2.7. Single molecule tracking (Cell 7):
- **Purpose:** Obtaining single molecule trajectories from cropped 3D localisation data. Uses code from https://github.com/wb104/trajectory-analysis.

- **Note**: This runs the tracking code for ALL the 3D localisation csv files within the 'data/cropped_3Dlocs' folder. Hence, it should be ran once you have processed and cropped all your 3D localisation datasets for a given sample/condition.

- **Output:**  csv files of single particle trajectories for each dataset saved in the **'data/tracks/'** folder.

---

## 2.8. Analysing and classifying trajectories (Cell 8) :
- **Purpose:** To compute biophysical parameters of each single particle trajectory, classify each trajectory into DNA-bound or freely diffusing, and output global parameters of the TF such as diffusion constant, association and dissociate rates. Further information about this analysis can be found from the [supplement methods of Basu et al., 2023](https://static-content.springer.com/esm/art%3A10.1038%2Fs41594-023-01095-4/MediaObjects/41594_2023_1095_MOESM1_ESM.pdf).

- **Parameters:**
    - To adjust parameters, open the matlab script from `src/AnalyzeTrackingData_withDirection_master.m` and modify params at the very start of the script.
    - Key parameters: `minNumPoints`, `minNumPointsLongTraj`, `numMSDpoints` (details provided in the script).

- **Important**: This cell runs the analysis by inputting all of the tracks csv files from the **'data/tracks/'** folder. Hence, ensure that all datasets within the **'data/tracks/'**  correspond to the SAME sample/condition!!

- **Output:** An output folder named by the data & time of analysis saved to the **'results**` folder. Can rename the output folder to the sample type. Output folder contains loads of results.

---

## 2.9. Plotting per FOV distributions and generating analysis summary for a sample (Cell 9)
- **Purpose:** To produce jitter plots showing per FOV distribution of specific TF properties (% bound to DNA, diffusion constant when DNA-bound, diffusion constant when unbound) of a given sample. Also generates a summary csv file of analysis properties like number of 2Dlocs, num3Dlocs, numTracks per FOV.  

- **Inputs:** 
    - **`results_dir`**: absolute path to the matlab output folder of a given sample. 
    - **`sample_name`**: name of sample. e.g. 'SOX2_mESCs'

- **Output:** Saves output to a folder named after 'sample_name' within the '../results/' folder. output includes a csv summary file, pdf plots and txt files noting the statistics.

---

## 2.10. Plotting per FOV distributions for multiple samples for comparison (Cell 10)
- **Purpose:** Similar to Cell 9 but plots distributions of different samples in the same plot and performs statistical analysis to test for significance.

- **Args:** 
    - **`results_dirs`**: a list of absolute paths to the matlab output directories (each corresponding to a sample type) to be compared.
    - **`summary_csvs`**: a list of absolute paths to the summary csv file generated from cell 9 (each corresponding to a sample type) to be compared.
    - **`sample_names`**: a list of the names of the sample types in the same order as the list of result directories/csv paths.

- **Output:** Saves .pdf plots to a folder named after 'sample_names' within a subfolder named after 'sample_names' in the results folder. Also saves .txt files containing output of statistics (mean, std_dev, etc) and significance test results (type of test, p_value). 
 


## 2.11. Residence Time Analysis (Cell 11)
- **Purpose:** To calculate residence time of the TF to DNA from timelapse imaging experiments. SPT length distributions of timelapse experiments with different time intervals but same camera integration time are fitted to an exponential decay to compute effective decay rates, k_eff. Linear fitting of the k_eff values for each timelapse experiment then allows calculation of dissociation rate and photobleaching rate. 
For more information refer to - Gebhardt et al., Nat Methods, 2013.

- **Inputs:** 
    - **`tracks_dict`**: a dictionary where each key is a total timelapse interval in seconds, and each respective value is the path to a folder containing only the tracks CSV files (generated by cell 7) corresponding to that specific timelapse experiment.
    - **`tau_int`**: a float corresponding to the exposure time. 
        - an incorrect input here only affects the photobleaching rate calculation.
    - **`sample_name`**: name of sample for which residence time analysis is carried out. for labelling purposes.

- **Output:** Saves pdf plots and a txt result file to a folder named after 'sample_name' within the results folder. Plots include the exponential decay fits to obtain k_eff estimates for each timelapse experiment and a linear fit of the k_eff values to obtain the dissociation rate, k_off, and photobleaching rate, k_b.

---
 
### *The pipeline has been tested using a 20ms SMLFM dataset of a HaloTag-tagged NANOG mutant in live mESCs. Raw data, configs and results for this can be found in the 'tests' folder.*   










