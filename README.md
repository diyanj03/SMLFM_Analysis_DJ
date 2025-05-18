# SMLFM_Analysis_DJ

This pipeline quantifies the biophysical parameters of a nuclear transcription factor from live-cell single molecule light field microscopy (SMLFM) data. It first performs 2D fitting of raw images, followed by 3D fitting to obtain axial positions. Single particle trajectories are then obtained and are classified as DNA-bound or freely diffusing based on track-wise biophysical parameters. It  then computes global parameters of the TF such as DNA-bound fraction, diffusion coefficient, and TF-DNA association/dissociation rates.


**Sections for this guide:**

1.  **Python Environment Setup**: Downloading and installing system-level tools & softwares, configuring paths, creating a virtual environment, and installing Python dependencies.

2.  **Running the Analysis (using Jupyter Notebook)**: Running the pipeline step-by-step, inputs to provide, and handling configurations.

*Note: This guide assumes Windows OS for environment setup. For a Unix-like system (macOS/Linux/WSL), you can set up the environment in an analgous way to the instructions that will follow.* <br>

***Recommended**: use a code editor like [VSCode](https://code.visualstudio.com/download) with Python and Jupyter extensions installed to run the analysis using the Jupyter Notebook.*


## 1. Python Environment Setup

### 1.1. Download Project Repository

1. Go to `https://github.com/diyanj03/SMLFM_Analysis_DJ` on github.
2. Click `<> Code` → `Download ZIP`.
3. Extract the folder to your desired location. Your workspace should initially have:
    * `src/` (source code)
    * `configs/` (configuration files)
    * `tests/` (tested dataset and results)
    * `README.md` (this file)
    * `requirements.txt` (Python dependencies list)

*Note your extracted path, e.g.: `C:/Users/username/Documents/SMLFM_Analysis_DJ-main`*

---

### 1.2. Install Core Software

Install the following software. Tested versions are indicated in parentheses.


**Python (3.13):**
- Download from [python.org](https://www.python.org/downloads/)
- During install: **tick "Add Python X.XX to PATH"**
- Verify installation on terminal: `py --version` and `pip --version`
<br><br>

**JDK (21.0.6.7-hotspot):**
- Download from [Adoptium](https://adoptium.net/temurin/releases/)
- Extract to a suitable location, e.g.: `C:\Program Files\Eclipse Adoptium\jdk-21.0.6.7-hotspot`.
- Set Environment Variables:
    1. Windows search → **"Edit the system environment variables"** → **Environment Variables...**
    2. Under *System variables* → **New...**  
        - **Variable name:** `JAVA_HOME`  
        - **Variable value:** `C:\your\path\to\Eclipse Adoptium\jdk-21.0.6.7-hotspot`
        - Click **OK**
    3. Still in *System variables* → Select **Path** → **Edit...** → Select **New** → enter `%JAVA_HOME%\bin`
    4. Click **OK** on all windows to apply changes.
- Verify installation on terminal: `java -version`. 
<br><br>

**Apache Maven (3.9.9):**

- Download the "Binary zip archive" from [maven.apache.org](https://maven.apache.org/download.cgi)
- Extract to a suitable location, e.g.: `C:\Program Files\apache-maven-3.9.9`
- Set Environment Variables:
    * Under *System variables*, create `M2_HOME` and set value to `C:\your\path\to\apache-maven-3.9.9` 
    * Add `%M2_HOME%\bin` to your system `Path`.
- Verify installation on terminal: `mvn -version`.
<br><br>

**Fiji ImageJ2:**
- Download version 2.16.0/1.54p with GDSC-SMLM2 PeakFit plugin from this [onedrive_link](https://1drv.ms/u/c/4b95c84a5c5eb7e0/EW_BqP-NbElKhGFYc3vsQFcBVx6-iJrRClgTlfgZeFLxwA?e=akZzXw)
- Extract to a location outside **'C:/Program Files/'**, e.g.: `C:\Users\username\Applications\Fiji.app` and note this path.
<br><br>

**Microsoft C++ Build Tools (Windows only):**
- Download from [Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
- Run the installer (`vs_buildtools.exe`):
    * Select "Desktop development with C++" workload.
    * Ensure these components are checked: "MSVC v143 - VS 2022 C++ x64/x86 build tools (Latest)", "Windows SDK" (latest appropriate), "C++ CMake tools for Windows".
    * Click **"Install"**.
- (Optional) Add MSVC binary path to your system `Path` for easier access by `pip`. 
<br><br>

**MATLAB (R2024b):**
- Download from [MathWorks](https://www.mathworks.com/downloads/).
- Add MATLAB's `bin` directory (e.g.: `C:\Program Files\MATLAB\R2024b\bin`) to your system `Path`.
- Verify installation on terminal: `matlab -batch "disp(['MATLAB version: ' version]); exit"`.

---

### 1.3. Create & Activate Python Virtual Environment


1.  Run the following command on terminal to navigate to the `SMLFM_Analysis_DJ-main` project directory:
    ```
    cd your/path/to/SMLFM_Analysis_DJ-main
    ```
    *(Replace `your/path/to/SMLFM_Analysis_DJ-main` with the actual path to the folder noted in step 1.1).*

2. Create virtual environment, once inside project directory:
    ```
    py -m venv .venv
    ```

3. Activate venv by running the following command: 
    * CMD: `.venv\Scripts\activate`
    * PowerShell: `.venv\Scripts\Activate.ps1`
    
    (Once active, your command prompt will then be prefixed with `(.venv)`)

---
### 1.4. Install Python Dependencies

With the active virtual environment inside the project directory, run the following:
```
pip install -r requirements.txt
```
This installs all python packages required for the analysis pipeline.
<br><br>

---
## 2. Implementing the Analysis Pipeline (in main.ipynb)

### Open `main.ipynb` Jupyter Notebook from the `src` folder and select the `.venv` kernel.



### 2.1. Defining root directory and importing packages (cell 1)
- **Purpose:** to import py packages and define the project root directory.
- **Inputs**:
    - **`root_dir`**: absolute path of your SMLFM_Analysis_DJ-main folder as noted in step 1.1. 
- Run this cell evertyime you restart the kernel.
<br><br>

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

<br>

### 2.3. 3D Fitting using PySMLFM (Cell 3)
- **Purpose:** To obtain 3D localisations (XYZ coordinates) across all frames from 2D localisation csv files. Uses the PySMLFM package from https://github.com/Photometrics/PySMLFM.

- **Inputs:** 
    - **`runLoop`**
        - set to `False` to run 3D fitting for a single 2D localisation file.
        - set to `True` to batch proccess all 2D localisation files within the `data/2D_locs_csv` folder.


    - **`locs2D_path`:** path to the 2D localisation file to be analysed if runLoop = False.
        - set to `return_locs2D_path` if you ran 2D fitting with a single dataset.
        - set to absolute path to locs2D csv file if the **cell2_locs2D_path** returned from cell 2 is lost. e.g.: `r"C:/path/to/SMLFM_Analysis_DJ-main/data/2D_locs_csv/you_file_locs2D.csv"` 

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

<br>

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
    
<br>

## 2.5. Plotting 3D Localisations (Cell 5)
- **Purpose:** To produce and display scatter plots of 3D localisations across all frames for qualitative assessment of a dataset.

- **Inputs:**
    - **`fm_locs3D_path`:** path to the formatted 3D localisation file to be plotted.
        - set to `return_locs3D_path` to automatically plot the 3D localisation file returned from the most recent 3D fitting run.
        - set to absolute path to locs2D csv file if the **return_locs2D_path** returned from cell 3 is lost/undefined. e.g.: `r"C:/path/to/SMLFM_Analysis_DJ-main/data/formatted_3Dlocs/your_locs3D_formatted.csv"` 

- **Note**: After plotting 3D locs for a given dataset, apply the cropping using Cell 6 (see section 2.6) before moving on to the next dataset.

- **Output:** 2D and 3D scatter plots of all 3D localisations fitted across all frames for the given dataset. Must run next cell after this plot.

<br>

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

<br>

## 2.7. Single molecule tracking (Cell 7):
- **Purpose:** Obtaining single molecule trajectories from cropped 3D localisation data. Uses code from https://github.com/wb104/trajectory-analysis.

- **Note**: This runs the tracking code for ALL the 3D localisation csv files within the 'data/cropped_3Dlocs' folder. Hence, it should be ran once you have processed and cropped all your 3D localisation datasets for a given sample/condition.

- **Output:**  csv files of single particle trajectories for each dataset saved in the **'data/tracks/'** folder.

<br>

## 2.8. Analysing and classifying trajectories (Cell 8) :
- **Purpose:** To compute biophysical parameters of each single particle trajectory, classify each trajectory into DNA-bound or freely diffusing, and output global parameters of the TF such as diffusion constant, association and dissociate rates. Further information about this analysis can be found from the [supplement methods of Basu et al., 2023](https://static-content.springer.com/esm/art%3A10.1038%2Fs41594-023-01095-4/MediaObjects/41594_2023_1095_MOESM1_ESM.pdf).

- **Parameters:**
    - To adjust parameters, open the matlab script from `src/AnalyzeTrackingData_withDirection_master.m` and modify params at the very start of the script.
    - Key parameters: `minNumPoints`, `minNumPointsLongTraj`, `numMSDpoints` (details provided in the script).

- **Important**: This cell runs the analysis by inputting all of the tracks csv files from the **'data/tracks/'** folder. Hence, ensure that all datasets within the **'data/tracks/'**  correspond to the SAME sample/condition!!

- **Output:** An output folder named by the data & time of analysis saved to the **'results**` folder. Can rename the output folder to the sample type. Output folder contains loads of results.

<br>

## 2.9. Plotting results for a sample (Cells 9)
- **Purpose:** To produce jitter plots showing per FOV distribution of specific TF properties (% bound to DNA, diffusion constant when DNA-bound, diffusion constant when unbound) of a given sample.  

- **Inputs:** 
    - **`results_dir`**: absolute path to the specific results folder of a given sample. 
    - **`sample_name`**: name of sample. e.g. 'SOX2_mESCs'
    - **`destination_dir`** absolute path to destination folder to save jitter plots.

- **Output:** Saves .png plots to the provided destination directory. 


## 2.10. Plotting results comparing samples (Cell 10)
- **Purpose:** Similar to Cell 9 but plots distributions of different samples in the same plot and performs statistical analysis to test for significance.

- **Args:** 
    - **`results_dirs`**: a list of absolute paths to the results directories (each corresponding to a sample type) to be compared.
    - **`sample_names`**: a list of the names of the sample types in the same order as the list of result directories.  
    - **`destination_dir`** absolute path to destination folder to save jitter plots and stat results.

- **Output:** Saves .pdf plots to the provided destination directory. Also saves .txt files containing output of statistics (mean, std_dev, etc) and significance test results (type of test, p_value). 

<br>

## 2.11. Saving data and results (Cell 11)
- **Purpose:** To copy the whole `data/` and `results/` directories to a destination folder.

- **Inputs:** 
    - **`destination_dir`**: absolute path to the destination folder to save the data from the **'data/'** and **'results/'** folders.

- Can now empty the workspace in SMLFM_Analysis_DJ for another round of analysis with another sample type or condition.

---
 
### *The pipeline has been tested using a sample SOX2_mESC SMLFM dataset from the 'tests' folder.*   










