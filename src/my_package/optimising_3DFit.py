import numpy as np
import skopt
from skopt import gp_minimize
from skopt.space import Real
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import copy
import os
from my_package import fitting_3D  
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'my_package')))

def optimise_MLA_alignment(root_dir, search_space, locs2D_path, base_configName, n_calls=200, n_initial_points=20):
    """
    Args:
    - workspace directory
    - search_space: list of tuples defining search bounds. [(anglemin, anglemax), (xmin, xmax), (ymin, ymax)]
    - locs2D_path: path to 2D locs data that will be ran for optimising 3D fitting params. 
    - base_configPath: base config file that will be used for optimisation (with changes only to mla angle and offset)

    Returns:
    - output of gp_minimize() function for the given dataset.
    """
    base_configPath = os.path.join(root_dir, 'configs', base_configName)
    if os.path.exists(base_configPath):
        pass
    else:
        base_configPath = os.path.join(root_dir, 'configs', 'default_configs', 'fitting3D_configDefault_ref')
        print(f'Input config file not found, instead using default config as reference from {base_configPath}')
        

    def run3D_fit(angle, x, y):
        with open(base_configPath, 'r') as file:
            config = json.load(file)
        new_config = copy.deepcopy(config)
        new_config['mla_rotation'] = float(angle)
        new_config['mla_offset'] = [float(x), float(y)]

        dir_path, file_name = os.path.split(base_configPath)
        base_name, ext = os.path.splitext(file_name)
        
        new_name = f'{base_name}_{angle:.2f}_{x:.2f}_{y:.2f}{ext}'
        new_filepath = os.path.join(dir_path, new_name)
        
        with open(new_filepath, 'w') as new_file:
            json.dump(new_config, new_file, indent=4)

        cfg_name = new_name
        numlocs3D = fitting_3D.fittingAxialLocs_opt(locs2D_path, root_dir, cfg_name)[1]

        return numlocs3D

    # Wrapper for skopt
    def run3D_fit_wrapper(params):
        angle, x, y = params
        return -run3D_fit(angle, x, y)  # Return negative because gp_minimize minimizes

    search_space1 = [
        Real(search_space[0][0], search_space[0][1], name='angle'),
        Real(search_space[1][0], search_space[1][1], name='x'),
        Real(search_space[2][0], search_space[2][1], name='y'),
    ]

    result_gp = gp_minimize(run3D_fit_wrapper, search_space1, n_calls=n_calls, n_initial_points=n_initial_points, random_state=42)



    # Deleting the config files generated
    config_dir = os.path.join(root_dir, 'configs')
    base_name = os.path.splitext(os.path.split(base_configPath)[1])[0]
    for fname in os.listdir(config_dir):
        if fname.startswith(f"{base_name}_") and fname.endswith(".json"):
            full_path = os.path.join(config_dir, fname)
            os.remove(full_path)



    align_opt_dir = os.path.join(root_dir, 'data', '3D_fitting_alignmentOpt')
    os.makedirs(align_opt_dir, exist_ok=True)

    locs2D_name = os.path.split(locs2D_path)[1]
    wo_ext_namr = os.path.splitext(locs2D_name)[0]
    results_dir = os.path.join(align_opt_dir, wo_ext_namr)
    os.makedirs(results_dir, exist_ok=True)



    results = []
    for i, params in enumerate(result_gp.x_iters):
        angle, x, y = params
        num3Dlocs = -result_gp.func_vals[i]
        results.append((angle, x, y, num3Dlocs))

    results.sort(key=lambda x: x[3], reverse=True)

    angle_min, angle_max = search_space[0]
    x_min, x_max = search_space[1]
    y_min, y_max = search_space[2]

    angle_diff = 0.1 * (angle_max - angle_min)
    x_diff = 0.1 * (x_max - x_min)
    y_diff = 0.1 * (y_max - y_min)

    top_hits = [results[0]]
    for result in results[1:]:
        angle, x, y, num3Dlocs = result
        for top_hit in top_hits:
            top_angle, top_x, top_y, _ = top_hit
            if (abs(angle - top_angle) >= angle_diff or
                abs(x - top_x) >= x_diff or
                abs(y - top_y) >= y_diff):
                top_hits.append(result)
                break
        if len(top_hits) >= 5:
            break


    num3Dlocs_values = [num3Dlocs for _, _, _, num3Dlocs in results]
    avg_num3Dlocs = np.mean(num3Dlocs_values)
    std_num3Dlocs = np.std(num3Dlocs_values)

    output_file_path = os.path.join(results_dir, 'results_alignmentOpt.txt')
    with open(output_file_path, 'w') as file:
        file.write("Top hits (local maxima):\n")
        file.write("Angle, X, Y, Num3DLocs\n")
        
        for angle, x, y, num3Dlocs in top_hits:
            file.write(f"{angle:.2f}, {x:.2f}, {y:.2f}, {num3Dlocs:.2f}\n")
        
        file.write("\nSearch space used for each parameter:\n")
        file.write(f"Angle: {angle_min} to {angle_max}\n")
        file.write(f"X: {x_min} to {x_max}\n")
        file.write(f"Y: {y_min} to {y_max}\n")
        
       
        file.write("\nGP optimisation settings:\n")
        file.write(f"n_calls: {n_calls}\n")
        file.write(f"n_initial_points: {n_initial_points}\n")
        file.write(f"random_state: 42\n")
        
       
        file.write("\nStatistics:\n")
        file.write(f"Average num3DLocs: {avg_num3Dlocs:.2f}\n")
        file.write(f"Standard deviation of num3DLocs: {std_num3Dlocs:.2f}\n")
        
       
        file.write("\nAll parameter combinations searched and their associated num3DLocs (sorted by num3DLocs):\n")
        file.write("Angle, X, Y, Num3DLocs\n")
        
       
        results.sort(key=lambda x: x[3])  # Sort by num3DLocs in ascending order
        for angle, x, y, num3Dlocs in results:
            file.write(f"{angle:.2f}, {x:.2f}, {y:.2f}, {num3Dlocs:.2f}\n")


    print(f'GP complete! Top alignment parameters and other results saved in {results_dir}.' 
          '\nNext steps: run the 3D fitting with the top parameters and visually assess alignment.')

   
    plt.figure(figsize=(8, 5))
    plt.plot(-np.array(result_gp.func_vals))
    plt.xlabel('Iteration')
    plt.ylabel('Number of 3D Localisations')
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(results_dir, 'search_summary.png')
    plt.savefig(save_path)
    plt.show()

    
    params = result_gp.x_iters
    angles = [p[0] for p in params]
    x_vals = [p[1] for p in params]
    y_vals = [p[2] for p in params]
    scores = result_gp.func_vals
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(angles, x_vals, y_vals, c=scores, cmap='viridis', marker='o')
    ax.set_xlabel('Angle')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    cbar = plt.colorbar(sc)
    cbar.set_label('Objective Function Value')
    save_path2 = os.path.join(results_dir, '3D_paramSpace.png')
    plt.savefig(save_path2)
    plt.show()
  
    return results_dir
