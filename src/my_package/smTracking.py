import os
import subprocess


def sm_tracking(root_dir, data_dir = None):
    src_dir = "src"
    if data_dir == None:
        data_dir = os.path.join(root_dir, 'data','cropped_3Dlocs')
    else:
        pass

    trajectory_analysis_path = os.path.join(root_dir,'src', "my_package", "trajectoryAnalysis.py")
    # constructing the command
    command = [
        "py",
        trajectory_analysis_path,
        data_dir,
        "-numDimensions", "3",
        "-maxJumpDistance", "800",
        "-minNumPositions", "2",
        "-saveTracks",
        "-savePositionsFramesIntensities"
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("trajectoryAnalysis.py executed successfully.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error executing trajectoryAnalysis.py: {e}")
        print("Standard Output:")
        print(e.stdout)
        print("Standard Error:")
        print(e.stderr)
    except FileNotFoundError:
        print(f"Error: Python or trajectoryAnalysis.py not found.")