import subprocess
import sys
import os

def sm_tracking(root_dir, data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(root_dir, 'data','cropped_3Dlocs')

    trajectory_analysis_path = os.path.join(root_dir, 'src', 'lfm_spt_py', 'trajectoryAnalysis.py')
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
        # Run subprocess with stdout and stderr piped
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # Print output line-by-line as it comes
        for line in process.stdout:
            print(line, end='')  # already has newline

        process.wait()  # wait for process to finish

        if process.returncode == 0:
            print("trajectoryAnalysis.py executed successfully.")
        else:
            print(f"trajectoryAnalysis.py exited with return code {process.returncode}.")

    except FileNotFoundError:
        print("Error: Python executable or trajectoryAnalysis.py not found.")
