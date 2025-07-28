import subprocess
import os

def matlab_trackingDataAnalysis(root_dir):
    matlab_cmd = f"matlab -batch \"cd('{os.path.join(root_dir, 'src').replace(os.sep, '/')}'); AnalyzeTrackingData_withDirection_master\""

    process = subprocess.Popen(matlab_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # continuously printing output as it's generated
    for line in process.stdout:
        print(line, end="")

    for line in process.stderr:
        print(line, end="")  

    process.wait()
