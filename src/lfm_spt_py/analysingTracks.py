import subprocess
import os

# def matlab_trackingDataAnalysis(root_dir):
#     matlab_cmd = f"matlab -batch \"cd('{os.path.join(root_dir, 'src').replace(os.sep, '/')}'); AnalyzeTrackingData_withDirection_master\""
    
#     # Capture output and error streams
#     result = subprocess.run(matlab_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

#     print("Output:")
#     print(result.stdout)
    
#     print("Errors:")
#     print(result.stderr)



import subprocess
import os

def matlab_trackingDataAnalysis(root_dir):
    matlab_cmd = f"matlab -batch \"cd('{os.path.join(root_dir, 'src').replace(os.sep, '/')}'); AnalyzeTrackingData_withDirection_master\""
    
    # Use Popen to allow live output printing
    process = subprocess.Popen(matlab_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Continuously print the output as it's generated
    for line in process.stdout:
        print(line, end="")  # Print standard output live

    # Also print any errors if they occur
    for line in process.stderr:
        print(line, end="")  # Print error output live

    # Wait for the process to finish and get the return code
    process.wait()
