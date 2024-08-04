import subprocess

def run_script(filename):
    print(f"Starting {filename}...")
    try:
        result = subprocess.run(['python', filename], check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {filename}:")
        print(e.output)
    print(f"{filename} terminated succesfully")


if __name__ == '__main__':
    scripts = ['01_preprocessing.py', '02_model.py', '03_validation.py']

    for script in scripts:
        run_script(script)
