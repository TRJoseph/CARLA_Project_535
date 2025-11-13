# CARLA_535_Project
This repository is the host of a project utilizing CARLA and our TOKEN setup. By: Thomas and Yussef

# 1. Setup
To get setup with this project, first you must install [CARLA](https://github.com/carla-simulator/carla/releases/tag/0.9.16/)

Next, clone this repository if you have not done so already.

# 2. Create Python Virtual Environment
First make sure you are in your project directory. 
## Check Python Version
Run: `python --version`
## Create Virtual Environment
This project uses python version 3.12, but should work with later releases. 
Run: `py -{version} -m venv .venv` where version is 3.13 or 3.12, etc.
## Activate Virtual Environment
On Windows: 
`.venv\Scripts\activate.bat`
On macOS/Linux:
`source .venv/bin/activate`
# 3. Install All Project Dependencies
Run: `pip install -r requirements.txt`

Proceed to verify the installation by running: `pip list` and cross reference the requirements.txt file.

# 4. Add the CARLA install folder to python environment variables.
Add the `carla` and `carla/agents` as part of a new user variable (Absolute path to these directories). I called mine PYTHONPATH.
