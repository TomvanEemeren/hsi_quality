# Hyperspectral Image Quality

# Getting Started

## Step 1
Assuming that you have setup an SSH key, clone this repository to your workspace:
```
git clone git@github.com:TomvanEemeren/hsi_quality.git
```
Navigate to the newly created directory:
```
cd hsi_quality
```

## Step 2
Setup a virtual environment ***or*** a conda environment.
### Setting up a virtual environment
Create the virtual environment:
```
python3 -m venv .venv
```
Activate the virtual environment:
```
source .venv/bin/activate
```
Install dependencies in the virtual environment:
```
pip install -r requirements.txt
```

### Setting up a conda environment
Create the conda environment:
```
conda env create -f environment.yml
```
Activate the conda environment:
```
conda activate hsi312
```

## Step 3
Run the `preprocess_data.py` file to download and preprocess the hyperspectral image dataset:
```
python preprocess_data.py
```
Note that you must be connected to the local internet network at the NTNU to access the server and download the data!