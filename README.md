# [BirdCLEF+ 2025](https://www.kaggle.com/competitions/birdclef-2025)

This project is designed to assist with bird sound analysis and classification. Below are the instructions to set up a virtual environment and install the required dependencies.

## Setting Up the Environment

You can use either `conda` or `micromamba` to create and manage the virtual environment.
### Setting Up the Environment

1. Create a new environment using either `conda` or `micromamba`:
    ```bash
    conda create -p ./.conda
    # or
    micromamba create -p ./.conda
    ```
2. Activate the environment:
    ```bash
    conda activate ./.conda
    # or
    micromamba activate ./.conda
    ```
3. Install the required packages:
    ```bash
    conda install ffmpeg conda
    conda install -c conda-forge librosa pydub
    # or
    micromamba install kaggle ffmpeg conda pytorch librosa
    ```

## Logging in to Kaggle and Downloading Data

1. Log in to your Kaggle account using your browser.
2. Navigate to your account settings and scroll down to the **API** section.
3. Click on **Create New API Token**. This will download a file named `kaggle.json`.
4. Place the `kaggle.json` file in the `.kaggle` directory in your home folder:
    ```bash
    mkdir -p ~/.kaggle
    mv /path/to/kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    ```
5. Download the competition data using the Kaggle CLI:
    ```bash
    kaggle competitions download -c birdclef-2025
    ```
6. Extract the downloaded files if necessary:
    ```bash
    unzip birdclef-2025.zip -d data/
    ```
    

Ensure the `kaggle` CLI is installed and configured before attempting to download the data.

## Notes

- Ensure you have `conda` or `micromamba` installed on your system before proceeding.
- Replace `python=3.9` with your preferred Python version if necessary.

You're now ready to start working on the BirdCLEF+ 2025 project!
