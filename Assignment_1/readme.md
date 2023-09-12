# My Project

## Description

Getting started with setting up a Python environment using Conda or pip, install dependencies, and modify the `PYTHONPATH`.
_PLEASE NOTE: SHOULD ANY ERROR BE THROWN THAT A MODULE CANNOT BE FOUND, ADD IT TO THE *requirements.txt* FILE AND FOLLOW  THE INSTALL INSTRUCTION TO INSTALL IT._

## Prerequisites

- Python 3.9.9 and above
- Conda (for Conda environment setup)
- pip (for pip environment setup)

## Setup

### Option 1: Using Conda

1. **Clone the Repository**

    ```bash
    git clone https://github.com/smelliecat/CMSC421-FALL23-Students.git
    cd my_project
    ```

2. **Create a Conda Environment**

    ```bash
    conda create --name my_project_env python=3.x
    ```

    Replace `3.x` with your desired Python version.

3. **Activate the Environment**

    ```bash
    conda activate my_project_env
    ```

4. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

### Option 2: Using pip

1. **Clone the Repository**

    ```bash
    git clone https://github.com/smelliecat/CMSC421-FALL23-Students.git
    cd my_project
    ```

2. **Create a Virtual Environment**

    ```bash
    python3 -m venv my_project_env
    ```

    Or, if you're on Windows:

    ```bash
    py -m venv my_project_env
    ```

3. **Activate the Environment**

    - On macOS and Linux:

        ```bash
        source my_project_env/bin/activate
        ```

    - On Windows:

        ```bash
        .\my_project_env\Scripts\activate
        ```

4. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

## Add to PYTHONPATH

After setting up your environment, you'll need to add the `Assignment_1` directory to your `PYTHONPATH`.

- On Unix/Linux/Mac:

    ```bash
    export PYTHONPATH=$PYTHONPATH:/path/to/Assignment_1
    ```

    To make this change permanent, add the above line to your shell's startup script (e.g., `.bashrc`, `.zshrc`).

- On Windows:

    ```bash
    set PYTHONPATH=%PYTHONPATH%;C:\path\to\Assignment_1
    ```

    To make it permanent, add it to the System Environment Variables through the Control Panel.

Replace `/path/to/Assignment_1` or `C:\path\to\Assignment_1` with the actual path to the `Assignment_1` directory.
