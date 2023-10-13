
# Assignment Preparation and Extra Resources

Hello Students,

Before diving into the questions within the assignment folder, there are some additional resources and setup steps we'd like you to be aware of. Following these will ensure you have a smooth experience while attempting the questions.

## 📘 Extra Resources

    To provide a more profound understanding of some of the concepts covered in this assignment, we recommend going through the following resources:

    - **Backpropagation Tutorial**: If you're tackling deep learning or neural network questions, understanding backpropagation is essential. We found a detailed guide from PyImageSearch to help you grasp this concept. [Check out the tutorial here](https://pyimagesearch.com/2021/05/06/backpropagation-from-scratch-with-python/).

## 🧰 Prerequisites

- Python 3.9.9 and above
- Conda (for Conda environment setup)
- pip (for pip environment setup)

## ⚙️ Setup Before Starting

To ensure your local development environment is consistent and doesn't conflict with other Python projects, we highly recommend setting up a virtual environment. Here's how you can do it for both `pip` and `conda` and on different OS:

### Pip

#### Windows

    ```bash
    python -m venv myenv
    myenv\Scripts\activate
    ```

#### Mac

    ```bash
    python3 -m venv myenv
    source myenv/bin/activate
    ```

### Conda

#### Create a Conda Environment

**Windows & Mac**:

    ```bash
    conda create --name myenv python=3.x

    ```

Replace `3.x` with your desired Python version.

#### Activate the Environment

    ```bash
    conda activate my_project_env
    ```

Replace `myenv` with any name you prefer for your virtual environment.

### 📦 Installing Packages

Once you've activated your virtual environment, you should install the necessary Python packages provided in the requirements.txt file in the assignment folder. This ensures that you have all the necessary dependencies for the assignment.

**For pip users:**

    ```
    pip install -r requirements.txt
    ```

**For Conda users:**

    ```
    pip install -r requirements.txt
    ```
this is because `conda` does not directly support the `-r` flag with requirements.txt files.

### 💡 Tips

- **Commit Often**: If you're using version control like git, remember to commit your changes often to save your progress.
- **Ask for Help**: Should you face any challenges, don't hesitate to reach out to instructors or peers. We're here to help!

### 🚀 Conclusion

With the right resources and a properly set up environment, you're all set to tackle the assignment questions with confidence. Dive in and happy coding!

---
