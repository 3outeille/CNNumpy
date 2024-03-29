<img src="./img/logo.png" hspace="20%" width="60%">

## Introduction

- **CNNumpy** is a Convolutional Neural Network written in pure Numpy (educational purpose only).
- There are 2 implementation versions:
    - Slow: The naive version with nested for loops.
    - Fast: The im2col/col2im version.
- The [slow implementation][slow-implementation] takes around **4 hours for 1 epoch** where the [fast implementation][fast-implementation] takes only **6 min for 1 epoch**.
- For your information, with the same architecture using **Pytorch**, it will take around **1 min for 1 epoch**.
- **For more details, here are my blog posts explaining in depth what's going on under the hood for each implementation ([slow][slow-blog] and [fast][fast-blog]).**
- In the [`demo-cnnumpy-fast.ipynb`][demo-notebook] notebook, the im2col/col2im implementation can achieve an accuracy up to  **97.2% in 1 epoch (~6 min)**. Here are some results:

<img src="./img/correct-classification.png" hspace="20%" width="50%">
<img src="./img/incorrect-classification.png" hspace="20%" width="50%">


## Installation

- Create a virtual environment in the root folder using [virtualenv][virtualenv] and activate it.

```bash
# On Linux terminal, using virtualenv.
virtualenv myenv
# Activate it.
source myenv/bin/activate
```

- Install **requirements.txt**.

```bash
pip install -r requirements.txt
# Tidy up the root folder.
python3 setup.py clean
```

## Usage of demo notebooks

To play with the `demo-notebooks/` files, you need to make sure jupyter notebook can select your virtual environnment as a kernel.

- Follow **"Installation"** instructions first and make sure your virtual environment is still activated.
- Run the following line in the terminal.
```bash
python -m ipykernel install --user--name=myenv
```
- Run the notebook file **only from `demo_notebooks/`** and then select **Kernel > Switch Kernel > myenv**. You are now ready to go !

<!---
Variables with links.
-->
[slow-implementation]: https://github.com/3outeille/CNNumpy/tree/master/src/slow
[fast-implementation]: https://github.com/3outeille/CNNumpy/tree/master/src/fast
[slow-blog]: https://hackmd.io/@machine-learning/blog-post-cnnumpy-slow
[fast-blog]: https://hackmd.io/@machine-learning/blog-post-cnnumpy-fast
[demo-notebook]: https://github.com/3outeille/CNNumpy/blob/master/src/demo_notebooks/demo-cnnumpy-fast.ipynb
[virtualenv]: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
