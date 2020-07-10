<img src="./img/logo.png" hspace="20%" width="60%">

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

## Usage of demo jupyter notebook files

To play with the `demo-notebooks/` files, you need to make sure jupyter notebook can select your virtual environnment as a kernel.

- Follow **"Installation"** instructions first and make sure your virtual environment is still activated.
- Run the following line in the terminal.
```bash
python -m ipykernel install --user--name=myenv
```
- Run the notebook file and then select **Kernel > Switch Kernel > myenv**. You are now ready to go !

<!---
Variables with links.
-->

[virtualenv]: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
