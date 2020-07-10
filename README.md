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
- Remember that you can't run files from `src/`. You have to go to either `fast/` or `slow/` folder first.