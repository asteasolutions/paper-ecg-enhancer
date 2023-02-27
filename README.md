# Paper ECG Enhancer

Open source project for enhancing imperfectly acquired images of ECG printouts.

## Installation

To use this package as part of another project, install it using pip or pipenv. For example:

```sh
pip install <github_url>
```

## Running Locally

Alternatively, you can checkout the code and run it locally. First you need to create a virtual environment and install the dependencies using Pipenv:

```sh
pipenv install --python 3.9
pipenv shell
```

Then, use the file `run.py` to process ECG images from the command line:

```sh
python run.py <path_to_image> <ecg_polygon>
```

See the usage instructions in `run.py` for more info.

By default, the output of the enhancement process is stored in the `out` directory.

## License

[CC BY-NC-SA](LICENSE.md)
