# Building the Documentation

## Installing Dependencies

Please install the following packages before building the PyNIF3D documentation:

```
pip install -U m2r2 mock sphinx sphinx_rtd_theme sphinx_markdown_tables
```

## Building

From the `docs` directory, please run the following:

```
make html
```

The documentation will be generated in `_build/html`.

## Viewing the Documentation

From the `docs` directory, please run the following:

```
python -m http.server
```

Open a web browser and go to `http://0.0.0.0:8000`, then navigate to `_build/html`. The
API documentation should now be available.
