![skdh_badge](https://github.com/PfizerRD/scikit-digital-health/workflows/skdh/badge.svg)

Scikit Digital Health (SKDH) is a Python package with methods for ingesting and analyzing wearable inertial sensor data.

- Documentation: https://scikit-digital-health.readthedocs.io/en/latest/
- Bug reports: https://github.com/PfizerRD/scikit-digital-health/issues
- Contributing: https://scikit-digital-health.readthedocs.io/en/latest/src/dev/contributing.html

SKDH provides the following:

- Methods for ingesting data from binary file formats (ie Axivity, GeneActiv)
- Preprocessing of accelerometer data
- Common time-series signal features
- Common time-series/inertial data analysis functions
- Inertial data analysis algorithms (ie gait, sit-to-stand, sleep, activity)

### Availability


SKDH is available on both `conda-forge` and `PyPI`.

```shell
conda install scikit-digital-health -c conda-forge
```

or 

```shell
pip install scikit-digital-health
```

> [!WARNING]
> Windows pre-built wheels are provided as-is, with limited/no testing on changes made to compile extensions for Windows.

> [!NOTE]
> Windows users may need to install an additional requirement: Microsoft Visual C++ redistributable >14.0. The 2015 version can be found here: https://www.microsoft.com/en-us/download/details.aspx?id=53587

### Build Requirements

As of 0.9.15, Scikit Digital Health is built using Meson.


### Citation

If you use SKDH in your research, please include the following citation:

<a id="1">[1]</a>
L. Adamowicz, Y. Christakis, M. D. Czech, and T. Adamusiak, “SciKit Digital Health: Python Package for Streamlined Wearable Inertial Sensor Data Processing,” JMIR mHealth and uHealth, vol. 10, no. 4, p. e36762, Apr. 2022, doi: 10.2196/36762.

