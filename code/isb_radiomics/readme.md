# IBSI-Compliant Radiomics Extraction

Extract IBSI-compliant radiomics features from DCE-MRI (ISPY1/2, phantoms).

## Install & Build

```bash
pip install cython numpy
python setup.py build_ext --inplace
pip install .