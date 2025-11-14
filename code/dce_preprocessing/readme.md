# DCE-MRI Preprocessing & Perturbation

A **reproducible, citable** pipeline for:

* VOI cropping & padding  
* Isotropic interpolation (3-D or slice-wise)  
* Intensity resegmentation & fixed-bin discretisation  
* Image-filter banks (your `image_filtering`)  
* Robustness testing via **contour randomisation + rigid perturbations**

Used in the ISPY-2 radiomics-to-genomics project.

## Install

```bash
pip install dce-preprocessing