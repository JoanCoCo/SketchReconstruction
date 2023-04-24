# Reconstruction from Multi-view Sketches: an Inverse Rendering Approach
Code repository for our inverse-rendering-based 3D reconstruction system for drawings and sketches presented at WSCG 2023.

## Abstract
Reconstruction from real images has evolved very differently from reconstruction from sketches. Even though both present similarities, the latter aims to surpass the subjectivity that drawings present, increasing the task's uncertainty and complexity. In this work, we draw inspiration from reconstruction over real multi-view images and adapt it to work over sketches. We leverage inverse rendering as a refinement process for 3D colored meshes while proposing modifications for the domain of drawings. Compared to previous methods for sketches, our proposal recovers not only shape but color, offering an optimization system that does not require previous training. Through the results, we evaluate how different quality factors in sketches affect the reconstruction and report how our proposal adapts to them compared to directly applying existing inverse rendering systems for real images.
![](https://github.com/JoanCoCo/SketchReconstruction/blob/main/images/system_summary.png?raw=true)

## Implementation
This repository contains the implementation of our reconstruction system. It was developed using Python 3.9 using the following packages:
- [PyTorch](https://pytorch.org)
- [NumPy](https://numpy.org)
- [Pillow](https://pillow.readthedocs.io/en/stable/)
- [PyMeshLab](https://pymeshlab.readthedocs.io/en/latest/)
- [Matplotlib](https://matplotlib.org)
- [PyRedner](https://redner.readthedocs.io/en/latest/)
- [TorchMetrics](https://torchmetrics.readthedocs.io/en/latest/)
- [Chamferdist](https://github.com/krrish94/chamferdist)

Please, note that we used a modified version of pyredner with support for mesh color materials. This version can be found at [JoanCoCo/mesh-colors-redner](https://github.com/JoanCoCo/mesh-colors-redner).

### Initial mesh estimation
The initial mesh required for the reconstruction can be generated using ``generate_initial_mesh.py`` from a subset of views. Optionally, the scaling-down factor for the sketches before the projection can be also especified through ``--reduction``, being the default value 15. Example:
``generate_initial_mesh.py --cameras data/plane/views/reference_views.json --output data/plane/meshes/init.obj``

### Reconstruction
Given the drawings (with alpha channel masks), initial mesh and the viewpoints, the reconstruction can be run using either ``train.py`` or ``guardian.py``. The ``data`` folder contains references that can be used for generating example reconstructions, and the ``config`` folder contains examples of training configurations that can be run through ``guardian.py``. Example:
``guardian.py --config configs/plane_r.json``

### Evaluation
Once the reconstruction is finished, it can be evaluated from a set of evaluation images and a reference 3D model using ``validation.py``.
