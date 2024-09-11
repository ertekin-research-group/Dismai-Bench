# Disordered Materials &amp; Interfaces Benchmark (Dismai-Bench)
Dismai-Bench is a generative model benchmark for inorganic materials, that evaluates models on datasets of large disordered materials.

<details open>
  <summary>Click to collapse/expand the figure below</summary>
  <p align="center">
    <img src="assets/gen_examples.gif" width="500"> 
  </p>
</details>

## Prerequisites
The following packages are required for benchmarking:
- ase = 3.22.0
- m3gnet = 0.0.7
- matminer = 0.7.6
- matplotlib = 3.5.1
- numpy = 1.26.2
- pandas = 1.4.1
- pymatgen = 2023.10.11
- quippy-ase = 0.9.12
- scipy = 1.11.3
- seaborn = 0.13.2
- tqdm = 4.63.0
- vasppy = 0.7.1.0

Note that these are only suggested package versions; other versions may also be compatible.

Consider running the following commands to set up the environment:
```
conda create --name dismai_bench_env python=3.9
conda activate dismai_bench_env
pip install ase==3.22.0 m3gnet==0.0.7 matminer==0.7.6 matplotlib==3.5.1 numpy==1.26.2 pandas==1.4.1 pymatgen==2023.10.11 quippy-ase==0.9.12 scipy==1.11.3 seaborn==0.13.2 tqdm==4.63.0 vasppy==0.7.1.0
```

If you wish to run any of the generative models, you will need to set up separate environment(s). Please see [Generative models](#generative-models) below.

## Generative models
The five generative models benchmarked on Dismai-Bench are as follows:
1. [CDVAE](https://github.com/ertekin-research-group/Dismai-Bench/tree/main/models/cdvae)
2. [DiffCSP](https://github.com/ertekin-research-group/Dismai-Bench/tree/main/models/diffcsp)
3. [CrysTens](https://github.com/ertekin-research-group/Dismai-Bench/tree/main/models/crystens)
4. [UniMat](https://github.com/ertekin-research-group/Dismai-Bench/tree/main/models/unimat)
5. [CryinGAN](https://github.com/ertekin-research-group/CryinGAN)

Please follow the links for instructions to set up and run any of the models. Note that CryinGAN is located in its own repository.

Also note that 1-4 are modified versions of the original models, as described in the Dismai-Bench paper.

## Datasets and interatomic potentials
The datasets and interatomic potentials are available [here](https://doi.org/10.5281/zenodo.12710372).

`dismai_bench_train_ref_data` contains the training dataset reference data for calculating benchmark metrics.

`potentials` contains the M3GNet and SOAP-GAP interatomic potentials.

`train_val_data` contains the training and validation sets in .extxyz format, as well as in the input format of each generative model.

`generated_data` contains the generated structures (post-processed and relaxed) from each generative model.

## Benchmarking
Please follow the links below for specific benchmarking instructions:
1. [Disordered LSC-LCO interface](https://github.com/ertekin-research-group/Dismai-Bench/tree/main/benchmark/int)
2. [Amorphous silicon](https://github.com/ertekin-research-group/Dismai-Bench/tree/main/benchmark/a_Si)
3. [Disordered stainless steel alloy](https://github.com/ertekin-research-group/Dismai-Bench/tree/main/benchmark/alloy)

It is recommended to run benchmarking on multiple cpus instead of your own PC to speed up the process. 
The scripts are written to facilitate parallel jobs.

## Citations
*Please cite the appropriate paper(s) if used in your work:*

<details>
  <summary>Dismai-Bench</summary>
  
  ```
  @article{yong2024dismaibench,
           author = {Yong, Adrian Xiao Bin and Su, Tianyu and Ertekin, Elif},
           title = {Dismai-Bench: benchmarking and designing generative models using disordered materials and interfaces},
           journal = {Digital Discovery},
           year = {2024},
           volume = {3},
           issue = {9},
           pages = {1889-1909},
           publisher = {RSC},
           doi = {10.1039/D4DD00100A},
           url = {http://dx.doi.org/10.1039/D4DD00100A}
  }
  ```
</details>

<details>
  <summary>Disordered stainless steel alloy dataset and cluster expansion potential</summary>
  
  ```
  @article{su2024ssalloy,
           author = {Su, Tianyu and Blankenau, Brian J. and Kim, Namhoon and Krogstad, Jessica A. and Ertekin, Elif},
           title = {First-principles and cluster expansion study of the effect of magnetism on short-range order in Fe–Ni–Cr austenitic stainless steels},
           journal = {Acta Materialia},
           volume = {276},
           pages = {120088},
           ISSN = {1359-6454},
           DOI = {https://doi.org/10.1016/j.actamat.2024.120088},
           url = {https://www.sciencedirect.com/science/article/pii/S1359645424004397},
           year = {2024}
  }
  ```
</details>

<details>
  <summary>Amorphous silicon dataset</summary>

  ```
  @article{deringer2021asidata,
           author = {Deringer, Volker L. and Bernstein, Noam and Csányi, Gábor and Ben Mahmoud, Chiheb and Ceriotti, Michele and Wilson, Mark and Drabold, David A. and Elliott, Stephen R.},
           title = {Origins of structural and electronic transitions in disordered silicon},
           journal = {Nature},
           volume = {589},
           number = {7840},
           pages = {59-64},
           ISSN = {1476-4687},
           DOI = {10.1038/s41586-020-03072-z},
           url = {https://doi.org/10.1038/s41586-020-03072-z},
           year = {2021}
  }
  ```
</details>

<details>
  <summary>SOAP-GAP interatomic potential</summary>

  ```
  @article{bartok2018soapgap,
           author = {Bartók, Albert P. and Kermode, James and Bernstein, Noam and Csányi, Gábor},
           title = {Machine Learning a General-Purpose Interatomic Potential for Silicon},
           journal = {Physical Review X},
           volume = {8},
           number = {4},
           pages = {041048},
           DOI = {10.1103/PhysRevX.8.041048},
           url = {https://link.aps.org/doi/10.1103/PhysRevX.8.041048},
           year = {2018}
  }
  ```
</details>

<details>
  <summary>M3GNet interatomic potential</summary>

  ```
  @article{chen2022m3gnet,
           author = {Chen, Chi and Ong, Shyue Ping},
           title = {A universal graph deep learning interatomic potential for the periodic table},
           journal = {Nature Computational Science},
           volume = {2},
           number = {11},
           pages = {718-728},
           ISSN = {2662-8457},
           DOI = {10.1038/s43588-022-00349-3},
           url = {https://doi.org/10.1038/s43588-022-00349-3},
           year = {2022}
  }
  ```
</details>

<details>
  <summary>CDVAE</summary>

  ```
  @misc{xie2022cdvae,
        title={Crystal Diffusion Variational Autoencoder for Periodic Material Generation}, 
        author={Tian Xie and Xiang Fu and Octavian-Eugen Ganea and Regina Barzilay and Tommi Jaakkola},
        year={2022},
        eprint={2110.06197},
        archivePrefix={arXiv},
        primaryClass={cs.LG},
        url={https://arxiv.org/abs/2110.06197}
  }
  ```
</details>

<details>
  <summary>DiffCSP</summary>

  ```
  @misc{jiao2024diffcsp,
        title={Crystal Structure Prediction by Joint Equivariant Diffusion}, 
        author={Rui Jiao and Wenbing Huang and Peijia Lin and Jiaqi Han and Pin Chen and Yutong Lu and Yang Liu},
        year={2024},
        eprint={2309.04475},
        archivePrefix={arXiv},
        primaryClass={cond-mat.mtrl-sci},
        url={https://arxiv.org/abs/2309.04475}
  }
  ```
</details>

<details>
  <summary>CrysTens</summary>

  ```
  @article{alverson2024crystens,
           author = {Alverson, Michael and Baird, Sterling G. and Murdock, Ryan and Ho, Sin-Hang and Johnson, Jeremy and Sparks, Taylor D.},
           title = {Generative adversarial networks and diffusion models in material discovery},
           journal = {Digital Discovery},
           volume = {3},
           number = {1},
           pages = {62-80},
           DOI = {10.1039/D3DD00137G},
           url = {http://dx.doi.org/10.1039/D3DD00137G},
           year = {2024}
  }
  ```
</details>

<details>
  <summary>UniMat</summary>

  ```
  @misc{yang2024unimat,
        title={Scalable Diffusion for Materials Generation}, 
        author={Sherry Yang and KwangHwan Cho and Amil Merchant and Pieter Abbeel and Dale Schuurmans and Igor Mordatch and Ekin Dogus Cubuk},
        year={2024},
        eprint={2311.09235},
        archivePrefix={arXiv},
        primaryClass={cs.LG},
        url={https://arxiv.org/abs/2311.09235}
  }
  ```
</details>
