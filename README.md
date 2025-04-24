# CKGConv-GAD: Graph Anomaly Detection with Continuous Kernel Graph Convolution

This project explores the integration of **CKGConv** into unsupervised Graph Anomaly Detection (GAD) frameworks. We replace standard GCN encoders with CKGConv in state-of-the-art models like **DOMINANT** and **CoLA**, exploring performance on both real and synthetic datasets.

CKGConv is a general graph convolution operator based on continuous kernels and graph positional encodings, introduced in:

> **CKGConv: General Graph Convolution with Continuous Kernels**  
> Liheng Ma et al., ICML 2024  
> [[Paper](https://arxiv.org/abs/2404.13604)] | [[Code](https://github.com/networkslab/CKGConv)]

## Requirements
 
Install packages using `environment.yml`:

##  Experimental Results

### 📊 Benchmark Results (AUC / AP / F1)

| Model            | Dataset     | AUC             | AP              | F1              |
|------------------|-------------|------------------|------------------|------------------|
| cola             | gen_500     | 0.5350 ± 0.0653  | 0.0758 ± 0.0378  | 0.0743 ± 0.0508  |
| cola             | gen_1000    | 0.5296 ± 0.0892  | 0.0464 ± 0.0276  | 0.0500 ± 0.0314  |
| cola             | gen_time    | 0.4980 ± 0.0261  | 0.2035 ± 0.0100  | 0.1488 ± 0.0204  |
| cola             | books       | 0.5203 ± 0.0447  | 0.0225 ± 0.0077  | 0.0200 ± 0.0157  |
| cola             | disney      | 0.4799 ± 0.1477  | 0.0807 ± 0.0505  | 0.0737 ± 0.0999  |
| dominant         | gen_500     | 0.7915 ± 0.0044  | 0.5179 ± 0.0025  | 0.3143 ± 0.0000  |
| dominant         | gen_1000    | 0.7344 ± 0.0010  | 0.4113 ± 0.0051  | 0.1833 ± 0.0000  |
| dominant         | gen_time    | 0.7643 ± 0.0001  | 0.6880 ± 0.0002  | 0.6574 ± 0.0000  |
| dominant         | books       | 0.4769 ± 0.0379  | 0.0244 ± 0.0110  | 0.0235 ± 0.0175  |
| dominant         | disney      | 0.4018 ± 0.0957  | 0.0476 ± 0.0093  | 0.0211 ± 0.0444  |
| ckgconv_dominant | gen_500     | 0.7890 ± 0.0078  | 0.5204 ± 0.0078  | 0.3143 ± 0.0000  |
| ckgconv_dominant | gen_1000    | 0.7144 ± 0.0240  | 0.1265 ± 0.0231  | 0.1633 ± 0.0205  |
| ckgconv_dominant | gen_time    | 0.7591 ± 0.0014  | 0.6848 ± 0.0040  | 0.6623 ± 0.0066  |
| ckgconv_dominant | books       | 0.3636 ± 0.0201  | 0.0150 ± 0.0009  | 0.0095 ± 0.0094  |
| ckgconv_dominant | disney      | 0.4808 ± 0.0007  | 0.0564 ± 0.0000  | 0.0000 ± 0.0000  |
| ckgconv_cola     | gen_500     | 0.5124 ± 0.0691  | 0.0493 ± 0.0117  | 0.0571 ± 0.0504  |
| ckgconv_cola     | gen_1000    | 0.5026 ± 0.0921  | 0.0244 ± 0.0076  | 0.0317 ± 0.0214  |
| ckgconv_cola     | gen_time    | 0.5025 ± 0.0177  | 0.1933 ± 0.0094  | 0.1349 ± 0.0245  |
| ckgconv_cola     | books       | 0.5380 ± 0.0503  | 0.0245 ± 0.0050  | 0.0318 ± 0.0112  |
| ckgconv_cola     | disney      | 0.4952 ± 0.1068  | 0.0634 ± 0.0249  | 0.0421 ± 0.0736  |


## Citation

The original CKGConv paper:

```bibtex
@inproceedings{ma2024ckgconv,
  title={CKGConv: General Graph Convolution with Continuous Kernels},
  author={Ma, Liheng and Pal, Soumyasundar and Zhang, Yitian and others},
  booktitle={International Conference on Machine Learning},
  year={2024},
  url={https://arxiv.org/abs/2404.13604}
}


---

## Acknowledgements

Built upon [PyGOD](https://github.com/pygod-team/pygod) and [CKGConv](https://github.com/networkslab/CKGConv).
```
