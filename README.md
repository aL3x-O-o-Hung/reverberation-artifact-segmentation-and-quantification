# Weakly-and Semisupervised Probabilistic Segmentation and Quantification of Reverberation Artifacts

Official code for [Weakly-and Semisupervised Probabilistic Segmentation and Quantification of Reverberation Artifacts](https://downloads.spj.sciencemag.org/bmef/2022/9837076.pdf).

# Pretrained Model
The pretrained model for the second network in our pipeline can be found [here](https://drive.google.com/file/d/1WNSb3xNdmy8hk2SwlhCxxTl-LGH-fqvp/view?usp=sharing), where you should initialize the network by the following parameters:
```python
	model=HierarchicalProbUNet(2,5,[64,128,256,512,1024],3,[4,8,16],name='ProbUNet')
	model.build(input_shape=(None,256,256,5))
	model.load_weights('9.h5',by_name=True,skip_mismatch=True)
```

# Data
The [training data](https://drive.google.com/file/d/1jgg3Sw2VmiEMtI0F5a8SMc-xLNbxHL6c/view?usp=sharing) and the [test data](https://drive.google.com/file/d/1T7oBkT9Z4C3O7AUOLDX0WDHIPjWefQzl/view?usp=sharing) are included. If you need further details about these data, please contact the authors.


# Credit 
If you use the code or the paper in any of your work, please remember to cite us
```bash
@article{hung2022weakly,
  title={Weakly-and Semisupervised Probabilistic Segmentation and Quantification of Reverberation Artifacts},
  author={Hung, Alex Ling Yu and Chen, Edward and Galeotti, John},
  journal={BME Frontiers},
  volume={2022},
  year={2022},
  publisher={AAAS}
}
```