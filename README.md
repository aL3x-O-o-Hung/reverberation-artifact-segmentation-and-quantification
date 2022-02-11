# Weakly- and Semi-Supervised Probabilistic Segmentation and Quantification of Ultrasound Needle-Reverberation Artifacts to Allow Better AI Understanding of Tissue Beneath Needles

Official code for [Weakly- and Semi-Supervised Probabilistic Segmentation and Quantification of Ultrasound Needle-Reverberation Artifacts to Allow Better AI Understanding of Tissue Beneath Needles](https://arxiv.org/pdf/2011.11958.pdf).

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
@article{hung2020weakly,
  title={Weakly-and Semi-Supervised Probabilistic Segmentation and Quantification of Ultrasound Needle-Reverberation Artifacts to Allow Better AI Understanding of Tissue Beneath Needles},
  author={Hung, Alex Ling Yu and Chen, Edward and Galeotti, John},
  journal={arXiv preprint arXiv:2011.11958},
  year={2020}
}
```