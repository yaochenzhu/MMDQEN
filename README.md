# MMDQEN: Multimodal Deep Quality Embedding Network for Affective Video Content Analysis


 >**Affective video content analysis via multimodal deep quality embedding network,**  
 >Yaochen Zhu, Zhenzhong Chen, Feng Wu  
 >Accepted as a journal paper in IEEE Trans. Affect. Compute.

## Environment

The codes are written in Python 3.6.5 with the following packages.  

- numpy == 1.16.3
- tensorflow-gpu == 1.13.0
- tensorflow-probability == 0.6.0

## Datasets

We are still applying for permission to release the collected stratified and cleaned version of LIRIS-ACCEDE dataset.

For the original LIRIS-ACCEDE dataset, please visit this [URL](https://liris-accede.ec-lyon.fr).

## Examples to run the codes

- **Extract the multimodal feature as described in the paper**: 

- **Train the MMDQEN model via**: 
	
	```python train.py --affect {val, aro}```

For more advanced arguments, run the code with --help argument.
  

## If you find the codes useful, please cite:

	@inproceedings{zhu2019multimodal,
	  title={Multimodal deep denoise framework for affective video content analysis},
	  author={Zhu, Yaochen and Chen, Zhenzhong and Wu, Feng},
	  booktitle={Proceedings of the 27th ACM International Conference on Multimedia},
	  pages={130--138},
	  year={2019}
	}

	@article{zhu2020affective,
	  title={Affective video content analysis via multimodal deep quality embedding network},
	  author={Zhu, Yaochen and Chen, Zhenzhong and Wu, Feng},
	  journal={IEEE Transactions on Affective Computing},
	  year={2020},
	  publisher={IEEE}
	}
