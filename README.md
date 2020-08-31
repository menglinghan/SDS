# Measuring Discrimination to Boost Comparative Testing for Multiple Deep Learning Models
This repository stores our experimental codes for the paper "Measuring Discrimination to Boost Comparative Testing for Multiple Deep Learning Models". SDS is short for the approach we proposed in this paper: Sample Discrimination based Selection.
## Datasets
The datasets we used (MNIST,Fashion-MNIST,CIFAR-10) all can be loaded through python's keras package, such as:
```python
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```
## Main experimental codes
You can easily reproduce our method or modify the code we provide.
Our experiments and code are organized according to datasets，so there are three folders in the repository: MNIST, Fashion-MNIST, CIFAR-10. You can find our method and the baselines used in each folder.
Our method is called SDS for short, and you can find the code of our method in the folder named SDS, which is named SDS.py. And in each subfolder you can find the model we use, 28 for MNIST, 25 for Fashion-MNIST, 27 for CIFAR-10. You can run SDS.py directly to reproduce our original experimental results. If you want to conduct your own experiments, you can modify the model and code accordingly.
In the folder of each method, we will save the model accuracy data measured by sampling into the corresponding model folder for later calculation of the Spearman coefficient and the Jaccard coefficient.
## Baseline methods
We have used four baseline methods in total, and you can find the corresponding method in each folder named after the dataset, 'SRS', 'CES', 'RDG', 'DDG'. We will explain these abbreviations and explain the experimental settings of the baseline.
'SRS': short for the Simple Random Selection.
'CES': short for the method proposed in the paper 'Boosting Operational DNN Testing Efficiency through Conditioning'. CES is a sampling method for a single model. Since our experiment is for a multi-model scenario, we use the CES method to sample for all models, and then select the best performing subset as the baseline. The best performance here means that we use the sampled subset of the model to measure the spearman coefficients and jaccard coefficients of all model accuracy (35-180) and the final model accuracy, and calculate the mean value of all points, and the largest mean value is regarded as the optimal subset. When the code is reproduced, it is necessary to modify the model name in the corresponding position of the code when sampling for different models.
'RDG': RDG is the baseline we used after randomization according to the method proposed in the paper 'DeepGini: Prioritizing Massive Tests to Enhance the Robustness of Deep Neural Networks'. For each sample, we calculated the \xi values of all models, and selected the smallest value as the mark of the sample, sorted the samples in ascending order according to the mark, and randomly sampled in the first 25%. We also carried out experiments by taking the largest value of all models as the mark and random sampling in the top 25%, but the effect was not as good as described before, so we still use the previous method as the baseline.
'DDG': This is also using the method in the paper 'DeepGini: Prioritizing Massive Tests to Enhance the Robustness of Deep Neural Networks'. But this time we do not perform random sampling, but take the top 180 marks ranked as the test subset. In other words, this is a deterministic sampling method.
You can find the corresponding code in the folder named by the corresponding method to reproduce. If you need to conduct your own experiment, you can also modify the code.
