# Predicting the Sensory Appeal of Tea Using Machine Learning

## Abstract

This study addresses the challenge of predicting the sensory appeal of tea based on its chemical composition using machine learning models. Given the limited availability of data in this domain, I employed techniques for data imputation and generation, including Kernel Density Estimation (KDE) for generating a training set and a Generative Adversarial Network (GAN) for imputing sensory data. I developed and compared three models: a Random Forest, a Multilayer Perceptron (MLP), and a Recurrent Neural Network (RNN), to predict an aggregated 'Overall Sensory Score' from tea's chemical constituents. The models were evaluated based on their prediction accuracy, and their predictions were visualized to offer insights into the relationship between chemical composition and sensory appeal. The trained models and data preprocessing scalers were exported into the ONNX format for deployment in a web application, facilitating the practical application of our findings.

### 1. Introduction

Green tea (*Camellia sinensis*), a beverage enjoyed globally, is renowned for its intricate chemical makeup, which plays a crucial role in shaping its sensory characteristics. Despite its popularity, accurately predicting its sensory appeal based on chemical composition, including elements like polyphenols and caffeine, presents a significant challenge. This study aims to address these challenges by employing machine learning methods to develop a predictive model. Such a model has the potential to provide valuable insights for tea manufacturers and tea drinkers alike, explaining the impact of various chemical constituents on the sensory experience of green tea.

### 2. Methodology

#### 2.1 Compilation of Datasets

The initial step involved compiling the dataset from varied academic sources, focusing on tea catechins and sensory evaluations. This foundational phase set the stage for the subsequent preprocessing and enhancement steps.

#### 2.2 Data Imputation and Generation

- **Iterative Imputer:** An iterative imputer was employed to handle missing data, ensuring the dataset's completeness and enhancing its quality for further processing.
- **Kernel Density Estimation (KDE) for Synthetic Data Generation:** A key strategy implemented was the use of KDE for the first round of synthetic data generation. This approach was aimed at augmenting the limited dataset with additional, plausible data points, expanding it in a statistically sound manner.
- **Generative Adversarial Network (GAN):** A GAN, developed in PyTorch, was utilized to generate sensory data. This enriched the dataset with imputed scores, maintaining the integrity and variability of the original sensory data.
- **Second KDE for GAN Training Set:** A second KDE-generated dataset was specifically crafted for training the GAN. This addressed the unique challenge of imputing missing sensory evaluation scores, particularly relevant since most datasets primarily included observations of polyphenols and caffeine.

#### 2.3 Feature Engineering

In this phase, individual sensory scores were aggregated into an 'Overall Sensory Score.' This simplification aimed to streamline the modeling process and provide a unified target variable for prediction, enhancing both the model building and the user interaction with the model.

#### 2.4 Data Scaling

To normalize the dataset and prepare it for model training, Min-Max scaling was applied to both chemical compositions and sensory scores separately. This step was crucial for ensuring that the data was in a suitable format for effective model training and analysis.

#### 2.5 Dimensionality Reduction with Principal Component Analysis (PCA)

Although PCA was used for data analysis, it was ultimately excluded from direct application in the user interface (UI) portion of the project. This decision was motivated by the desire to maintain the transparency and interpretability of the original features for end-users, ensuring that the information remained accessible and comprehensible.

### 3. Models Development

I developed and trained three different models:

- **Random Forest:** A robust ensemble method known for its high accuracy and ability to handle non-linear data.
- **Multilayer Perceptron (MLP):** A class of feedforward artificial neural network that can model complex relationships between inputs and outputs.
- **Recurrent Neural Network (RNN):** An advanced neural network architecture that can capture temporal dynamic behavior, suitable for sequence prediction tasks.

Each model's performance was evaluated based on mean squared error (MSE) between the predicted and actual sensory scores.

### 4. Results

Through evaluation, the Multilayer Perceptron (MLP) model delivered the most accurate predictions of the overall sensory score, closely followed by the Random Forest and then the Recurrent Neural Network (RNN) model. The visualization of the models' predictions facilitated a comparison of their performance and highlighted the complex relationship between the chemical composition of tea and its sensory appeal.

### 5. Discussion

This study demonstrates the significant promise of machine learning techniques in forecasting the sensory appeal of tea from its chemical composition. The MLP model's consistent accuracy and its near-perfect predictions underscore its aptness for mapping the intricate links between tea's chemical makeup and its sensory appeal.

### 6. Conclusion

This study marks a significant advancement in the field by introducing a novel methodology for predicting the sensory appeal of tea based on its chemical composition. By addressing the challenge of limited data availability through cutting-edge data imputation and generation techniques, our approach provides a valuable tool for the agricultural industry.

### References

- [Scikit-learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html)
- [Scikit-learn FunctionTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
- [Seaborn Pairplot](https://seaborn.pydata.org/generated/seaborn.pairplot.html)
- [PyTorch GPU Usage Check](https://stackoverflow.com/questions/48152674/how-do-i-check-if-pytorch-is-using-the-gpu)
- [Scikit-learn SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
- [Kernel Density Estimation Explained](https://towardsdatascience.com/kernel-density-estimation-explained-step-by-step-7cc5b5bc4517)
- [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [GAN Simple Implementation with PyTorch](https://kikaben.com/gangenerative-adversarial-network-simple-implementation-with-pytorch/)
- [Google Developers GAN Structure](https://developers.google.com/machine-learning/gan/gan_structure)
- [PyTorch-GAN by Erik Linder-Norén](https://github.com/eriklindernoren/PyTorch-GAN)
- [Deep Learning for Computer Vision: PyTorch GAN](https://www.run.ai/guides/deep-learning-for-computer-vision/pytorch-gan)
- [Scikit-learn Kernel PCA Explained Variance](https://stackoverflow.com/questions/29611842/scikit-learn-kernel-pca-explained-variancepca.explained_variance_)
- [Scikit-learn Kernel PCA](https://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html)
- [PyTorch MLP Documentation](https://pytorch.org/vision/main/generated/torchvision.ops.MLP.html)
- [Building Multilayer Perceptron Models in PyTorch](https://machinelearningmastery.com/building-multilayer-perceptron-models-in-pytorch/)
- [PyTorch Image Classification with MLP](https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb)
- [Reset Parameters of a Neural Network in PyTorch](https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch)
- [Introduction to PyTorch Training](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)
- [PyTorch RNN Documentation](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
- [A Beginner's Guide on Recurrent Neural Networks with PyTorch](https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/)
- [Visualizing Models, Data, and Training with PyTorch](https://github.com/szagoruyko/pytorchviz)
- [Super Resolution with ONNXRuntime](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
- [skl2onnx: Convert Scikit-learn models to ONNX](https://pypi.org/project/skl2onnx/)
- [Bootstrap Card Components](https://getbootstrap.com/docs/5.3/components/card/)
- [Tailwind CSS Documentation](https://tailwindcss.com/)
- [Configuring HTTPS in AWS Elastic Beanstalk](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/configuring-https.html)
- [Handling Non-JSON-Serializable Data](https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable)

#### Data sources

- [Journal of Agricultural and Food Chemistry](https://doi.org/10.1080/10942910701299430)
- [Antioxidants](https://doi.org/10.3390/antiox8060180)
- [Journal of Food Composition and Analysis](https://doi.org/10.1016/j.jfca.2020.103684)
- [ICAR](https://krishi.icar.gov.in/jspui/bitstream/123456789/68751/2/S0889157520313892-main.pdf)
- [Journal of Chromatography A](https://doi.org/10.1016/S0021-9673(00)00215-6)
- [Journal of Agricultural and Food Chemistry](https://doi.org/10.1021/jf980223x)
- [Food Chemistry](https://doi.org/10.1016/j.foodchem.2012.03.039)
- [Nutritional Cancer](https://doi.org/10.1207/S15327914NC4502_13)
- [Food Chemistry](https://doi.org/10.1016/j.foodchem.2010.08.02)
- [Journal of Dietary Supplements](https://doi.org/10.1080/J157v03n03_03)
- [Phenol-Explorer](http://phenol-explorer.eu)
- [Food Chemistry](https://doi.org/10.1016/j.foodchem.2017.09.126)
- [Journal of Food Science](https://doi.org/10.1111/j.1365-2621.2009.02040.x)
- [Food Science & Nutrition](https://doi.org/10.1002%2Ffsn3.1143)
- [Molecules](https://doi.org/10.3390/molecules23071689)