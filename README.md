# Deep Harvest: Deep Learning Approaches for Rice Grain Classification

## 1. Abstract:
This research emphasizes on the challenge of food security and quality control in the global rice supply while considering the importance of precise rice grain classification. As traditional methods demand a new transformative approach in this study, this approach uses advanced deep learning models, like ResNet, VGG, EfficientNet, MobileNet, and Inception which automates the process of rice grains classification. Evaluation is conducted on a diverse dataset of 25,000 images across five distinct rice categories. The models resulted in providing remarkable accuracy, efficiency, and scalability. The Inception model achieved an accuracy of 99.95%, while MobileNet  Model could complete the task in minimal time. The research is also useful in the foundation for elevating global food safety standards, refining agricultural practices, and consumer satisfaction and is not just limited to the field of automated rice grain classification.

## 2. Introduction:
With the use of deep learning models such as ResNet, VGG, DenseNet, Inception, Xception, and MobileNet among which Inception outshines other models with a high accuracy of 99.95% this proposed deep learning approach significantly advances the field of rice grain classification. It also reduces the computational load associated with model training, making the approach more feasible and approachable. This helped models exhibit good generalization ability, indicating their potential for real-world applications. Furthermore, our study contributes to advancing the state-of-the-art in rice grain classification, with implications for agricultural productivity, food management industries, food quality management departments, and food security. 

## 3. Proposed Methodology:
![image](https://github.com/piyush8227/Rice-Grain-Image-Classification-Task/assets/78916771/ddf06449-1b73-4bf2-b7e8-ef9fa362a990)

### Inception Model Architecture:
![image](https://github.com/piyush8227/Rice-Grain-Image-Classification-Task/assets/78916771/451848f0-13fc-462f-92fc-bca6a138e1fe)

## 4. Experimental Setup
Dataset: A dataset of 25,000 rice grain images, evenly divided into five classes: Karacadag, Basmati, Jasmine, Arborio, and Ipsala, was collected from a public source. These images are in RGB format and boast a resolution of 250 x 250 pixels, each capturing a single grain of rice. This large volume of images and the diversity of rice types provide a challenging environment for DL models, facilitating a rigorous comparison of their performance. Figure 3 below provides some examples of each rice grain class from the dataset.

![image](https://github.com/piyush8227/Rice-Grain-Image-Classification-Task/assets/78916771/721c9cac-c8c2-4445-a2ef-58339dc83c9c)

Figure: Example of rice grain classes: (a) Arborio (b) Basmati (c)Jasmine (d) Ipsala (e) Karacadag
The images were pre-processed to ensure consistency in size, color, and orientation. This involved resizing and normalizing pixel values. Seven pre-trained deep learning models were employed for rice grain classification: VGG16, ResNet, DenseNet, Inception, Xception, MobileNet, and VGG19. These models were selected based on their proven performance in image classification tasks and their ability to extract complex features from image data. 

## 5. Conclusion
The experiment results highlight the exceptional performance of MobileNet, leading with 99.97% accuracy, followed closely by Inception, Xception, DenseNet, and ResNet. The superiority of newer models over VGG16 and VGG19 underscores the critical role of model architecture in optimizing rice grain classification. This emphasizes deep learning's efficacy in automating rice grain classification, offering significant benefits to the agricultural and food industries, and enhancing quality control, sorting, and grading processes.

## 6. Tech Stack used:-
* Python
* VS Code Jupyter Notebooks
* Tensorflow, Keras, Deep learning models
* Visualization libraries like Matplotlib.Pyplot, and Seaborn.
* Numpy, Pandas, and Sklearn.
  
## 7. References:-
* **[Keras](https://keras.io/api/)**
* **[TensorFlow](https://www.tensorflow.org/)**
* **[Scikit Learn](https://scikit-learn.org/stable/)**
* **[Articles](https://www.sciencedirect.com/science/article/pii/S1746809422007224)**
* **[Articles](https://www.nature.com/articles/s41598-021-93832-2)**
