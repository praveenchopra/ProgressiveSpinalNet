# ProgressiveSpinalNet
ProgressiveSpinalNet architecture for FC layers.
In deeplearning models the FC (fully connected) layer has biggest important role for classification of the input based on the learned features from previous layers. The FC layers has highest numbers of parameters and fine-tuning these large numbers of parameters, consumes most of the computational resources, so in this paper we are aiming to reduce these large numbers of parameters significantly with improved performance. The motivation is inspired from SpinalNet and other biological architecture. The proposed architecture has a gradient highway between input to output layers and this solves the problem of diminishing gradient in deep networks. In this all the layers receives the input from previous layers as well as the CNN layer output and this way all layers contribute in decision making with last layer. This approach has improved classification performance over the SpinalNet architecture and has SOTA performance on many datasets such as Caltech101, KMNIST, QMNIST and EMNIST. The structure of this network does not requires too many parameters, this leads to improvement in performance with less computations. This network concept can be easily expanded to wide range of real-world scenarios.
