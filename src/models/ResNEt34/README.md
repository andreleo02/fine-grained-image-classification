# ResNet34
The advent of residual neural networks (ResNets) represented a significant development in the field of deep learning, as they were the first architectures to address the crucial problem of vanishing gradients. The core concept behind ResNets is the introduction of "residual blocks" in the network, which enable the network to focus on learning residuals (the difference between the input and the output) instead of attempting to learn a direct mapping from the input to the output. Formally, if the desired underlying mapping is \( H(x) \), ResNet reformulates it as \( H(x) = F(x) + x \), where \( F(x) \) represents the residual mapping that the network learns. 

<br>

<p align="center">
  <img src="https://github.com/andreleo02/deep-dream-team/blob/7c501fee584c114c5da3420a5671b1a56808972b/res%20net.png?raw=true" width="512"/>  
</p>

<p align="center">
  <sub><em>Residual Block of a Residual Network (ResNet): the Residual Connection skips two layers.</em></sub>
</p>

For further information please visit the official Pytorch [documentation](https://pytorch.org/hub/pytorch_vision_resnet/) about ResNet.
