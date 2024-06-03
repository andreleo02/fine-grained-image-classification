# ResNet34

The advent of residual neural networks (ResNets) represented a significant development in the field of deep learning, as they were the first architectures to address the crucial problem of vanishing gradients. The core concept behind ResNets is the introduction of "residual blocks" in the network, which enable the network to focus on learning residuals (the difference between the input and the output) instead of attempting to learn a direct mapping from the input to the output. Formally, if the desired underlying mapping is \( H(x) \), ResNet reformulates it as \( H(x) = F(x) + x \), where \( F(x) \) represents the residual mapping that the network learns.

<br>

<p align="center">
  <img src="./res net.png" width="512"/>  
</p>

<p align="center">
  <sub><em>Residual Block of a Residual Network (ResNet): the Residual Connection skips two layers.</em></sub>
</p>

This reformulation is implemented through the use of "skip connections," another innovation introduced by ResNets. These connections facilitate the flow of the gradient through the network during backpropagation, thereby preventing the vanishing gradient problem

***Official paper [available](https://arxiv.org/pdf/1512.03385v1).***

For further information please visit the official Pytorch [documentation](https://pytorch.org/hub/pytorch_vision_resnet/) about ResNet.
