# ViT 16

The Vision Transformer (ViT) architecture, introduced by Dosovitskiy et al., diverges from the conventional approach of using convolutions to process images. Instead, it splits an image into fixed-size patches, which enables the transformer to capture long-range dependencies, a capability that is particularly beneficial for understanding complex spatial relationships in images. The core idea is to treat image patches as tokens, analogous to words in a sentence, and employ a transformer encoder to model their relationships.

<br>

<p align="center">
  <img src="https://github.com/andreleo02/deep-dream-team/blob/031f002da2bcf765f06f9bd4f48f8dcc7812f741/Vit.png?raw=true" width="512"/>  
</p>

<p align="center">
  <sub><em> ViT 16 architecture: as can be seen in this figure, ViT16 splits the input image into fixed-size patches, it linearly embed them and feed this embeddings to the Transformer Encoder. Then, a standard MLP Head is used for classification.</em></sub>
</p>

For further information please visit the official Pytorch [documentation (https://pytorch.org/vision/main/models/vision_transformer.html) about Vision Transformer.
