For the first time, with EfficientNet, a new approach has been introduced: compound scaling. This approach focuses on the simultaneous scaling of all three dimensions of a neural network (width, depth and resolution of the input), rather than on a single dimension alone. In the Efficient Net, three constant coefficients are determined by conducting a small grid search on the model. The depth (d) denoted by \(\alpha^\Phi\), the width (w) represented by \(\beta^\phi\) and finally the image size (r) represented by \(\gamma^\phi\). These three values are scaled by a compound coefficient \(\phi\) such that:
\[
\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2
\]
\[
\alpha \geq 1, \quad \beta \geq 1, \quad \gamma \geq 1
\]
