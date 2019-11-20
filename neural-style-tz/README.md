# Neural-Style-TZ

Last modified @ 2019/11/19

<img src="content/nju-1.jpg" width="420">
<img src="style/1.jpg" width="420">
<img src="result/result-nju-1.jpg" width="420">

## contibutor
tuzhuo @ croplab, HZAU

## operating environment:
1. Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GHz * 2 (36 cores)
2. Tesla P100-PCIE-16GB * 2
3. 512GB RAM
4. tensorflow-gpu, NumPy, SciPy, Pillow, CUDA, vgg.mat
5. for a 1200*950 pixel image, with the environment above, it only takes 5 mins for 1000 iterations. Using a GPU is highly recommended due to the huge speedup.

## Appendix
[source project](https://github.com/anishathalye/neural-style)