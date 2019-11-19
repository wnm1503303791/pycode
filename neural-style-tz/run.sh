#Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GHz * 2, 36 cores
#Tesla P100-PCIE-16GB * 1 (UUID: GPU-ae5cde47-bf7f-a6c6-8a68-8a3c96b2dadf)

CUDA_VISIBLE_DEVICES=1 python neural_style.py --content content/nju-cs-1.jpg --styles style/1.jpg --output result/result-nju-cs-1.jpg
