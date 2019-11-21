#Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GHz * 2, 36 cores
#Tesla P100-PCIE-16GB * 1 (UUID: GPU-ae5cde47-bf7f-a6c6-8a68-8a3c96b2dadf)

CUDA_VISIBLE_DEVICES=1 python neural_style.py --content content/tb-1.jpg     --styles style/7.jpg --output result/result-tb-7.jpg
CUDA_VISIBLE_DEVICES=1 python neural_style.py --content content/jw-1.jpg     --styles style/7.jpg --output result/result-jw-7.jpg
CUDA_VISIBLE_DEVICES=1 python neural_style.py --content content/nju-1.jpg    --styles style/7.jpg --output result/result-nju-7.jpg
CUDA_VISIBLE_DEVICES=1 python neural_style.py --content content/nju-cs-1.jpg --styles style/7.jpg --output result/result-nju-cs-7.jpg

CUDA_VISIBLE_DEVICES=1 python neural_style.py --content content/tb-1.jpg     --styles style/8.jpg --output result/result-tb-8.jpg
CUDA_VISIBLE_DEVICES=1 python neural_style.py --content content/jw-1.jpg     --styles style/8.jpg --output result/result-jw-8.jpg
CUDA_VISIBLE_DEVICES=1 python neural_style.py --content content/nju-1.jpg    --styles style/8.jpg --output result/result-nju-8.jpg
CUDA_VISIBLE_DEVICES=1 python neural_style.py --content content/nju-cs-1.jpg --styles style/8.jpg --output result/result-nju-cs-8.jpg

bash ../push.sh
