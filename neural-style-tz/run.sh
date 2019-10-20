
source activate tensorflow-gpu1

CUDA_VISIBLE_DEVICES=1 python neural_style.py --content examples/concert.jpg --styles s/starry_night_google.jpg --output r2/concert1.jpg
CUDA_VISIBLE_DEVICES=1 python neural_style.py --content examples/concert.jpg --styles s/the_scream.jpg --output r2/concert2.jpg
CUDA_VISIBLE_DEVICES=1 python neural_style.py --content examples/concert.jpg --styles s/woman-with-hat-matisse.jpg --output r2/concert3.jpg