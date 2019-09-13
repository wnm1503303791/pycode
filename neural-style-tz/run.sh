#########################################################################
# File Name: run.sh
# Author: tuzhuo
# mail: xmb028@163.com
# Created Time: 2019年09月12日 星期四 15时59分03秒
#########################################################################
#!/bin/bash

source activate tensorflow_env1
python neural_style.py --content examples/tz.jpg --styles examples/1-style.jpg --output r/r-tz.jpg
