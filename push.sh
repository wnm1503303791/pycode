#########################################################################
# File Name: push.sh
# Author: tuzhuo
# mail: xmb028@163.com
# Created Time: 2019年09月11日 星期三 19时56分03秒
#########################################################################
#!/bin/bash
cur_date="date +%Y-%m-%d"
git add -A
git commit -m "${cur_date}"
git push origin master
