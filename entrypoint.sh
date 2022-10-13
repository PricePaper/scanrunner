#!/bin/sh

while /bin/true
do
   su -l -c "cd /scanner; python3 /docscanner.py ." scanner
   sleep 300
done
