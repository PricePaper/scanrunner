#!/bin/sh

while /bin/true
do
   su -l -c "cd /scanner; /docscanner.py --stats ." scanner
   sleep 300
done
