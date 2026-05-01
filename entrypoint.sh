#!/bin/sh

while /bin/true
do
   su -l -c "cd /scanner; nice -n 10 /docscanner.py --stats ." scanner
   sleep 300
done
