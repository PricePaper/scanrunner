#!/bin/sh

while /bin/true
do
   su -l -c "cd /scanner; python3 /read_inv.py ." scanner
   sleep 300
done
