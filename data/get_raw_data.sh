#!/bin/bash

mkdir temp

echo fetch document ids
wget https://solr.idl.ucsf.edu/solr/ltdl3/document\?wt\=json\&indent\=true\&id\=mswj0233\&mlt.count\=1000000000 -O temp/ids.json

echo extract pdf file names
grep -F '.pdf' temp/ids.json | sed 's/.*\([^ ][^ ][^ ][^ ][^ ][^ ][^ ][^ ]\.pdf\).*/\1/' | sort | uniq > temp/pdf_files.txt

echo clean pdf file names
grep '^[a-z][a-z][a-z][a-z][0-9][0-9][0-9][0-9]\.pdf$' temp/pdf_files.txt > temp/pdf_files.clean.txt

echo filter only first folder
grep '^ff' temp/pdf_files.clean.txt > temp/pdf_files.clean.ff.txt

echo genereate links
awk '{split($0,a,""); print "https://s3-us-west-2.amazonaws.com/edu.ucsf.industrydocuments.artifacts/"a[1]"/"a[2]"/"a[3]"/"a[4]"/"a[1]a[2]a[3]a[4]a[5]a[6]a[7]a[8]"/"a[1]a[2]a[3]a[4]a[5]a[6]a[7]a[8]".pdf"}' temp/pdf_files.clean.ff.txt > temp/pdf_urls.txt

echo download documents
mkdir -p ./data/raw
cd ./data/raw || exit
parallel -j 100 wget -nc -x < ../../temp/pdf_urls.txt
cd ../../ || exit

echo clean up
rm -r temp
find ./data/raw -size 0 -print -delete
find ./data/raw -empty -type d -print -delete
