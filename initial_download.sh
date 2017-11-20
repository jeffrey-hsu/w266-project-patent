#!/bin/bash

cd
mkdir ~/patent_data
echo Downloading dataset...
wget https://bulkdata.uspto.gov/data/patent/pair/economics/2016/csv.zip -P ~/patent_data/2016_csv.zip
echo Zip file downloaded.
echo Unzipping file ... 
unzip ~/patent_data/2016_csv.zip -d ~/patent_data/dataset
echo Finished unzipping file.
