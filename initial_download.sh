#!/bin/bash

cd
mkdir ~/patent_data
echo Downloading dataset...
wget https://bulkdata.uspto.gov/data/patent/claims/economics/2014/patent_claims_fulltext.csv.zip -P ~/patent_data/
echo Zip file downloaded.
echo Unzipping file ... 
unzip ~/patent_data/patent_claims_fulltext.csv.zip -d ~/patent_data/
rm -f ~patent_data/patent_claims_fulltext.csv.zip
echo Finished unzipping file.
