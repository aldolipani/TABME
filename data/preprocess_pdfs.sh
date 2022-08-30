#!/bin/bash

echo make a copy of raw data
cp -r ./data/raw ./data/preprocessed

echo resize pdfs and remove id information
find ./data/preprocessed -name "*.pdf" -exec sh -c "convert -density 150 '{}' -colorspace Gray -resize 1025x1025 -gravity NorthWest -shave 25x25 '{}' || { echo {} will be deleted; rm {}; }" \;

echo filter pdfs with more than 20 pages
function len_pdf() {
  pdfinfo "$1" | grep -aF Pages | sed 's/Pages:[ ]*//g'
}

while read -r pdf_path; do
  if [[ $(len_pdf "$pdf_path") -gt 20 ]]; then
    echo "$pdf_path will be deleted"
    rm "$pdf_path"
  fi
done < <(find ./data/preprocessed -name "*.pdf")

echo generate training, validation and testing data 18:1:1
find ./data/preprocessed -name "*.pdf" > ./data/pdfs.txt
sort -R ./data/pdfs.txt > ./data/pdfs.random.txt
head -n $(( $(wc -l < ./data/pdfs.random.txt) * 9 / 10)) ./data/pdfs.random.txt > ./data/pdfs.train.txt
tail -n $(( $(wc -l < ./data/pdfs.random.txt) * 1 / 10 + 1)) ./data/pdfs.random.txt > ./data/pdfs.val_test.txt
head -n $(( $(wc -l < ./data/pdfs.val_test.txt) * 1 / 2)) ./data/pdfs.val_test.txt > ./data/pdfs.val.txt
tail -n $(( $(wc -l < ./data/pdfs.val_test.txt) * 1 / 2)) ./data/pdfs.val_test.txt > ./data/pdfs.test.txt

mkdir -p ./data/train
xargs -I {} cp {} ./data/train/ < ./data/pdfs.train.txt
mkdir -p ./data/val
xargs -I {} cp {} ./data/val/ < ./data/pdfs.val.txt
mkdir -p ./data/test
xargs -I {} cp {} ./data/test/ < ./data/pdfs.test.txt

rm ./data/pdfs.txt
rm ./data/pdfs.random.txt
rm ./data/pdfs.train.txt
rm ./data/pdfs.val_test.txt
rm ./data/pdfs.val.txt
rm ./data/pdfs.test.txt

echo convert pdfs to jpg folders
while read -r pdf_path; do
  pdf_name=$(basename "$pdf_path")
  pdf_name="${pdf_name%.*}"
  mkdir -p ./data/train/"$pdf_name"
  convert -density 150 "$pdf_path" -resize 1000x1000 "./data/train/$pdf_name/$pdf_name.jpg"
  rm "$pdf_path"
done < <(find ./data/train -name "*.pdf")

while read -r pdf_path; do
  pdf_name=$(basename "$pdf_path")
  pdf_name="${pdf_name%.*}"
  mkdir -p ./data/val/"$pdf_name"
  convert -density 150 "$pdf_path" -resize 1000x1000 "./data/val/$pdf_name/$pdf_name.jpg"
  rm "$pdf_path"
done < <(find ./data/val -name "*.pdf")

while read -r pdf_path; do
  pdf_name=$(basename "$pdf_path")
  pdf_name="${pdf_name%.*}"
  mkdir -p ./data/test/"$pdf_name"
  convert -density 150 "$pdf_path" -resize 1000x1000 "./data/test/$pdf_name/$pdf_name.jpg"
  rm "$pdf_path"
done < <(find ./data/test -name "*.pdf")

echo get OCR using tesseract
export filter="./data/ocr_filter.awk"
while read -r pdf_path; do
  pdf_name="${pdf_path%.*}"
  tesseract -l eng --dpi 300 "$pdf_path" stdout tsv 2> /dev/null | $filter > "$pdf_name.tsv"
done < <(find ./data/train -name "*.jpg")

while read -r pdf_path; do
  pdf_name="${pdf_path%.*}"
  tesseract -l eng --dpi 300 "$pdf_path" stdout tsv 2> /dev/null | $filter > "$pdf_name.tsv"
done < <(find ./data/val -name "*.jpg")

while read -r pdf_path; do
  pdf_name="${pdf_path%.*}"
  tesseract -l eng --dpi 300 "$pdf_path" stdout tsv 2> /dev/null | $filter > "$pdf_name.tsv"
done < <(find ./data/test -name "*.jpg")

echo generate virtual folders
python data/sample_folders.py ./data/train 11 100000 > ./data/train_folders.txt
python data/sample_folders.py ./data/val 11 5000 > ./data/val_folders.txt
python data/sample_folders.py ./data/test 11 5000 > ./data/test_folders.txt

echo clean up
rm -r ./data/preprocessed
