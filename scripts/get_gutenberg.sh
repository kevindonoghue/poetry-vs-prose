#! /bin/bash

cd ../data

for i in {0..9}
do
	wget -r -np -R 'index.html*' -A 'txt'  http://gutenberg.readingroo.ms/$i/
done

mv gutenberg.readingroo.ms gutenberg_documents