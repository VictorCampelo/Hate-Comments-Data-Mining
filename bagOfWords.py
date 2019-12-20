
import pandas as pd
import string
import re

from sklearn.feature_extraction.text import CountVectorizer

bagOfWords = []

ref_arquivo = open("hate.txt","r")
frases = []
for line in ref_arquivo:
	test = line.split("'")
	regex = r"((a)a+|(e)e+|(i)i+|(o)o+|(u)u+|(b)b+|(c)c+|(d)d+|(e)e+|(f)f+|(g)g+|(h)h+|(j)j+|(k)k+|(l)l+|(m)m+|(n)n+|(p)p+|(q)q+|(r)r+|(s)s+|(t)t+|(u)u+|(v)v+|(x)x+|(z)z+)"
	subst = "\\2\\3\\4\\5\\6\\7\\8\\9\\10\\11\\12\\13\\14\\15\\16\\17\\18\\19\\20\\21\\22\\23\\24\\25\\26\\27"

	result = re.sub(regex, subst, test[1].lower(), 0, re.IGNORECASE)
	frases.append(result)

documents = frases

count_vector = CountVectorizer(documents)

print(count_vector)

count_vector.fit(documents)
count_vector.get_feature_names()

doc_array = count_vector.transform(documents).toarray()
doc_array

frequency_matrix = pd.DataFrame(doc_array,index=documents,columns=count_vector.get_feature_names())
print(frequency_matrix)

ref_arquivo.close()