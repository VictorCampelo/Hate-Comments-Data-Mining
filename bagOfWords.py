# coding: utf-8
import pandas as pd
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
import spacy

def main():

	bagOfWords = []
	dictionary = []
	dictionary_StW = []
	badLanguage = []
	documents = []
	inf = []
	

	wrds 			=open("words_dictionary.txt","r")
	stopwords 		=open("stopwords.txt","r")
	badL 			=open("listBlock.txt","r")
	ref_arquivo 	=open("hate.txt","r")

	for wrd in wrds:
		dictionary.append(wrd)	
	for wrd in stopwords:
		dictionary_StW.append(wrd)
	for wrd in badL:
		badLanguage.append(wrd.lower())
	badLanguage = ' '.join(badLanguage)
	for line in ref_arquivo:
		test = line.split("'")
		regex = r"((a)a+|(e)e+|(i)i+|(o)o+|(u)u+|(b)b+|(c)c+|(d)d+|(e)e+|(f)f+|(g)g+|(h)h+|(j)j+|(k)k+|(l)l+|(m)m+|(n)n+|(p)p+|(q)q+|(r)rr+|(s)ss+|(t)t+|(u)u+|(v)v+|(x)x+|(z)z+)"
		subst = "\\2\\3\\4\\5\\6\\7\\8\\9\\10\\11\\12\\13\\14\\15\\16\\17\\18\\19\\20\\21\\22\\23\\24\\25\\26\\27"
		result = re.sub(regex, subst, test[1].lower(), 0, re.IGNORECASE)
		documents.append(result)
		inf.append(test[0].lower())

	documents = list(dict.fromkeys(documents))

	count_vector = CountVectorizer(documents)

	#print(count_vector)

	count_vector.fit(documents)
	count_vector.get_feature_names() #tokens

	#remove stopwords
	i = 0
	for token in count_vector.get_feature_names():
		if token in dictionary_StW:
			del count_vector.get_feature_names()[i]
		i+=1	

	doc_array = count_vector.transform(documents).toarray()
	#print(doc_array)

	#frequency_matrix = pd.DataFrame(doc_array,index=documents,columns=count_vector.get_feature_names())
	#print(frequency_matrix)
	
	#select the 500 word
	#first: create a list of tuple
	list_token = []
	for j in range(0, len(count_vector.get_feature_names())-1):
		count = 0
		for i in range(0, len(documents)-1):
			if doc_array[i][j] == 1:
				count += 1
		t = (count_vector.get_feature_names()[j], count)
		list_token.append(t)
	#sort the list of tuple 
	list_token.sort(key=lambda x: x[1])	


	arff = open("data.arff","w+")
	#make arff head 
	arff.write("@relation hateComments\n\n")
	arff.write("@attribute @@class {no,yes}\n")
	arff.write("@attribute qtd_offensive numeric\n")
	arff.write("@attribute qtd_words numeric\n")
	arff.write("@attribute aver_size numeric\n")
	arff.write("@attribute amount_character numeric\n")
	for token in list_token[0:500]:
		arff.write("@attribute "+str(token[0])+" {true, false}\n")
	arff.write("\n")
	arff.write("@data\n")
	#def qtd character
	#def qtd words
	for i in range(0, len(documents)-1):
		array_true_false = []
		qtd_offen = 0
		qtd_words = 0
		aver_words = 0
		qtd_character = 0
		for j in range(0, len(count_vector.get_feature_names())-1):
			if doc_array[i][j] == 1:
				qtd_words += 1
				qtd_character += len(count_vector.get_feature_names()[j])
				if re.search("\\b"+str(count_vector.get_feature_names()[j]+"\\b"), badLanguage, re.IGNORECASE):
					qtd_offen += 1
		for j in range(0, len(list_token[0:500])):
			if doc_array[i][j] == 1:			
				array_true_false.append("true")
			else:
				array_true_false.append("false")
		try:
			aver_words = qtd_character/qtd_words
		except Exception as e:
			aver_words = 0
		arff.write(str(inf[i])+str(qtd_offen)+","+str(qtd_words)+","+"%.3f" % round(aver_words,3)+","+str(qtd_character)+","+','.join(array_true_false)+"\n")
	
	wrds.close()
	stopwords.close()
	badL.close()
	ref_arquivo.close()
	arff.close()
main()	