from re import S
import numpy as np
import string
import pickle
import json
import sys


with open("D:/USCFall/NLP/HW3/it_isdt_train_tagged.txt",encoding='utf-8') as f:
    lines = f.read().splitlines()

poss_tags=[]    
for val in lines:
    lst=val.split(' ')   # getting total number of tags
    for var in lst:
        tag=var.split('/')[-1]
        poss_tags.append(tag)

poss_tags=list(set(poss_tags))
no_of_tags=len(poss_tags)   #Number of unique tags in the training set

words=[]
for val in lines:
    lst=val.split(' ')
    for var in lst:
        word=var.rsplit('/', 1)[0]
        words.append(word)

poss_words=list(set(words))  #unique words

no_of_words=len(poss_words)     #Number of unique words in the training set

emission_dic={}

for line in lines:
    lst=line.split(' ')
    for val in lst:
        word=val.rsplit('/', 1)[0]
        tag=val.split('/')[-1]
        if tag not in emission_dic:
                emission_dic[tag]={}
                emission_dic[tag][word]=1
        elif tag in emission_dic and word not in emission_dic[tag]:
                emission_dic[tag][word]=1
        else:
                emission_dic[tag][word]+=1


#Calculating emission probabilities: Count(word,tag)/Count(tag) in the corpus
tag_dic={}
for tag in emission_dic:
    word_dic=emission_dic[tag]
    count=0
    for word in word_dic:
        count+=word_dic[word]
    tag_dic[tag]=count  # counting the number of times a tag occurs in the dataset


emission_prob={}  #contains the emission prob
for tag in emission_dic:
    for word in emission_dic[tag]:
        #emission_prob[((word,tag))]= ((emission_dic[tag][word]) +1) / ((tag_dic[tag]) + no_of_tags) 
        emission_prob[((word,tag))]= ((emission_dic[tag][word])) / ((tag_dic[tag]))


# Calculating the transition probabilities P(ti | ti-1)

#Adding a tag at the start for calculation of the starting tag P (tag | start)


transition_dic={}
for line in lines:
    lst=line.split(' ')
    prev_tag='Start'
    # print (line)
    for val in lst:
        tag=val.split('/')[-1]
        if prev_tag not in transition_dic:
                        transition_dic[prev_tag]={}
                        transition_dic[prev_tag][(prev_tag,tag)]=1
        elif prev_tag in transition_dic and (prev_tag,tag) not in transition_dic[prev_tag]:
                        transition_dic[prev_tag][(prev_tag,tag)]=1
        else:
                        transition_dic[prev_tag][(prev_tag,tag)]+=1
        prev_tag=tag

    if prev_tag not in transition_dic:
        transition_dic[prev_tag]={}
        transition_dic[prev_tag][(prev_tag,'End')]=1  
    elif prev_tag in transition_dic and (prev_tag,'End') not in transition_dic[prev_tag]:
        transition_dic[prev_tag][(prev_tag,'End')]=1 
    else:
         transition_dic[prev_tag][(prev_tag,'End')]+=1     

# Calculating transition probabilities:
prev_tag_dic={}
for prev_tag in transition_dic:
    count=0
    tmp_dic=transition_dic[prev_tag]
    for val in tmp_dic:
        count+=tmp_dic[val]
    prev_tag_dic[prev_tag]=count


transition_prob={}  #contains the transition prob
for tag in transition_dic:
    for prev_tag in transition_dic[tag]:
        #transition_prob[prev_tag]=(transition_dic[tag][prev_tag] + 1 )/(prev_tag_dic[tag]+ no_of_tags)   # smoothing
        transition_prob[prev_tag]=(transition_dic[tag][prev_tag] )/(prev_tag_dic[tag]) 


# print (transition_dic)
# print (prev_tag_dic)
# print (transition_prob)
# print (emission_prob)
training_dic={}
training_dic['transition_probability']=transition_prob
training_dic['emission_probability']=emission_prob
training_dic['tag_count']=prev_tag_dic
training_dic['total_tag_count']=no_of_tags
training_dic['word_lst']=poss_words
training_dic['tag_lst']=poss_tags

with open('D:/USCFall/NLP/HW3/hmmmodel.txt', 'w',encoding='utf-8') as file:
     file.write(str(training_dic))

#print (emission_dic)
#print (emission_prob)

#print (str("'s"))
