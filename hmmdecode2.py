import numpy as np
import string
import ujson
import json
import math
import string
import sys

#file=sys.argv[1]

# use it_isdt_dev_tagged.txt as a training model and it_isdt_dev_raw.txt to predict the tags

def cal_transition_prob(previous_tag,current_tag):
    if (previous_tag,current_tag) not in transition_prob:
        return 1/(tag_count[previous_tag] + no_of_tags)
    return transition_prob[(previous_tag,current_tag)]

def cal_emission_prob(current_word,current_tag):
    #check unknown words
    # if current_word not in poss_words:
    #     return 1
    if (current_word,current_tag) not in emission_prob:
        return 1e-7
    return emission_prob[(current_word,current_tag)]

def Viterbi(words_lst,poss_tags,poss_words):
    
    viterbi={}
    backpointer={}
    
    for tag in poss_tags:   # initialisation step
         if words_lst[0] in poss_words:
            viterbi[tag,0] =  math.log(cal_transition_prob('Start',tag)) + math.log(cal_emission_prob(words_lst[0],tag))            # probability that first word has a tag V,N,....etc
            # backpointer[tag,0] = 0
         else:
            viterbi[tag,0] =  math.log(cal_transition_prob('Start',tag))
    
    for t in range(1,len(words_lst)):
        word=words_lst[t]
        # if word not in poss_words:
        #     for tag1 in poss_tags:
        #         #print (tag1)
        #         lst=[(viterbi[new_tag1, t - 1] + math.log (cal_transition_prob(new_tag1,tag1)) + math.log(cal_emission_prob(word,tag1)), new_tag1) for new_tag1 in open_class_tags]
        #         new_tag1 = sorted(lst)[-1][1]   # one with the max probability
        #         viterbi[tag1,t] = viterbi[new_tag1,t-1] + math.log(cal_transition_prob(new_tag1,tag1)) + math.log(cal_emission_prob(word,tag1))
        # else:
        for tag in poss_tags:
            if word not in poss_words:
                for new_tag in open_class_tags:
                    lst=[(viterbi[new_tag, t - 1] + math.log (cal_transition_prob(new_tag,tag)), new_tag)]
                new_tag = sorted(lst)[-1][1]   # one with the max probability
                viterbi[tag,t] = viterbi[new_tag,t-1] + math.log(cal_transition_prob(new_tag,tag))
            else:
                for new_tag in poss_tags:
                    lst=[(viterbi[new_tag, t - 1] + math.log (cal_transition_prob(new_tag,tag)) + math.log(cal_emission_prob(word,tag)), new_tag)]
                new_tag = sorted(lst)[-1][1]   # one with the max probability
                viterbi[tag,t] = viterbi[new_tag,t-1] + math.log(cal_transition_prob(new_tag,tag)) + math.log(cal_emission_prob(word,tag))
               
                

    #decoding step
    best_states = []
    for t in range(len(words_lst) - 1, -1, -1):
        k = sorted([(viterbi[k, t], k) for k in poss_tags])[-1][1]
        best_states.append((words_lst[t], k))
    best_states.reverse()
    return best_states
  


# def Viterbi2(words_lst,poss_tags):
#     pred_tags=[]
#     storing_values = {}
#     for q in range(len(s)):
#         step=words_lst[0]
#         if q==1:
#             storing_values[q]={}



with open("D:/USCFall/NLP/HW3/it_isdt_dev_raw.txt", encoding='utf-8') as f:
    lines = f.read().splitlines()


# reading the trasnition and emission probabilities from the trained model

with open("D:/USCFall/NLP/HW3/hmmmodel.txt",encoding='utf-8') as file:
    training_dic = file.read()


training_dic=eval(training_dic)

transition_prob= training_dic['transition_probability']  #transition prob matrix
emission_prob=training_dic['emission_probability']  #emission prob matrix
tag_count=training_dic['tag_count']
no_of_tags=training_dic['total_tag_count']
poss_words=set(training_dic['word_lst'])
poss_tags=training_dic['tag_lst']

lst=list(emission_prob.keys())
unique_tag_word={}
visited=set()
for val in lst:
    tg=val[-1]
    if tg not in unique_tag_word:
        unique_tag_word[tg]=1
        visited.add(val[0])
    elif tg in unique_tag_word and val[0] not in visited:
        unique_tag_word[tg]+=1
        visited.add(val[0])
unique_tag_word=dict(sorted(unique_tag_word.items(), key=lambda item: item[1],reverse=True))
print (unique_tag_word)
# total=0
# for val in unique_tag_word:
#     total+=unique_tag_word[val]
# avg_val=int(total/len(unique_tag_word))

open_class_tags=[]
cnt=0
for val in unique_tag_word:
        if cnt<=2:
            open_class_tags.append(val)
        cnt+=1
print (open_class_tags)
# tag_count_dic={}
# lst=list(transition_prob.keys())
# poss_tags=[]
# for val in lst:
#     poss_tags.append(val[0])
#     poss_tags.append(val[1])
#     if val[0] not in tag_count_dic:
#         tag_count_dic[val[0]]=1
#     else:
#         tag_count_dic[val[0]]+=1
#     if val[1] not in tag_count_dic:
#         tag_count_dic[val[1]]=1
#     else:
#         tag_count_dic[val[1]]+=1

# poss_tags=list(set(poss_tags))  # list of all possible tags from the trained corpus

# poss_tags.remove('Start')

# #print (tag_count_dic)
# lst=list(emission_prob.keys())
# word_to=[]
# for val in lst:
#     word_to.append(val[0])
#     #word_to.append(val[1])
# poss_words=set(word_to)

write_lst=[]
for sentence in lines:
    words_lst=sentence.split(' ')
    best_states=Viterbi(words_lst,poss_tags,poss_words)
    s=""
    for val in best_states:
        s+=val[0]+'/'+val[1]+" "
    write_lst.append(s)


with open("D:/USCFall/NLP/HW3/hmmoutput.txt", 'w', encoding='utf-8') as f:
    for line in write_lst:
        f.write(f"{line}\n")


