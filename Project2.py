# Venkata Naga Satya Sai Vineeth Kondisetty
# MavID: 1001772021

import os
import operator
import time

location = '20_newsgroups'
all_files = []
final_probability = {}
cnt = 0
train_dic = {}
test_dict = {}
folder_list = []
fileList = []
folder_words_count= {}
word_dict = {}
prob = {}
probabilities = {}
a = 0
count = 0
start = time.process_time()
def list_of_categories():
    print("The following are the list of Categories:")
    for filenames in (os.listdir(location)):
        print(filenames)
    print("\n------------------------------ Training data now.... ------------------------------")
        
def data_processing(text):
    
    text = text.lower()
    punctuations = ['~','`','!','@','^','$','%','&','*','(',')','+','=','{','}','[',']',';',':','|','\\','"',"'",'\n','<','>',',','.','?','/','-','*']
    for x in text:
        if x in punctuations:
            text = text.replace(x, " ")
    stop_words = [ 'about', 'above', 'after', 'again', 'against', 'all', 'am', "an", 'and', 'any', 'are', "aren't", 'as', 'at',
 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 
 'can', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during',
 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's",
 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's",
  "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself',
 "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself',
 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours' 'ourselves', 'out', 'over', 'own',
 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 
 'than', 'that',"that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", 
 "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 
 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where',
 "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's",'will', 'with', "won't", 'would', "wouldn't", 
 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', 
 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'hundred', 'thousand', '1st', '2nd', '3rd',
 '4th', '5th', '6th', '7th', '8th', '9th', '10th']
    for j  in text.split():
        if j in stop_words:
            text = text.replace(j,"")   
    return text


def traindict(line):
    list_of_words = set()
    list_of_words.update(data_processing(line).split()) 
    for wrd in list_of_words:
        if wrd in word_dict: 
            word_dict[wrd] = word_dict[wrd] + 1
        else:
            word_dict[wrd] = 1
    return word_dict


def test_data_process(filename,subfile):
    with open(os.path.join(os.path.join(location, filename), subfile), encoding="utf-8",errors='ignore') as ind:
        words_set = set()
        for lines in ind:
            words_set.update(data_processing(lines).split())
    return words_set


def testdata_prob(k):
    probability = prob[k]
    for w in words_set:
        if (w, k) in final_probability:
            probability = (probability * final_probability[(w, k)]) * 100           
        else:
             probability = (probability * 0.01 * number / folder_words_count[k])
    probabilities[k] = probability 
    return probabilities

if __name__ == "__main__":
    list_of_categories()

for className in (os.listdir(location)):
    fileList.append(className)
    for txt_file in os.listdir(os.path.join(location, className)):
        folder_list.append(txt_file)
        cnt += 1
    number = len(folder_list)//2
    train_dic[className] = folder_list[0:number]   # Training dataset
    test_dict[className] = folder_list[number:]   #Testing dataset
    all_files.append(len(folder_list))
    folder_list = []
for i in range(0, len(all_files)):
    prob[(fileList[i])] = (all_files[i] / cnt)
for i in train_dic:
    for j in train_dic[i]:
        with open(os.path.join(os.path.join(location, i), j), encoding="utf-8", errors='ignore') as lines:
            for line in lines:
                word_dict = traindict(line)
    folder_words_count[i] = len(word_dict.values())
    x = word_dict.keys() 
    for w in (x):
        if word_dict[w] != 0:
            final_probability[(w, i)] = ((word_dict[w] / folder_words_count[i]) * number)
    else:
        final_probability[(w, i)] = ((0.01 / folder_words_count[i]) * number)
    word_dict = {}
print("Data has been trained")
print("Training time:",time.process_time()-start)
print("-------------------------- Testing Data now.... ----------------------------------")
test = time.process_time()
for i in test_dict:
    for j in test_dict[i]:
        count = count + 1
        words_set = set()
        words_set.update(test_data_process(i,j)) 
        for k in train_dic:
            probabilities=testdata_prob(k)
        if max(probabilities.items(), key=operator.itemgetter(1))[0] == i:
            a += 1
        else: None

print("Testing time:",time.process_time()-test)
print("Total execution time:",time.process_time()-start)
print("---------------------------------------------------------------------------------")
print("The accuracy in classifying the documents is: ",str(a/count*100) + "%")