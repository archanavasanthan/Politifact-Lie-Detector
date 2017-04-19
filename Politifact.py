# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 22:15:31 2017

@author: archa
"""

from bs4 import BeautifulSoup
from operator import itemgetter
import re
import requests
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import VotingClassifier
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import FreqDist
import matplotlib
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier

import io

def getFacts():
    Names=[]
    Facts=[]
    Results=[]
    ActualResults=[]
    Months=[]
    page=None
    vName=''
    vFact=''
    vResult=''
    resultText=''
    vMonth=''
    vActualResult=''
    fDataset = io.open('Dataset.txt','w',encoding='utf-8')
    values=['true','mostly-true','barely-true','false','pants-fire']
    for one in values:
        print (one)
        pageurl='http://www.politifact.com/truth-o-meter/rulings/'+one+'/'
        for i in range(1):
            try:
                response=requests.get(pageurl,headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36', })
                html=response.content 
                break 
            except Exception as e:
                print ('failed attempt',i)
                if not html:
                    continue
        soup = BeautifulSoup(html.decode('ascii', 'ignore'),'lxml') 
        page1=soup.find('span',{'class':'step-links__current'})
        page2=page1.text
        page=page2[23:26]
        aa=int(page)
        print (aa)
        for every in range(1,aa):
            print (every)
            purl='http://www.politifact.com/truth-o-meter/rulings/'+one+'/?page='+str(every)
            for i in range(1):
                try:
                    response=requests.get(purl,headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36', })
                    htm=response.content
                    break 
                except Exception as e:
                    print ('failed attempt',i)
                    if not htm:
                        continue
            sou = BeautifulSoup(htm.decode('ascii', 'ignore'),'lxml')       
            allreviews=sou.findAll('div',{'class':'scoretable__item'})
            for each in allreviews:
                Name1=each.find('p',{'class':'statement__source'})
                try:                
                    if Name1.text:
                        vName=Name1.text
                except Exception as a:
                        vName='NA'
                Fact1=each.find('p',{'class':'statement__text'})
                try:                
                    if Fact1.text.strip():
                        vFact=Fact1.text.strip()
                except Exception as a:
                        vFact='NA'
                       
                Result1=each.find('img',{'src':re.compile('http://static.politifact.com.s3.amazonaws.com/rulings/')})
                resultText=Result1['alt'].strip()                
                if (resultText == 'Mostly True'):
                    vResult='True'
                elif (resultText == 'Pants on Fire!'):
                    vResult='False'
                elif (resultText =='Mostly False'):
                    vResult='False'
                else:
                    vResult=resultText
                
                
                Date1=each.find('span',{'class':'article__meta'})
                try:                
                    if Date1.text:
                        vDate=Date1.text.strip()
                        parts=vDate.split(",")
                        vMonth=parts.split(" ")[1]
                except Exception as a:
                        vMonth='NA'   
                        
                #Clean Data
                vFact=cleanData(vFact)
                vName=cleanData(vName)
                vResult=cleanData(vResult)
                vMonth=cleanData(vMonth)
                vActualResult=cleanData(resultText)
                
                #Append elements to a list
                Results.append(vResult)
                Facts.append(vFact)
                Names.append(vName)
                Months.append(vMonth)
                ActualResults.append(vActualResult)
                
                #fDataset.write(str(vFact)+'\n')                
                fDataset.write(vName+'\t'+vFact+'\t'+vResult+'\t'+vMonth+'\t'+vActualResult+'\n')
    
    fDataset.close()
    print(Names[0])
    print(Results[0])
    print(Facts[0])
    print(Months[0])
    print(ActualResults[0])
    return Names, Facts, Results,Months,ActualResults

def cleanData(data):
        data=data.lower().strip()
        data=re.sub('[^a-zA-Z\d]',' ',data)
        return data

def filterData(path,values):
    Names=[]
    Facts=[]
    Results=[]
    Months=[]  
    ActualResults=[]
    FNames=[]
    FFacts=[]
    FResults=[]
    FMonths=[]  
    count=0
    stopWords=set(stopwords.words('english'))
    
    fDataset = io.open('Dataset.txt','r',encoding='utf-8')
    lines=fDataset.readlines()
    for line in lines:
        name,fact,result,month,actualResult=line.strip().split("\t")
        Results.append(result)
        Facts.append(fact)
        Names.append(name)
        Months.append(month)
        ActualResults.append(actualResult)
        
    for name,fact,result,month,actualResult in list(zip(Names, Facts, Results,Months,ActualResults)):
      for value in values:  
        newFact=[]
        count=count+1
        if actualResult == value:
            words = word_tokenize(fact)
            for word in words:
                if word not in stopWords:
                    newFact.append(word)
        
            FNames.append(name)
            FFacts.append(" ".join(newFact))
            FResults.append(result)
            FMonths.append(month)    
    
    fDataset.close()       
    return FNames, FFacts, FResults,FMonths

def countResults(Results,values):
    #Count the number of True and False statements
    countTrue=0
    countFalse=0
    
    for line in Results:
        if line in ['true','mostly true'] : countTrue+=1
        elif line in['pants on fire','mostly false','false'] : countFalse+=1
    
    print("Count of True Statements:",countTrue)
    print("Count of False Statements:",countFalse)
    print("Total Number of Statements:",countTrue+countFalse)
    
    return countTrue,countFalse

def countName(Names,Results,countTrue,countFalse):
    #Count the number of unique Names,unique names in True and False
    uniqueName=set()
    uniqueTrueName=set()
    uniqueFalseName=set()
    
    for name,result in list(zip(Names,Results)):
        if name not in uniqueName:
            uniqueName.add(name)
        if name not in uniqueTrueName and result =='true' :
            uniqueTrueName.add(name)
        if name not in uniqueFalseName and result =='false':
            uniqueFalseName.add(name)
            
    print("Total Unique Number of Names in the data set:",len(uniqueName))  
    print("% Unique Names in True Statements:",len(uniqueTrueName)/countTrue)
    print("% Unique Names in False Statements:",len(uniqueFalseName)/countFalse)
    

def plotmaximumName(Names,Results):
    freqTrue={} 
    freqFalse={} 
        
    for name,result in list(zip(Names,Results)):
            if result=='true':
                if freqTrue.get(name) == None:
                    freqTrue[name] = 1        
                    #print(freqTrue[person])
                else: freqTrue[name] +=1
            elif result.strip()=='false':
                if freqFalse.get(name) == None:
                    freqFalse[name] = 1             
                else: freqFalse[name] +=1
    ##Plotting frequency of Top Ten Names for True Statements            
    trueTopName=[]    
    trueTopValue=[] 
    falseTopName=[]    
    falseTopValue=[] 
    #print(sortedTrue[:10])
    
    for w in sorted(freqTrue.items(),key=itemgetter(1),reverse=True):
        trueTopName.append(w[0])
        trueTopValue.append(w[1])
    #print(trueTopName[:10])
    #print(trueTopValue[:10])
    matplotlib.style.use('ggplot') 
    indexes = np.arange(len(trueTopName[:10]))
    width = 0.7
    plt.figure(1)
    plt.subplot(211)
    plt.bar(indexes, trueTopValue[:10], width,color='b',alpha=0.5)
    plt.xticks(indexes + width * 0.5, trueTopName[:10])
    plt.title("Frequency of Names for True Statements(Top 20)")

    ##Plotting frequency of Top Ten Names for False Statements            
    #print(sortedfalse[:10])
    for w in sorted(freqFalse.items(),key=itemgetter(1),reverse=True):
        falseTopName.append(w[0])
        falseTopValue.append(w[1])
    #print(falseTopName[:10])
    #print(falseTopValue[:10])
    indexes = np.arange(len(falseTopName[:10]))
    width = 0.7
    plt.subplot(212)
    plt.bar(indexes, falseTopValue[:10], width,alpha=0.5,color='r')
    plt.xticks(indexes + width * 0.5, falseTopName[:10])
    plt.title("Frequency of Names for False Statements(Top 10)")
    plt.subplot(212)
    plt.show()
    
def plotmaximumWords(Facts):
    TopWord=[]
    TopValue=[]
    words=[]
    for fact in Facts:
        for word in word_tokenize(fact):
            words.append(word)
    fdist = FreqDist(words)
    
    ##Plotting frequency of Top Fifty Words
    for w in sorted(fdist.items(),key=itemgetter(1),reverse=True):
        TopWord.append(w[0])
        TopValue.append(w[1])

    matplotlib.style.use('ggplot') 
    indexes = np.arange(len(TopWord[:15]))
    width = 0.5

    plt.bar(indexes,TopValue[:15], width,alpha=0.5,color='g')
    plt.xticks(indexes + width*0.5, TopWord[:15])
    plt.title("Top 15 Frequency Words")
    plt.show()

def plotimportantFeatures(feature_names,feature_importance,comb):
        TopFeature=[]
        TopValue=[]
        features_dict={}
        totalNum=0
        
        for name,importance in list(zip(feature_names,feature_importance)):
                    if features_dict.get(name) == None:
                        features_dict[name] = importance   
                    else: features_dict[name] += importance
           
        for w in sorted(features_dict.items(),key=itemgetter(1),reverse=True):         
               TopFeature.append(w[0])
               TopValue.append(w[1])
    
        ##Plotting Top 15 Important Features
        plt.figure()
        matplotlib.style.use('ggplot') 
        indexes = np.arange(len(TopFeature[:20]))
        width = 0.3

        plt.bar(indexes,TopValue[:20], width,alpha=0.7,color='g')
        plt.xticks(indexes + width*0.5, TopFeature[:20])
        plt.title("Top 15 Features with Importance - "+comb)
        plt.show()
        """
        ##Total number of features##
        totalNum=len(feature_names)
        print("nTotal Number of features: ",totalNum)
        print("Top 10 Features:")
        for i in range(0,11,1):
            print(TopFeature[i]+ ": "+str(TopValue[i]))
            
        print("\nLast 10 Features:")
        for i in range(totalNum-1,totalNum-11,-1):
            print(TopFeature[i]+ ": "+str(TopValue[i]))
        """

def CountVectorizerTrain(training_data,test_data,min_df_value):
        feature_names=[]
        counter=CountVectorizer(min_df=min_df_value)
        counter.fit(training_data)
        counts_train=counter.transform(training_data)
        counts_test=counter.transform(test_data)  
        feature_names=counter.get_feature_names()
        return counts_train,counts_test,feature_names
        
def NaiveBayesAlgo(counts_train,counts_test,training_labels,test_labels,comb):
        clf=MultinomialNB()
        clf.fit(counts_train,training_labels)
        pred=clf.predict(counts_test)   
        print ("Accuracy of NB model - "+comb+"= ",accuracy_score(pred,test_labels)*100,"%")

def LogisticRegAlgo(counts_train,counts_test,training_labels,test_labels,comb):
        LREG_classifier=LogisticRegression(random_state=123)
        LREG_classifier.fit(counts_train,training_labels)
        pred=LREG_classifier.predict(counts_test)
        print ("Accuracy of Logistic model-"+comb+"= ",accuracy_score(pred,test_labels)*100,"%")  

def decisionTreeAlgo(counts_train,counts_test,training_labels,test_labels,comb):
        DT_classifier = DecisionTreeClassifier(max_depth=8)
        DT_classifier.fit(counts_train,training_labels)
        pred=DT_classifier.predict(counts_test)
        print ("Accuracy of DT model- "+comb+"= ",accuracy_score(pred,test_labels)*100,"%")
 
def KNNAlgo(counts_train,counts_test,training_labels,test_labels,comb):
        KNN_classifier=KNeighborsClassifier(n=5)
        KNN_classifier.fit(counts_train,training_labels)
        pred=KNN_classifier.predict(counts_test)
        print ("Accuracy of KNN model- "+comb+"= ",accuracy_score(pred,test_labels)*100,"%")

def RandomForestAlgo(counts_train,counts_test,training_labels,test_labels,comb):
        RForest_classifier=RandomForestClassifier(n_estimators=3)
        RForest_classifier.fit(counts_train,training_labels)
        pred=RForest_classifier.predict(counts_test)
        print ("Accuracy of RandomForest model- "+comb+"= ",accuracy_score(pred,test_labels)*100,"%")
        
        return RForest_classifier.feature_importances_
            
if __name__=="__main__":

    Names=[]
    Facts=[]
    Results=[]
    Months=[]
    ActualResults=[]
    nameFact=[]
    countTrue=0
    countFalse=0
    feature_names=[]
    feature_importance=[]
    """  Remove triple quotes to create Data file for the first run and add them back before 2nd run
    ####################Creation and cleaning of Data file##################################
    #Data file 'Dataset.txt' is cretated by scraping statements with verdict ['true','mostly true','pants on fire','mostly false','false']
    #Data is also cleaned to keep characters and digits, which are converted to lower case
    Names, Facts, Results, Months, ActualResults=getFacts()
    print(np.shape(Names))
    print(np.shape(Facts))
    print(np.shape(Results))
    print(np.shape(Months))
    print(np.shape(ActualResults))
    Remove below triple quotes to create Data file for the first run and add them back before 2nd run """
    """
    ####################Exploratory Analysis of data set##################################
    values=['true','mostly true','pants on fire','mostly false','false']
    print("Exploratory Analysis on entire data set with values=['true','mostly true','pants on fire','mostly false','false']")
    Names, Facts, Results, Months=filterData('Dataset.txt',values)
    countTrue,countFalse = countResults(Results,values)
    plotmaximumName(Names,Results)
    plotmaximumWords(Facts)
    countName(Names,Results,countTrue,countFalse)
    """
    
    ############################Building of Models########################################
    ####Scenario 1: values=['true','false']
    #values=['true','mostly true','pants on fire','mostly false','false']
    values=['true','pants on fire']
    Names, Facts, Results, Months=filterData('Dataset.txt',values)
    print("\n\n")         
    print("##############Scenario 1: Statements=[true,pants on fire]##############")
    countTrue,countFalse = countResults(Results,values)
    print("\n")
    #Creating Training and Test data sets only using Facts as is
    training_data, test_data, training_labels, test_labels = train_test_split(Facts, Results, test_size=0.30, random_state=12)    
    
    #Using Count Vectorizer to covert the words into a matrix
    counts_train,counts_test,feature_names=CountVectorizerTrain(training_data,test_data,0.0000) 
    
    #Naive Bayes Algorithm
    NaiveBayesAlgo(counts_train,counts_test,training_labels,test_labels,'Only Facts')
    
    #Creating Training and Test data sets only using Name and Facts as is
    for name,fact in list(zip(Names,Facts)):
        vcombined=''
        #vcombined=name+'\t'+fact
        vcombined=name.replace(" ","")+' '+fact
        nameFact.append(vcombined) 
        
    #Creating Training and Test data sets only using Names and Facts 
    training_data, test_data, training_labels, test_labels = train_test_split(nameFact, Results, test_size=0.30, random_state=12)    

    #Using Count Vectorizer to covert the words into a matrix
    counts_train,counts_test,feature_names=CountVectorizerTrain(training_data,test_data,0.0000)    
    
    #Naive Bayes Algorithm
    NaiveBayesAlgo(counts_train,counts_test,training_labels,test_labels,'Names and Facts')
    
    #Creating Training and Test data sets using Name Identifier and Facts as is
    nameFact=[]
    for name,fact in list(zip(Names,Facts)):
        vcombined=''
        vcombined='p'+name.replace(" ","")+'\t'+fact+' p'+name.replace(" ","")+' p'+name.replace(" ","")
        nameFact.append(vcombined) 
    
    #Creating Training and Test data sets only using Name Identifier and Facts
    training_data, test_data, training_labels, test_labels = train_test_split(nameFact, Results, test_size=0.30, random_state=12)    
    
    #Using Count Vectorizer to covert the words into a matrix
    counts_train,counts_test,feature_names=CountVectorizerTrain(training_data,test_data,0.0000)    
    
    #Naive Bayes Algorithm
    NaiveBayesAlgo(counts_train,counts_test,training_labels,test_labels,'Name Identifier and Facts')

    #Logistic Regression
    LogisticRegAlgo(counts_train,counts_test,training_labels,test_labels,'Name Identifier and Facts')
    
    #KNN Classifier
    KNNAlgo(counts_train,counts_test,training_labels,test_labels,'Name Identifier and Facts')
    
    #Decision Trees
    decisionTreeAlgo(counts_train,counts_test,training_labels,test_labels,'Name Identifier and Facts')
    
    #Random Forest
    feature_importance=RandomForestAlgo(counts_train,counts_test,training_labels,test_labels,'Name Identifier and Facts')
    plotimportantFeatures(feature_names,feature_importance,'[true,pants on fire]')
    
    ###########Scenario 2: values=['true','mostly true','pants on fire','mostly false']
    print("\n\n")          
    print("##############Scenario 2: Statements=['true','mostly true','pants on fire','mostly false']##############")
    values=['true','mostly true','pants on fire','mostly false']
    Names, Facts, Results, Months=filterData('Dataset.txt',values)
    countTrue,countFalse = countResults(Results,values)
    
    #Creating Training and Test data sets only using Name and Facts as is
    nameFact=[]
    for name,fact in list(zip(Names,Facts)):
        vcombined=''
        #vcombined=name+'\t'+fact
        vcombined=name.replace(" ","")+' '+fact
        nameFact.append(vcombined) 
    
    training_data, test_data, training_labels, test_labels = train_test_split(nameFact, Results, test_size=0.30, random_state=12)    
    
    #Using Count Vectorizer to covert the words into a matrix
    counts_train,counts_test,feature_names=CountVectorizerTrain(training_data,test_data,0.0000)  
        
    #Naive Bayes Algorithm
    NaiveBayesAlgo(counts_train,counts_test,training_labels,test_labels,'Names and Facts')
    
    #Creating Training and Test data sets using Name Identifier and Facts as is
    nameFact=[]
    for name,fact in list(zip(Names,Facts)):
        vcombined=''
        vcombined='p'+name.replace(" ","")+' '+fact+' p'+name.replace(" ","")+' p'+name.replace(" ","")
        nameFact.append(vcombined) 

    #Creating Training and Test data sets only using Name Identifier and Facts
    training_data, test_data, training_labels, test_labels = train_test_split(nameFact, Results, test_size=0.30, random_state=12)    
    
    #Using Count Vectorizer to covert the words into a matrix
    counts_train,counts_test,feature_names=CountVectorizerTrain(training_data,test_data,0.0000)  
        
    #Naive Bayes Algorithm
    NaiveBayesAlgo(counts_train,counts_test,training_labels,test_labels,'Name Identifier and Facts')

    
    #Logistic Regression
    LogisticRegAlgo(counts_train,counts_test,training_labels,test_labels,'Name Identifier and Facts')
    
    #KNN Classifier
    KNNAlgo(counts_train,counts_test,training_labels,test_labels,'Name Identifier and Facts')
    
    ###Decision Trees
    decisionTreeAlgo(counts_train,counts_test,training_labels,test_labels,'Name Identifier and Facts')
    
    #Random Forest
    feature_importance=RandomForestAlgo(counts_train,counts_test,training_labels,test_labels,'Name Identifier and Facts')
    plotimportantFeatures(feature_names,feature_importance,'[true,mostly true,pants on fire,mostly false]')
    
    ###########Scenario 3: values=['true','false','mostly true','pants on fire','false']
    print("\n\n")      
    print("##############Scenario 3: Statements=['true','mostly true','pants on fire','false']##############")
    values=['true','mostly true','pants on fire','false']
    Names, Facts, Results, Months=filterData('Dataset.txt',values)
    countTrue,countFalse = countResults(Results,values)
    
    #Creating Training and Test data sets only using Name and Facts as is
    nameFact=[]
    for name,fact in list(zip(Names,Facts)):
        vcombined=''
        #vcombined=name+'\t'+fact
        vcombined=name.replace(" ","")+' '+fact
        nameFact.append(vcombined) 
        
    #Creating Training and Test data sets only using Names and Facts 
    training_data, test_data, training_labels, test_labels = train_test_split(nameFact, Results, test_size=0.30, random_state=12)    
    
    #Using Count Vectorizer to covert the words into a matrix
    counts_train,counts_test,feature_names=CountVectorizerTrain(training_data,test_data,0.0000)  
        
    #Naive Bayes Algorithm
    NaiveBayesAlgo(counts_train,counts_test,training_labels,test_labels,'Names and Facts')
    
    #Creating Training and Test data sets using Name Identifier and Facts as is
    nameFact=[]
    for name,fact in list(zip(Names,Facts)):
        vcombined=''
        vcombined='p'+name.replace(" ","")+' '+fact+' p'+name.replace(" ","")+' p'+name.replace(" ","")
        nameFact.append(vcombined) 
    #Creating Training and Test data sets only using Name Identifier and Facts
    training_data, test_data, training_labels, test_labels = train_test_split(nameFact, Results, test_size=0.30, random_state=12)    
    
    #Using Count Vectorizer to covert the words into a matrix
    counts_train,counts_test,feature_names=CountVectorizerTrain(training_data,test_data,0.0000)  
        
    #Naive Bayes Algorithm
    NaiveBayesAlgo(counts_train,counts_test,training_labels,test_labels,'Name Identifier and Facts')
    
    #KNN Classifier
    KNNAlgo(counts_train,counts_test,training_labels,test_labels,'Name Identifier and Facts')
    
    #Logistic Regression
    LogisticRegAlgo(counts_train,counts_test,training_labels,test_labels,'Name Identifier and Facts')
    
    #Decision Trees
    decisionTreeAlgo(counts_train,counts_test,training_labels,test_labels,'Name Identifier and Facts')
    
    #Random Forest
    feature_importance=RandomForestAlgo(counts_train,counts_test,training_labels,test_labels,'Name Identifier and Facts')
    plotimportantFeatures(feature_names,feature_importance,'[true,mostly true,pants on fire,false]')
   
    ###########Scenario 4: values=['true','false','mostly true','pants on fire','mostly false','false']
    print("\n\n")    
    print("#############Scenario 4: Statements=['true','mostly true','mostly false','pants on fire','false']#############")
    values=['true','mostly true','pants on fire','mostly false','false']
    Names, Facts, Results, Months=filterData('Dataset.txt',values)
    countTrue,countFalse = countResults(Results,values)    
    #Creating Training and Test data sets only using Name and Facts as is
    nameFact=[]
    for name,fact in list(zip(Names,Facts)):
        vcombined=''
        #vcombined=name+'\t'+fact
        vcombined=name.replace(" ","")+' '+fact
        nameFact.append(vcombined) 

    #Creating Training and Test data sets only using Names and Facts 
    training_data, test_data, training_labels, test_labels = train_test_split(nameFact, Results, test_size=0.30, random_state=12)    
    
    #Using Count Vectorizer to covert the words into a matrix
    counts_train,counts_test,feature_names=CountVectorizerTrain(training_data,test_data,0.0000)  
        
    #Naive Bayes Algorithm
    NaiveBayesAlgo(counts_train,counts_test,training_labels,test_labels,'Names and Facts')
    
    #Creating Training and Test data sets using Name Identifier and Facts as is
    nameFact=[]
    for name,fact in list(zip(Names,Facts)):
        vcombined=''
        vcombined='p'+name.replace(" ","")+' '+fact+' p'+name.replace(" ","")+' p'+name.replace(" ","")
        nameFact.append(vcombined) 
    
    #Creating Training and Test data sets only using Name Identifier and Facts
    training_data, test_data, training_labels, test_labels = train_test_split(nameFact, Results, test_size=0.30, random_state=12)    
    
    #Using Count Vectorizer to covert the words into a matrix
    counts_train,counts_test,feature_names=CountVectorizerTrain(training_data,test_data,0.0000)  
        
    #Naive Bayes Algorithm
    NaiveBayesAlgo(counts_train,counts_test,training_labels,test_labels,'Name Identifier and Facts')
    
    #Logistic Regression
    LogisticRegAlgo(counts_train,counts_test,training_labels,test_labels,'Name Identifier and Facts')
    
    #KNN Classifier
    KNNAlgo(counts_train,counts_test,training_labels,test_labels,'Name Identifier and Facts')
    
    #Decision Trees
    decisionTreeAlgo(counts_train,counts_test,training_labels,test_labels,'Name Identifier and Facts')
    
    #Random Forest
    feature_importance=RandomForestAlgo(counts_train,counts_test,training_labels,test_labels,'Name Identifier and Facts')
    plotimportantFeatures(feature_names,feature_importance,'[true,mostly true,pants on fire,mostly false,false]')
    
    
 
