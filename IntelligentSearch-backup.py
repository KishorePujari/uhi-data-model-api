
from operator import concat
import numpy as np
import pandas as pd

import json

import re
import os
import glob
from ast import literal_eval
import sys
from flask import Flask, request, jsonify

from difflib import SequenceMatcher

import time

import warnings
warnings.simplefilter("ignore")


from pathlib import Path

from config import *

from collections import Counter

from spellchecker import SpellChecker

import gradio as gr
app = Flask(__name__)

#import textdistance
import operator
import editdistance
import spacy

DISEASE_API=gr.Interface.load("ugaray96/biobert_ncbi_disease_ner", src='models')
PHARMA_API=gr.Interface.load("kormilitzin/en_core_med7_trf", src='models')
with open(os.path.join(MODEL_PATH,'symptons-all.txt'), 'r') as f:
    SYM_VOCAB = f.readlines()
SYM_VOCAB=list(set([word.strip().lower() for word in SYM_VOCAB]))
print(len(SYM_VOCAB))

with open(os.path.join(MODEL_PATH,'diagnosis_all.txt'), 'r',encoding="utf-8") as f:
    DIAG_VOCAB = f.readlines()
DIAG_VOCAB =list(set([word.strip().lower() for word in DIAG_VOCAB]))
print(len(DIAG_VOCAB))



NLP_NER = spacy.load(NER_MODEL_PATH)

'''
def getMostSimilarWord(input_word,vocab):
    
    sim = {v:1-(textdistance.Jaccard(qval=2).distance(v,input_word)) for v in vocab}
    df = pd.DataFrame.from_dict(sim, orient='index').reset_index()
    df = df.rename(columns={'index':'Word', 0:'Similarity'})
    output = df.sort_values(['Similarity'], ascending=False).head(1)
    return output['Word'].tolist()[0]



def autocorrect_similarity(input_text):
    with open(ENGLISH_VOCAB_FILE, 'r') as f:
        vocab = f.readlines()
    vocab=list(set([word.strip().lower() for word in vocab]))
    output_word=[]
    input_words=input_text.split()
    for input_word in input_words:
        if input_word in vocab:
            output_word.append(input_word)
        else:
            output_word.append(getMostSimilarWord(input_word,vocab))
    return " ".join(output_word)
'''
def authenticate(header):
    auth = header.get("X-Api-Key")
    if auth == 'intelli-search-csh':
        return True
    else:
        return False

def autocorrect(input_text):
    spell = SpellChecker()
    input_words=input_text.split()
    output_words=[]
    for word in input_words:
        output_words.append(spell.correction(word))
    return " ".join(output_words)

def predictSymptomOrDiagnosis(input_text,vocabs):
    print(len(vocabs))
    edistance={}
    #vocabs=[vocab.replace(" ","_") for vocab in vocabs]
    input_text=input_text.lower()
    #input_text=input_text.replace(" ","_")
    
    #if len(filter_vocab)==0:
    #filter_vocab=[word for word in vocabs if input_text in word]
    #print(len(filter_vocab))
    filter_vocab_list=[]
    for word in input_text.split():
        filter_vocab_list.append([w for w in DIAG_VOCAB if word.strip() in w.strip()])
        ### Get words whose count is equal to lebgh ong th splut string
    vocab_filtered = [x for xs in filter_vocab_list for x in xs]
    counters=dict(Counter(vocab_filtered))
    num_count=len(input_text.split())

    filtered_count={x: count for x, count in counters.items() if count >= num_count}
    filter_vocab=list(filtered_count.keys())

    '''
    final_vocab=[]
    for word in filter_vocab:
        word_list=word.split()
        for words in word_list:
            if words.startswith(input_text):
                final_vocab.append(word)
    '''
    for word in filter_vocab:
        edistance[word]=editdistance.eval(input_text,word)
    d=dict( sorted(edistance.items(), key=operator.itemgetter(1)))
    return list(d.keys())


def getDisease(input_text):
    
    spans = DISEASE_API(input_text)
    replaced_spans =[(key, None) if value=='No Disease' else (key, value) for (key, value) in spans]
    
    return replaced_spans

def getIndex(key_val_list,search_value="Disease"):
    disease_index=[]
    for idx,(key,value) in enumerate(key_val_list):
        #print(idx,key,value)
        if search_value==value:
            disease_index.append(idx)
    return disease_index

def concatNER(disease_index,diseases_span,accepted_labels):
    diseases_list=[]
    for i in range(0,len(disease_index)):
        start_idx=disease_index[i]
        if i!=len(disease_index)-1:
            end_idx=disease_index[i+1]
        else:
            end_idx=len(diseases_span)


        dat=diseases_span[start_idx:end_idx]
        keys=[key for (key,value) in dat if value in accepted_labels or key.strip()==""]

        diseases_list.append("".join(keys).strip())
    
    return diseases_list

@app.route("/extractDisease",methods=['GET','POST'])
def extract_disease():
    headers = request.headers
    authenticated = authenticate(headers)
    if authenticated == True:
        input_text=request.json['text']
        try:
            diseases_span=getDisease(input_text)
            diseases_index=getIndex(diseases_span,"Disease")
            print(diseases_index)
            diseases=concatNER(diseases_index,diseases_span,accepted_labels=['Disease Continuation','Disease'])
        

                


            print(diseases)
        except Exception as e:
            print(e)
            return json.dumps({"Status":"Error"})
        return json.dumps({"Status":"Success","extracted_diseases":diseases})
    else:
        return json.dumps({"Status":"Error: Unauthorized"}), 401



def get_pharma(input_text):
    
    print("In Pharma")
    spans = PHARMA_API(input_text)

    drug_index=getIndex(spans,"DRUG")
    print(drug_index)

    #med_strengths=[]
    #drugs=[]
    #frequencies=[]
    #durations=[]
    medications=[]
    if len(drug_index)==0:
        return medications
    for i in range(0,len(drug_index)):
        start_idx=drug_index[i]

        if i!=len(drug_index)-1:
            end_idx=drug_index[i+1]
        else:
            end_idx=len(spans)
        print(start_idx,end_idx)
        dat=spans[start_idx:end_idx]
        print(dat)
        drug=[key for (key,value) in dat if value=="DRUG"]
        med_strength=[key for (key,value) in dat if value=="STRENGTH"]
        frequency=[key for (key,value) in dat if value=="FREQUENCY"]
        duration=[key for (key,value) in dat if value=="DURATION"]
        
        drugs=" ".join(drug)
        if len(med_strength)>0:
            med_strengths=" ".join(med_strength)
        else:
            med_strengths=""
        
        if len(frequency)>0:
            frequencies=" ".join(frequency)
        else:
            frequencies=""
        if len(duration)>0:
            durations=" ".join(duration)
        else:
            durations=""
        medications.append({'drugs':drugs,"dosage":med_strengths,"frequency":frequencies,"duration":durations})
    return medications

    

@app.route("/extractmedication",methods=['GET','POST'])
def extract_medication():
    headers = request.headers
    authenticated = authenticate(headers)
    if authenticated == True:
        print("In Extract Medication")
        input_text=request.json['text']

        print(input_text)
        try:
            result=get_pharma(input_text)
        except Exception as e:
            print(e)
            return json.dumps({"Status":"Error"})

        return json.dumps({"Status":"Success","medications":result})
    else:
        return json.dumps({"Status":"Error: Unauthorized"}), 401



@app.route("/autocorrect", methods=['GET', 'POST'])
def autocorrect_text():
    headers = request.headers
    authenticated = authenticate(headers)
    if authenticated == True:
        input_text=request.json['search_text']
        try:
            output_word=autocorrect(input_text)
        except Exception as e:

            return json.dumps({"Status":"Error"})

        return json.dumps({"Status":"Success","corrected_search_text":output_word})
    else:
        return json.dumps({"Status":"Error: Unauthorized"}), 401

@app.route("/autopredict", methods=['GET', 'POST'])
def predict_symptom():
    headers = request.headers
    authenticated = authenticate(headers)
    if authenticated == True:
        input_text=request.json['text']
        text_type=request.json['text_type']
        
        try:
            if text_type=="symptoms":
                
                print("In Symptoms")
                relevant_symptoms=predictSymptomOrDiagnosis(input_text,SYM_VOCAB)
            if text_type=="diagnosis":
                print("In Symptoms")
                relevant_symptoms=predictSymptomOrDiagnosis(input_text,DIAG_VOCAB)
            if text_type not in ['symptoms','diagnosis']:
                return json.dumps({'Status:Error. Pass type to be either symptoms or diagnosis'})
            

        except Exception as e:

            return json.dumps({"Status":"Error"})

        return json.dumps({"Status":"Success","predicted_words":relevant_symptoms})
    else: 
        return json.dumps({"Status":"Error: Unauthorized"}), 401


def extractSearchEntities(input_text):
    doc=NLP_NER(input_text)
    entity_dict={}
    for word in doc.ents:
        if word.label_=="LOCATION":
            
            entity_dict["location"]=word.text
        if word.label_=="NAME":
            entity_dict['physician_name']=word.text
        if word.label_=="SPECIALITY":
            entity_dict['specialty']=word.text
        if word.label_=="HOSPITAL":
            entity_dict['hospital']=word.text
    return entity_dict

@app.route("/extractEntities", methods=['GET', 'POST'])
def extract_entities():
    headers = request.headers
    authenticated = authenticate(headers)
    if authenticated == True:
        input_text=request.json['query']
        try:
            entity_dict=extractSearchEntities(input_text)
        except Exception as e:

            return json.dumps({"Status":"Error"})

        return json.dumps({"Status":"Success","entities":entity_dict})
    else:
        return json.dumps({"Status":"Error: Unauthorized"}), 401

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

