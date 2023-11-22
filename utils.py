import numpy as np
import torch
import re
import copy
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from pylev import levenschtein
from constants import *
def parser_text(text):
    split_by_n = text.split("\n")
    stripped_text = np.array([t.strip() for t in split_by_n])
    text = np.array([word for word in stripped_text if word != "___" and len(word) != 0])
    distance_func = np.vectorize(levenschtein)
    ### need to create a function that still return the field name even though it's missing. Ex: Physical Exam ---> Physical ____
    ### or Discharge Diagnosis ---> ___ Diagnosis or Discharge ___. 
    ### this will replace the idx below 
    for term in standardized_terms:
        idx = distance_func(np.char.lower(text), (term + ":").lower())
        text[np.argmin(idx)] = term+":"
    return text
def get_disposition(text):
    disposition_idx = np.where(np.char.find(np.char.lower(text),"Discharge Disposition:".lower()) == 0)[0][0]
    terms = text[disposition_idx+1]#:diagnosis_idx]
    return terms
def get_allergy(text):
    allergy_idx = np.where(np.char.find(np.char.lower(text),"Allergies:".lower()) == 0)[0][0]
    terms = text[allergy_idx+1]#:diagnosis_idx]
    return terms
def get_info(text, feature1, feature2 = ''):
    """
    Get the paragraph for the feature 1. It is assumed that this information chunk is between feature 1 and feature 2
    """
    headings = np.array([t for t in text if t[-1] == ":"])
    idx1 = np.where(np.char.find(np.char.lower(text),feature1.lower()) == 0)[0][0]
    if len(feature2) == 0:
        feature2 = headings[np.where(headings == feature1)+1]
    idx2 = np.where(np.char.find(np.char.lower(text),feature2.lower()) == 0)[0][0]
    terms = text[idx1+1:idx2]
    return terms
def get_diagnosis(text):
    diagnosis_idx = np.where(np.char.find(np.char.lower(text),"Discharge Diagnosis:".lower()) == 0)[0][0]
    status_idx = np.where(np.char.find(np.char.lower(text),"Discharge Condition:".lower()) == 0)[0][0]
    terms = text[diagnosis_idx+1:status_idx]
    return terms
def get_history(text):
    history_idx = np.where(np.char.find(np.char.lower(text),"History of Present Illness:".lower()) == 0)[0][0]
    try:
        physical_idx = np.where(pd.Series(np.char.lower(text)).str.contains("Physical Exam:".lower()))[0][0]
    except:
        if "Physical ___".lower() in np.char.lower(text):
            physical_idx = np.where(np.char.find(np.char.lower(text),"Physical ___:".lower()) == 0)[0][0]
        elif "___ Exam:".lower() in np.char.lower(text):
            physical_idx = np.where(np.char.find(np.char.lower(text),"___ Exam:".lower()) == 0)[0][0]
        else:
            print(text)
            raise ValueError
    terms = text[history_idx+1:physical_idx]
    return terms
def token_length(text):
    ftxt =  " ".join(text)
    return len(re.findall(r'\w+', ftxt))
def fill_empty_disposition(text):
    if "___" in text:
        if "Extended" or "Care" in text:
            return "Extended Care"
        elif "With" or "Service" in text:
            return "Home With Service"
    else:
        return text

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1
    }