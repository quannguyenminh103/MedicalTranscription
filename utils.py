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

def impute_missing_headings(term, text):
    distance_func = np.vectorize(levenschtein)
    idx = distance_func(str(np.char.lower(text)), str((term + ":").lower()))
    print(text)
    text[np.argmin(idx)] = term+":"
    return text

def parser_text(text):
    split_by_n = text.split("\n")
    stripped_text = np.array([t.strip() for t in split_by_n])
    text = np.array([word for word in stripped_text if word != "___" and len(word) != 0])
    distance_func = np.vectorize(levenschtein)
    ### need to create a function that still return the field name even though it's missing. Ex: Physical Exam ---> Physical ____
    ### or Discharge Diagnosis ---> ___ Diagnosis or Discharge ___. 
    ### this will replace the idx below
    missing_bool = [False if np.min(distance_func(np.char.lower(text), (term + ":").lower())) == 0 else True for term in standardized_terms]
    if np.sum(missing_bool) > 0:
        for term in np.array(standardized_terms)[missing_bool]:
            idx = distance_func(np.char.lower(text), (term + ":").lower())
            missing_idx = np.where(np.char.find(np.char.lower(text[idx < 11]),"___") == 0)[0]
            if len(missing_idx) != 0:
                missing_headings = text[idx < 11][missing_idx[0]]
                text[text == missing_headings] = term+":"
            elif np.sum(idx <= 7) > 0 and term.lower() != "discharge disposition" and term.lower() != "discharge diagnosis":
                missing_headings = text[idx <= 7]
                correct_idx = [True if len(k.split()) == len(term.split()) else False for k in missing_headings]
                if np.sum(correct_idx) > 0:
                    text[text == missing_headings[correct_idx][0]] = term + ":"
            #text[np.argmin(idx)] = term+":"
    return np.array(text)

def get_disposition(text):
    disposition_idx = np.where(np.char.find(np.char.lower(text),"Discharge Disposition:".lower()) == 0)[0][0]
    terms = text[disposition_idx+1]
    return terms

def get_info(text, feature1):
    """
    Get the paragraph for the feature 1. It is assumed that this information chunk is between feature 1 and feature 2
    """
    headings = np.array([t for t in text if t[-1] == ":"])
    #try:
    idx1 = np.where(np.char.find(np.char.lower(text),feature1.lower()) == 0)[0]
    try:
        if len(idx1) != 0:
            if feature1 == "Allergies":
                idx2 = np.where(np.char.find(np.char.lower(text),"Attending:".lower()) == 0)[0]
                if len(idx2) > 0:
                    feature2 = "Attending:"
                else:
                    feature2 = headings[np.where(headings == (feature1+":"))[0]+1][0]
            elif feature1 == "History of Present Illness:":
                feature2 = headings[np.where(np.char.find(np.char.lower(headings),"history") == 0)[-1]+1][0]
            else:
                feature2 = headings[np.where(headings == (feature1+":"))[0]+1][0]
            idx2 = np.where(np.char.find(np.char.lower(text),feature2.lower()) == 0)[0][0]
            if (idx1[0] + 1) == idx2:
                return np.array(feature1+" is None.")
            #terms = text[idx1[0]:idx2]
            #terms = feature1 + " is " + text[idx1[0]+1:idx2]
            terms = np.concatenate((np.array([feature1 + ' is']), text[idx1[0]+1:idx2]), axis = 0)
            terms = " ".join(terms)
        else:
            terms = np.array(feature1+" is None.")
    except:
        terms = feature1+" is None."
    return terms
def token_length(text):
    #ftxt =  " ".join(text)
    return len(re.findall(r'\w+', text))
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