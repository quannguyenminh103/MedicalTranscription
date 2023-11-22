from utils import *
from constants import *
import itertools
import numpy as np
import pandas as pd
np.random.seed(0)
## working
data = pd.read_csv('./data/discharge.csv')
# field_names = ["Discharge Disposition:", "History of Present Illness:",
#                "Physical Exam:", "Chief Complaint:", "Allergies:"]#"Discharge Diagnosis:", "Discharge Condition:", 
# idx = [i for i in range(len(data)) if all(ext.lower() in data.text[i].lower() for ext in field_names)]
# data = data.iloc[idx]
# data.reset_index(drop=True,inplace=True)
label_list = ["Discharge Disposition:\nHome", "Discharge Disposition:\nHome With Service", 
              "Discharge Disposition:\nExtended Care", "Discharge Disposition:\nExpired"]
## Expired = Deceased => lowest #data
print(-1)
expired_dt = data.text[data.text.str.contains("Discharge Disposition:\nExpired\n")]
home_dt = data.text[data.text.str.contains("Discharge Disposition:\nHome\n")]
hws_dt = data.text[data.text.str.contains("Discharge Disposition:\nHome With Service\n")]
ext_dt = data.text[data.text.str.contains("Discharge Disposition:\nExtended Care\n")]
home_dt_sample = home_dt.sample(n = len(expired_dt))
hws_dt_sample = hws_dt.sample(n = len(expired_dt))
ext_dt_sample = ext_dt.sample(n = len(expired_dt))
print(0)
data_sample = pd.concat([home_dt_sample, hws_dt_sample, ext_dt_sample, expired_dt])
train = data_sample.sample(frac=1).to_numpy() ## shuffle data
data = data.sample(frac=0.20) # lower fraction to speed up process
data.to_csv('filtered_discharge.csv', index = False)
data = pd.read_csv('./data/filtered_discharge.csv')
#dt, test = train_test_split(data_sample, test_size=0.2)
#train, validation = train_test_split(dt, test_size=0.2)
processed_text = list(map(parser_text,train))
print(1)
#disposition = stages of patients
raw_disposition = list(map(get_disposition, processed_text)) # get list of stages
processed_disposition = list(map(fill_empty_disposition, raw_disposition)) # some stage-labels are missing, imputing them
dis_idx = [i for i in range(len(processed_disposition)) if processed_disposition[i] in disposition_results] # just keeps 4 labels
disposition = np.array(processed_disposition)[dis_idx] ### y_value
print(2)
## get corresponding train data
data_processed = [processed_text[i] for i in dis_idx]
print(3)
## get Allergies
allergies = list(map(get_allergy, data_processed))
print(allergies[:2])
## get Chief Complaint
chief_complaint = list(map(get_info, data_processed, "Chief Complaint"))
print(chief_complaint[:2])
## get Major Surgical or Invasive Procedure
surgery_procedure = list(map(get_info, data_processed, "Major Surgical or Invasive Procedure"))
print(surgery_procedure[:2])
## get History: "History of Present Illness", 'Past Medical History', 'Social History', 'Family History'
#train_history = list(map(get_history, train_processed))
history = list(map(get_info, data_processed, "History of Present Illness:", "Physical Exam:"))
print(history[:2])
## get Physical Exam
## get Brief Hospital Course
brief_hospital_course = list(map(get_info, data_processed, "Brief Hospital Course"))
print(brief_hospital_course[:2])
## get ...
## concat all data:
headings_used = ["Allergies", "Chief Complaint", "Major Surgical or Invasive Procedure", 
                 "History of Present Illness:", "Brief Hospital Course"]
zipped_data = zip(allergies, chief_complaint, surgery_procedure, history, brief_hospital_course)
concat_data = ["\n".join(list(itertools.chain.from_iterable(zip(headings_used, [a,b,c,d,e])))) for (a,b,c,d,e) in list(zipped_data)]
print(len(concat_data))
X = [" ".join(t) for t in tqdm(concat_data) if token_length(t) <= 512]
y = [disposition[i] for i in tqdm(range(len(history))) if token_length(concat_data[i]) <= 512]
print(len(X))
final_data = pd.DataFrame({"text":X, "label":y})
final_data.to_csv('stages_trial.csv', index = False)