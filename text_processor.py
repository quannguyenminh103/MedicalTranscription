from utils import *
from constants import *
import itertools
import numpy as np
import pandas as pd
np.random.seed(0)
## working
data = pd.read_csv('./data/subset_discharge.csv')
# data = pd.read_csv('./data/small200_discharge.csv')
# dt, test = train_test_split(data, test_size=0.4)
# train, val = train_test_split(data, test_size=0.9)
processed_text = list(map(parser_text, tqdm(data.text)))
#disposition = stages of patients
raw_disposition = list(map(get_disposition, processed_text)) # get list of stages
processed_disposition = list(map(fill_empty_disposition, raw_disposition)) # some stage-labels are missing, imputing them
dis_idx = [i for i in range(len(processed_disposition)) if processed_disposition[i] in disposition_results] # just keeps 4 labels
disposition = np.array(processed_disposition)[dis_idx] ### y_value
## get corresponding train data
data_processed = [np.array(processed_text[i]) for i in dis_idx]
data_processed = np.asarray(data_processed, dtype = 'object')
vfunc = np.vectorize(get_info, otypes=[object])
## get Allergies
allergies = vfunc(np.array(data_processed), "Allergies")
## get Chief Complaint
chief_complaint = vfunc(np.array(data_processed), "Chief Complaint")
## get Major Surgical or Invasive Procedures
surgery_procedure = vfunc(np.array(data_processed), "Major Surgical or Invasive Procedure")
## get History: "History of Present Illness", 'Past Medical History', 'Social History', 'Family History'
history = vfunc(np.array(data_processed), "History of Present Illness:")
## get Physical Exam
physical_exams = vfunc(np.array(data_processed), "Physical Exam")
## get Brief Hospital Course
brief_hospital_course = vfunc(np.array(data_processed), "Brief Hospital Course")
## get ...
## concat all data:
zipped_data = zip(allergies, chief_complaint, surgery_procedure, history, physical_exams, brief_hospital_course)
concat_data = [" [SEP] ".join(np.hstack(x)) for x in list(zipped_data)]
# X = [t for t in tqdm(concat_data) if token_length(t) <= 512]
# y = [disposition[i] for i in tqdm(range(len(concat_data))) if token_length(concat_data[i]) <= 512]
final_data = pd.DataFrame({"text":concat_data, "label":disposition})
final_data.to_csv('stages_subset_wPE_is.csv', index = False)