#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[11]:


from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
#필요한 패키지들 import하기
import pandas as pd
# import modin.pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
import gzip
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


icu_path = '/data2/je_hong/kaist/kaist/physionet.org/files/mimiciv/2.0/icu/'


# In[4]:


new_path = '/data2/je_hong/kaist/kaist/physionet.org/files/mimiciv/2.0/mimic_rev/'


# In[5]:


hos_path = '/data2/je_hong/kaist/kaist/physionet.org/files/mimiciv/2.0/hosp/'


# In[28]:


import pickle
with open(new_path+'new_dict/adm_stay.pkl','rb') as f:
    dict_ = pickle.load(f)


# ## admissions + icustay 불러오기 (hadm_id기준 labeling 한 데이터)

# In[29]:


dict_


# ## labeling 기준 feature들 불러오기

# In[12]:


feature_info = pd.read_csv("/data2/je_hong/kaist/kaist/physionet.org/files/mimiciv/2.0/mimic_rev/feature_info/feature_itemid.csv",  encoding='CP949')


# In[57]:


feature_dict = dict()
for i in feature_info['feature'].unique():
    id_ = (feature_info.loc[feature_info['feature'] == i]['itemid'].tolist())
    feature_dict[i] = id_


# In[113]:


itemids = feature_info.itemid.unique()


# In[58]:


feature_dict


# ## feature들 id 불러올 수 있는 chartevents불러오기

# ## 24시간 내 사망환자 제외한 데이터

# In[13]:


with gzip.open(new_path+'drop_2_chartevents.gz') as f:
    drop_chartevents = pd.read_csv(f, chunksize=10**2)
    drop_chartevents = list(drop_chartevents)
    drop_chartevents = pd.concat(drop_chartevents)
print(drop_chartevents.shape)


# In[14]:


drop_chartevents.head()


# In[16]:


## data type 변경
drop_chartevents['charttime'] = pd.to_datetime(drop_chartevents['charttime'])
# drop_chartevents['outtime'] = pd.to_datetime(drop_chartevents['outtime'])


# In[17]:


drop_chartevents.info()


# ## hadm_id가 중복될 시, 첫번째 것만 사용하기로 했어서 첫번째 feature들만 활용할 것

# In[30]:


HADMS = list(dict_.keys())


# In[31]:


len(HADMS)


# In[32]:


HADMS[0]


# In[38]:


dict_


# In[46]:


dict_[24510466][0]['STAYID']


# In[ ]:


#Function of making profiles
def make_profile(profile_dic, subject_id, hadm_id, gender, adm_time, dob, is_dead, dod):
    profile_list = profile_dic.get(subject_id, list())
    profile_list.append({'HADM_ID': hadm_id, 'GENDER': gender, 'DOB': dob, 'ADM_TIME': adm_time, 'IS_DEAD': is_dead, 'DOD': dod})
    profile_dic[subject_id] = profile_list
    return profile_dic


# In[232]:


dict_


# In[ ]:


## charttime이 intime - outtime 내에 있는가 확인


# In[74]:


drop_chartevents


# In[111]:


item_ids


# In[98]:


#Function of making profiles
def make_profile(profile_dic, subject_id, hadm_id, stay_id, charttime, itemid, valuenum, is_dead, intime, outtime, dod):
    profile_list = profile_dic.get(subject_id, list())
    profile_list.append({'HADM_ID': hadm_id, 'GENDER': gender, 'DOB': dob, 'ADM_TIME': adm_time, 'IS_DEAD': is_dead, 'DOD': dod})
    profile_dic[subject_id] = profile_list
    return profile_dic


# In[144]:


dict_2 = dict_.copy()


# In[174]:


drop_chartevents.info()


# In[176]:


drop_chartevents['charttime'].isnull().sum()


# ## charttime이 intime <= charttime <= outtime내에 있어야 함

# ### charttime에서 뽑아낼 것 timepoint(charttime):stay_id:item_id:value_num)
# - 추가 논의 사항 : 초기준? 분기준? 으로 데이터 뽑아낼 것인지??

# In[177]:


from datetime import datetime

NOT_CONVERTED = 'NOT_CONVERTED'


def str2datetime(s):
    def _convert(_s, _dformat):
        try:
            converted_dt = datetime.strptime(_s, _dformat)
        except Exception:
            return NOT_CONVERTED

        return converted_dt

    if isinstance(s, datetime):
        return s

    dformats = [
        '%Y-%m-%d %p %I:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%y-%m-%d %H:%M',
        '%Y-%m-%d',
        '%Y%m%d',
        '%Y-%m-%d +%H:%M',
        '%Y%m%d %H%M',
        '%Y%m%d%H%M%S',
        ]

    s = s.split('.')[0]
    for dformat in dformats:
        dt = _convert(s, dformat)
        if dt != NOT_CONVERTED:
            return dt

    return NOT_CONVERTED


# In[203]:


itemids


# In[215]:


from datetime import timedelta


# ## chartevent table 내에 있는 4시간 동안의 valuenum , itemid 가져오기

# In[218]:


drop_chartevents['charttime'] = drop_chartevents['charttime'].apply(lambda x: str2datetime(x))


# In[220]:


drop_chartevents['charttime']


# In[290]:


chart_dict= dict()
for hadm_id in tqdm(dict_2.keys()):
    ## adm_stay에서 중복 hadm 중 첫번째 stay_id만을 사용했으므로 똑같이 stay_id를 사용하면 된다. 
#     stay_id = (drop_chartevents.loc[drop_chartevents['hadm_id'] == hadm_id]['stay_id'].values[0]) #STAY_ID 중 첫번째 값만 사용했으니까
    stay_id = (dict_2[hadm_id][0]['STAYID'])
    intime = (dict_2[hadm_id][0]['INTIME'])
    outtime = (dict_2[hadm_id][0]['OUTTIME'])
    dod =  (dict_2[hadm_id][0]['DOD'])
    is_dead =  (dict_2[hadm_id][0]['IS_DEAD'])
    temp_df = (drop_chartevents.loc[drop_chartevents['stay_id'] == stay_id])
    sorted_temp_df = temp_df.sort_values(by=['charttime'], axis=0)
    if is_dead == 1:
        temp_df2 = (sorted_temp_df[(sorted_temp_df['charttime'] >= intime) & (sorted_temp_df['charttime'] <= dod)])
    else:
        temp_df2 = (sorted_temp_df[(sorted_temp_df['charttime'] >= intime) & (sorted_temp_df['charttime'] <= outtime)]) 
    # 4시간 동안의 데이터만 가져오기
    try:
        first_ct = str2datetime(temp_df2['charttime'].tolist()[0])
        charttimes = temp_df2[(temp_df2['charttime'] <= first_ct + timedelta(hours=4))]['charttime'].unique()
    #     print(charttimes)
        idx = 0
        temp_list = []
        for t in (charttimes):
            itemid = temp_df2.loc[(temp_df2['charttime'] == t) & (temp_df2['itemid'].isin(itemids))]['itemid'].tolist()
            valuenum = temp_df2.loc[(temp_df2['charttime'] == t) & (temp_df2['itemid'].isin(itemids))]['valuenum'].tolist()
            if len(itemid) == 0:
                pass
            else:
                temp_list.append("{}:{}:{}:{}".format(t, idx, itemid, valuenum))
                idx += 1
                if idx > 100:
                    pass
        chart_dict[stay_id] = temp_list
    except:
        pass
    
#     charttime_dict[stay_id] = temp_list 
        #     temp_df2[(temp_df2['itemid'].isin(itemids))]['charttime'].apply(lambda x:str2datetime(x))
#     print(temp_df2[(temp_df2['itemid'].isin(itemids))]['charttime'].values[0])
    
#     for t in (temp_df2.loc[(temp_df2['itemid'].isin(itemids)) & (temp_df2['stay_id'] == stay_id)]['charttime'].values):
#         itemid = temp_df2.loc[(temp_df2['itemid'].isin(itemids)) & (temp_df2['stay_id'] == stay_id) & (temp_df2['charttime'] == t)]['itemid'].values
#         valuenum = temp_df2.loc[(temp_df2['itemid'].isin(itemids)) & (temp_df2['stay_id'] == stay_id) & (temp_df2['charttime'] == t)]['valuenum'].values
#         print(itemid, valuenum, t)
#         temp_df2.loc[(temp_df2['itemid'].isin(itemids)) & (temp_df2['stay_id'] == stay_id)]['itemid']
#         temp_list = (dict_2.get(hadm_id))
#         temp_list[0]['ITEM_ID'] = itemid
#         temp_list[0]['VALUENUM'] = valuenum


# In[235]:


sorted_itemids = sorted(itemids)


# In[317]:


len(sorted_itemids)


# ### 각각 time step마다 itemid 갯수 제일 많은 것 찾기 -> 제일 많은 거에 맞춰서 순서 맞추게

# In[324]:


temp = []
for i in chart_dict.keys():
    for j in chart_dict[i]:
#         print(j.split(":"))
#             print(j.split(":")[-2])
            


# In[325]:


import pickle
with open(new_path+'new_dict/chartevents.pkl','wb') as f:
    pickle.dump(chart_dict,f)


# In[ ]:




