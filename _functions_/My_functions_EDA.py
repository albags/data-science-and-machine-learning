#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[1]:


def get_info_dataset(data):
    """ This function will be used to extract info from the dataset
    input: dataframe containing all variables"""
    
    print('Basic information from your dataset\n','---------------------------------------------')
    data.info()


# In[11]:


def get_info_dataset2(data, bool):
    """ This function will be used to extract info from the dataset
    input: dataframe containing all variables
    num: Boolean index. if it's True that means that you want to proint two list that
    will contain the col names of the categorical and numerical variables, respectively"""
    
    print('Basic information from your dataset\n','---------------------------------------------')
    data.info()
    
    if bool == True:
       num_var = data.select_dtypes(include=['int64', 'float64']).columns
       print('Numerical variables are:\n', num_var)
       print('-------------------------------------------------')

       categ_var = data.select_dtypes(include=['category', object]).columns
       print('Categorical variables are:\n', categ_var)
       print('-------------------------------------------------') 


# In[13]:


def get_info_dataset3(data, bool_index):
    """ This function will be used to extract info from the dataset
    input: dataframe containing all variables
    num: Boolean index. if it's True that means that you want to proint two list that
    will contain the col names of the categorical and numerical variables, respectively"""
    
    
    print('Basic information from your dataset\n','---------------------------------------------')
    data.info()
    
    if bool_index == True:
       num_var = data.select_dtypes(include=['int64', 'float64']).columns
       print('Numerical variables are:\n', num_var)
       print('-------------------------------------------------')

       categ_var = data.select_dtypes(include=['category', object]).columns
       print('Categorical variables are:\n', categ_var)
       print('-------------------------------------------------')
    return num_var, categ_var


# In[15]:


def get_info_datasetPrint(data, bool_index, print_index):
    """ This function will be used to extract info from the dataset
    input: dataframe containing all variables
            bool_index: Boolean index. if it's True that means that you want to proint two list that
            will contain the col names of the categorical and numerical variables, respectively
            print_index : True / False . If True == will show the results on the python screen"""
    
    if bool_index == True:
        num_var = data.select_dtypes(include=['int', 'float']).columns
        categ_var = data.select_dtypes(include=['category', object]).columns
        
    if print_index == True:
        print('Numerical variables are:\n', num_var)
        print('-------------------------------------------------')

        print('Categorical variables are:\n', categ_var)
        print('-------------------------------------------------')
    return num_var, categ_var


# In[24]:


def percentage_nullValues(data):
    """
    Function that calculates the percentage of missing values in every column of your dataset
    input: data --> dataframe
    """
    null_perc = round(data.isnull().sum() / data.shape[0],3) * 100.00
    null_perc = pd.DataFrame(null_perc, columns=['Percentage_NaN'])
    null_perc= null_perc.sort_values(by = ['Percentage_NaN'], ascending = False)
    return null_perc


# In[26]:


def select_threshold(data, thr):
    """
    Function that  calculates the percentage of missing values in every column of your dataset
    input: data --> dataframe
    
    """
    null_perc = percentage_nullValues(data)
      
    col_keep = null_perc[null_perc['Percentage_NaN'] <thr]
    col_keep = list(col_keep.index)
    print('Columns to keep:',len(col_keep))
    print('Those columns have a percentage of NaN less than', str(thr), ':')
    print(col_keep)
    data_c= data[col_keep]
    
    return data_c


# In[33]:


def fill_na(data):
    """
    Function to fill NaN with mode (categorical variabls) and mean (numerical variables)
    input: data -> df
    """
    for column in data:
        if data[column].dtype != 'object':
            data[column] = data[column].fillna(data[column].mean())  
        else:
            data[column] = data[column].fillna(data[column].mode()[0]) 
    print('Number of missing values on your dataset are')
    print()
    print(data.isnull().sum())
    return data


# In[2]:


def fill_naCool(data):
    """
    Function to fill NaN with mode (categorical variabls) and mean (numerical variables)
    input: data -> df
    """
    num_var, categ_var = get_info_datasetPrint(data, True, False)
    
    data[num_var] = data[num_var].fillna(data[num_var].mean())
    data[categ_var] = data[categ_var].fillna(data[categ_var].mode()[0]) 

    print('Number of missing values on your dataset are')
    print()
    print(data.isnull().sum())
    return data


# In[3]:


def corrCoef(data):
    """
    Function aimed to calculate the corrCoef between each pair of variables
    
    input: data->dataframe        
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    num_var, categ_var = get_info_datasetPrint(data, True, False)
    data_num = data[num_var]
    data_corr = data_num.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(data_corr,
                xticklabels = data_corr.columns.values,
               yticklabels = data_corr.columns.values,
               annot = True, vmax=1, vmin=-1, center=0, cmap= sns.color_palette("RdBu_r", 7))


# In[4]:


def corrCoef_Threshold(data, threshold):
    """
    Function aimed to calculate the corrCoef between each pair of variables
    
    input: data->dataframe 
           threshold -> True: we want to keep the variables with a corrCoef higher than the income
                        False: we want to keep all values, no filtering
            
            """
    
    num_var = data.select_dtypes(include=['int64', 'float64']).columns
    print(num_var)
    data_num = data[num_var]
    data_corr = abs(data_num.corr())
    data_cols = data_corr.columns
    
    if threshold == True:
        data_corr= pd.DataFrame(data_corr.unstack().sort_values(ascending = False), columns = ['corrCoef'])
       # threshold that I want to select. I will keep the variables with a corrCoef higher than the threshols
        thr = float(input('Threshold? (in positive sign, please) '))
        data_corr = data_corr[(data_corr.corrCoef >thr)].unstack()

        data_corr = pd.DataFrame(data_corr)
        
        mask = np.zeros_like(data_corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        
        # Create the plot
        plt.figure(figsize=(10,8))
        sns.heatmap(data_corr,xticklabels = data_cols,
                    yticklabels = data_cols,mask = mask,
                            annot = True, vmax=1, vmin=-1, center=0, cmap= sns.color_palette("RdBu_r", 7),
                   annot_kws={"size":10})
        
    else:
        plt.figure(figsize=(10,8))
        sns.heatmap(data_corr,
                    xticklabels = data_corr.columns.values,
                   yticklabels = data_corr.columns.values,
                   annot = True, vmax=1, vmin=-1, center=0, cmap= sns.color_palette("RdBu_r", 7),
                    annot_kws={"size":10})
    
    return data_corr

