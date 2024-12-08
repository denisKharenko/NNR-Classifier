import time
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from typing import List
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from collections import Counter
SEARCH_ITERATIONS = 100
LEARN_TIME_LIMIT = 270


def classify_with_NNR(data_trn, data_vld, data_tst) -> List:
    func_start = time.time()
    arr_tupple=create_df_and_scale(data_trn, data_vld, data_tst)     #create tupple that contains all the labels and scaled data seperated
    distances_tupple_vld_train= calc_all_distances(arr_tupple[0], arr_tupple[2])
    optimal_radius= learn_optimal_radius(arr_tupple[1],arr_tupple[2],arr_tupple[3],distances_tupple_vld_train,func_start)
    distances_tupple_tst_train= calc_all_distances(arr_tupple[4], arr_tupple[2])
    print(f'starting classification with {data_trn}, {data_vld}, and {data_tst}')
    predictions = list(calc_array_of_predictions(arr_tupple[2], arr_tupple[3],optimal_radius, distances_tupple_tst_train[2]))
    return predictions

def create_df_and_scale(data_trn, data_vld, data_tst) -> tuple: #the function recieves three csv paths represeting train valid and test data creates a data frame from them
    scaled_data_trn=  pd.read_csv(data_trn)                     #scales the feature vectors and seperates them from labels and returns a tupple containing all the seperated data
    scaled_data_vld = pd.read_csv(data_vld)
    scaled_data_tst = pd.read_csv(data_tst)
    data_labels_trn=pd.Series(scaled_data_trn['class'])
    scaled_data_trn= scaled_data_trn.drop(columns=['class'])
    data_labels_vld = pd.Series(scaled_data_vld['class'])
    scaled_data_vld = scaled_data_vld.drop(columns=['class'])
    data_labels_tst = pd.Series(scaled_data_tst['class'])
    scaled_data_tst= scaled_data_tst.drop(columns=['class'])
    scaler=StandardScaler()
    scaled_data_trn=scaler.fit_transform(scaled_data_trn)
    scaled_data_vld=scaler.transform(scaled_data_vld)
    scaled_data_tst=scaler.transform(scaled_data_tst)
    return (scaled_data_vld,data_labels_vld.to_numpy(),scaled_data_trn,data_labels_trn.to_numpy(),scaled_data_tst,data_labels_tst.to_numpy())


def learn_optimal_radius(data_labels_valid,data_trn,data_labels_trn,calculated_distances_tupple,time_stamp):# the function receives the valid labels and train data the calculated distances and a time stamp
    start_radius=calculated_distances_tupple[0]                                                             #between each vector from the valid to each vector in the train and the time stamp
    max_radius = calculated_distances_tupple[1]                                                             #the function returns optimal radius for prediciton
    step=max_radius - start_radius #calculate difference between min and max radius and adjust steps accordingly
    step= step/SEARCH_ITERATIONS
    best_radius=0
    best_accuracy=0.0
    for i in range(SEARCH_ITERATIONS):
        curr_radius= start_radius + (i*step)
        curr_accuracy=get_accuracy_score_from_valid_df(data_labels_valid, data_trn, data_labels_trn, curr_radius, calculated_distances_tupple[2])
        if(curr_accuracy > best_accuracy):
            best_accuracy = curr_accuracy
            best_radius = curr_radius
        if(time.time() - time_stamp > LEARN_TIME_LIMIT): #end search if total time exceeded 270 seconds and leave 30 seconds to finish the rest of the calculation
            break

    return (best_radius)
        
    

def get_accuracy_score_from_valid_df(data_labels_vld, data_trn, data_labels_trn, radius, distances):#the function receives the valid labels and train data the calculated distances and a radius
    array_of_predictions = calc_array_of_predictions(data_trn, data_labels_trn, radius, distances)  #returns an accuracy score by using a prediction array
    return(accuracy_score(data_labels_vld,array_of_predictions))


def calc_array_of_predictions(data_trn, data_labels_trn, radius, distances):#the function receives the train data the calculated distances and a radius
    array_of_predictions=np.array([])                                       #the function creates an array of class predicitions and returns it
    for i in range(distances.shape[0]):
        array_of_predictions = np.append(array_of_predictions,get_class_prediciton(data_trn,data_labels_trn,radius,distances[i]))
    return(array_of_predictions)


def get_class_prediciton(data_trn,data_labels_trn,radius,distances_of_vector):#the function receives the train data the calculated distances of a vector and a radius
    indexes_arr=np.where(distances_of_vector<radius)[0]                        #the function returns a predicted class for the vector using the radius and the distances received
    labels_array=data_labels_trn[indexes_arr]
    if(np.size(labels_array,axis=None)!=0):#if no train vectors in radius predict closest train vectors class
        counter_array=Counter(labels_array)
        return(counter_array.most_common()[0][0])
    else:
        return(data_labels_trn[np.argmin(distances_of_vector)])



def calc_all_distances(data, data_trn):                 #the function receives the train data and the data that we intend to predict their classes
    distances=cdist(data, data_trn, metric='euclidean') #the function calculates the euclidean distance between each vector in the data with each vector in the training data
    return(np.min(distances),np.max(distances),distances) #returns distances and also returns the min and max distance




if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    predicted = classify_with_NNR(config['data_file_train'],
                                  config['data_file_validation'],
                                  config['data_file_test'])

    df = pd.read_csv(config['data_file_test'])
    labels = df['class'].values

    if not predicted:  # empty prediction, should not happen in your implementation
        predicted = list(range(len(labels)))

    assert(len(labels) == len(predicted))  # make sure you predict label for all test instances
    print(f'test set classification accuracy: {accuracy_score(labels, predicted)}')

    print(f'total time: {round(time.time()-start, 0)} sec')
