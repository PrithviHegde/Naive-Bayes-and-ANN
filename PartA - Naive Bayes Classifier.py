import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from statistics import NormalDist
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


features = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','target']
categorical = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
continuous = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']

#task 1
def pre_proc(df:pd.DataFrame) -> pd.DataFrame:
    
    columnlist = df.columns.to_list()
    finaldf = [df[i].to_list() for i in columnlist]
    for i in range(len(finaldf)):


         for j in range(len(finaldf[i])):
            if (i==1):
                a = True
            if columnlist[i] in categorical and finaldf[i][j]==' ?':
                a = False
                finaldf[i][j] = df[columnlist[i]].mode()[0]
                #nancount +=1
            elif columnlist[i] in continuous and finaldf[i][j]==' ?':
                
                finaldf[i][j] = df[columnlist[i]].mode()[0]
                      
    fdf = np.array(finaldf)
    fdf = fdf.T

    finaldf = pd.DataFrame(fdf, columns=columnlist)
    return finaldf


def prepare_dataset(df:pd.DataFrame):
    train = df.sample(frac=0.8)
    test = df.drop(train.index)

    train_y = train['target'].to_list()
    train_t = [ 1 if i==' >50K' else 0 for i in train_y]
    train_t2 = np.array(train_t).reshape(len(train_t),1)

    test_y = test['target'].to_list()
    test_t = [ 1 if i==' >50K' else 0 for i in test_y]
    test_t2 = np.array(test_t).reshape(len(test_t),1)

    train_x = train.drop(['target'], axis=1).to_numpy()
    test_x = test.drop(['target'], axis=1).to_numpy()

    return train_x, train_t2, test_x, test_t2


#task 2, part 1
def class_prior(train_t):
    rich_prior = np.sum(train_t)/len(train_t)  
    poor_prior = 1-rich_prior

    return rich_prior, poor_prior  

#task 2, part 2: Implement a function to calculate the conditional probability of each 
# feature given to each class in the training set.

#Here, we have given the alpha value- smoothing factor- as a parameter. When it is set to zero, we see the case
# without any smoothing (the simple naive Bayes solution). When we change alpha, the probability values are impacted.
# This is to avoid the case where a class, unsampled, has zero probability and we can't model it.

# This smoothing method is called Laplacian smoothing. 

def conditional_features(trainx,traint,alpha=0):
    
    trainset = np.concatenate((trainx,traint),axis=1)

    result= []

    for i in range(len(trainset[0])-1):
        if features[i] in categorical:
            featvalnames=[]
            featvalnames = np.unique(np.array(trainset.T[i])).tolist()
            featvalcount = len(featvalnames)
            featvalprobs = [(0,0)]*featvalcount
                
            for j in range(len(trainset)):
                ind = featvalnames.index(trainset[j][i])
                featvalprobs[ind] = (featvalprobs[ind][0]+int(trainset[j][-1]+alpha),featvalprobs[ind][1]+1+((len(trainset[0])-1)*alpha))        

            featvals = [featvalprobs[i][0]/featvalprobs[i][1] for i in range(featvalcount)]
            feature = dict(zip(featvalnames,featvals))
            
            result.append(feature)
        
        else:
            posvals = []
            negvals = []
            for j in range(len(trainset)):
                if int(trainset[j][-1])==1:
                    posvals.append(trainset[j][i])
                else:
                    
                    negvals.append(trainset[j][i])
            pvals,nvals = np.array(posvals), np.array(negvals)
            pvals = pvals.astype(np.float64)
            nvals = nvals.astype(np.float64)

            result.append([np.mean(pvals), np.std(pvals),np.mean(nvals),np.std(nvals)])
        
        
    return result



            
        
    return result



#predict the class of a given instance using the Naive Bayes algorithm, and find accuracy.
def predict(conditional_features, traint, test_x):
    test_x = np.array(test_x).tolist()
    result = []

    more_prior, less_prior = class_prior(traint)
    for i in range(len(test_x)):
        prob_more=1
        prob_less=1
        lprob_m = 0
        lprob_l = 0
        example = test_x[i]
        #print(example)
        for j in range(len(example)):
            #print(example[j])
            if features[j] in categorical:
                    if example[j] not in list(conditional_features[j].keys()):
                        prob_more = prob_more*more_prior
                        prob_less = prob_less*less_prior 
                    else:
                        prob_more = prob_more*(conditional_features[j][example[j]])
                        prob_less = prob_less*(1-conditional_features[j][example[j]])
            else:
                #cond_prob = (NormalDist(conditional_features[j][0],conditional_features[j][1])).pdf(float(example[j]))/((NormalDist(conditional_features[j][0],conditional_features[j][1])).pdf(float(example[j]))+(NormalDist(conditional_features[j][2],conditional_features[j][3])).pdf(float(example[j])))
                prob_more = prob_more*((NormalDist(conditional_features[j][0],conditional_features[j][1])).pdf(float(example[j])))
                prob_less = prob_less*((NormalDist(conditional_features[j][2],conditional_features[j][3])).pdf(float(example[j])))
                #prob_more = prob_more*cond_prob
                #prob_less = prob_less*(1-cond_prob)
        
        if (prob_more*more_prior>=prob_less*less_prior):
            result.append(1)
        else:
            result.append(0)

    return result



def metrics(result,test_t):
    #accuracy calculation
    accuracy,precision,recall,f1,tp,tn,fp,fn = 0,0,0,0,0,0,0,0
    for i in range(len(result)):
        if (result[i]==1 and test_t[i][0]==1):
            tp +=1
        elif (result[i]==0 and test_t[i][0]== 0):
            tn += 1
        elif (result[i]==1 and test_t[i][0]==0):
            fp +=1
        elif (result[i]==0 and test_t[i][0]==1):
            fn +=1
        
    #print(tp,fp,tn,fn)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = precision*recall / (precision+recall)    
    f1*=2
    return accuracy, precision, recall, f1


def other_methods(train_x,train_t,test_x,test_t, columns):
    train_x = pd.DataFrame(data = train_x, columns=columns[:-1])
    test_x = pd.DataFrame(data = test_x, columns=columns[:-1])
    # print(train_x.education.value_counts())
    # # print(train_x.marital-status.value_counts())
    # print(np.unique(train_x["marital-status"]))


    def mapping1(columnName):
        columnName = columnName.strip()
        workclass = ['Private',
            'Self-emp-not-inc',
            'Local-gov',
            'State-gov',
            'Self-emp-inc',
            'Federal-gov',
            'Without-pay',
            'Never-worked'] 
        return workclass.index(columnName)
    
    train_x["workClassModif"] = train_x["workclass"].apply(lambda x: mapping1(x))
    train_x.drop(["workclass"], axis=1, inplace=True)


    def mapping2(columnName):
        columnName = columnName.strip()
        workclass = ['10th','11th','12th','1st-4th','5th-6th','7th-8th','9th'
,'Assoc-acdm','Assoc-voc','Bachelors','Doctorate','HS-grad'
,'Masters','Preschool','Prof-school','Some-college'] 
        
        return workclass.index(columnName)
    
    train_x["Education"] = train_x["education"].apply(lambda x: mapping2(x))
    train_x.drop(["education"], axis=1, inplace=True)


    def mapping3(columnName):
        columnName = columnName.strip()
        workclass = ['Divorced','Married-AF-spouse','Married-civ-spouse'
,'Married-spouse-absent','Never-married','Separated','Widowed']
        return workclass.index(columnName)
    
    train_x["MaritalStatus"] = train_x["marital-status"].apply(lambda x: mapping3(x))
    train_x.drop(["marital-status"], axis=1, inplace=True)


    def mapping4(columnName):
        columnName = columnName.strip()
        countries = ['Cambodia','Canada','China','Columbia','Cuba','Dominican-Republic'
,'Ecuador','El-Salvador','England','France','Germany','Greece'
,'Guatemala','Haiti','Holand-Netherlands','Honduras','Hong'
,'Hungary','India','Iran','Ireland','Italy','Jamaica','Japan'
,'Laos','Mexico','Nicaragua','Outlying-US(Guam-USVI-etc)','Peru'
,'Philippines','Poland','Portugal','Puerto-Rico','Scotland','South'
,'Taiwan','Thailand','Trinadad&Tobago','United-States','Vietnam'
,'Yugoslavia'] 
        countries = [elem.strip() for elem in countries]
        return countries.index(columnName)
    
    train_x["NativeCountry"] = train_x["native-country"].apply(lambda x: mapping4(x))
    train_x.drop(["native-country"], axis=1, inplace=True)


    def mapping5(columnName):
        columnName = columnName.strip()
        countries = ['Adm-clerical','Armed-Forces','Craft-repair','Exec-managerial'
,'Farming-fishing','Handlers-cleaners','Machine-op-inspct'
,'Other-service','Priv-house-serv','Prof-specialty','Protective-serv'
,'Sales','Tech-support','Transport-moving'] 
        countries = [elem.strip() for elem in countries]
        return countries.index(columnName)
    
    train_x["Occupation"] = train_x["occupation"].apply(lambda x: mapping5(x))
    train_x.drop(["occupation"], axis=1, inplace=True)


    def mapping6(columnName):
        columnName = columnName.strip()
        countries = ['Husband','Not-in-family','Other-relative','Own-child','Unmarried' ,'Wife']
        countries = [elem.strip() for elem in countries]
        return countries.index(columnName)
    
    train_x["Relationship"] = train_x["relationship"].apply(lambda x: mapping6(x))
    train_x.drop(["relationship"], axis=1, inplace=True)



    def mapping7(columnName):
        columnName = columnName.strip()
        countries = ['Amer-Indian-Eskimo','Asian-Pac-Islander','Black','Other','White'] 
        countries = [elem.strip() for elem in countries]
        return countries.index(columnName)
    
    train_x["Race"] = train_x["race"].apply(lambda x: mapping7(x))
    train_x.drop(["race"], axis=1, inplace=True)
    

    
    def mapping8(columnName):
        columnName = columnName.strip()
        countries = ['Female','Male']
        countries = [elem.strip() for elem in countries]
        return countries.index(columnName)
    
    train_x["Sex"] = train_x["sex"].apply(lambda x: mapping8(x))
    train_x.drop(["sex"], axis=1, inplace=True)

    test_x["workClassModif"] = test_x["workclass"].apply(lambda x: mapping1(x))
    test_x.drop(["workclass"], axis=1, inplace=True)

    test_x["Education"] = test_x["education"].apply(lambda x: mapping2(x))
    test_x.drop(["education"], axis=1, inplace=True)

    test_x["MaritalStatus"] = test_x["marital-status"].apply(lambda x: mapping3(x))
    test_x.drop(["marital-status"], axis=1, inplace=True)

    test_x["NativeCountry"] = test_x["native-country"].apply(lambda x: mapping4(x))
    test_x.drop(["native-country"], axis=1, inplace=True)

    test_x["Occupation"] = test_x["occupation"].apply(lambda x: mapping5(x))
    test_x.drop(["occupation"], axis=1, inplace=True)

    test_x["Relationship"] = test_x["relationship"].apply(lambda x: mapping6(x))
    test_x.drop(["relationship"], axis=1, inplace=True)

    test_x["Race"] = test_x["race"].apply(lambda x: mapping7(x))
    test_x.drop(["race"], axis=1, inplace=True)

    test_x["Sex"] = test_x["sex"].apply(lambda x: mapping8(x))
    test_x.drop(["sex"], axis=1, inplace=True)

    clf = LogisticRegression(max_iter=10000).fit(train_x,train_t.ravel())
    accuracy_LR = clf.score(test_x,test_t)
    nbor = KNeighborsClassifier(n_neighbors=4).fit(train_x,train_t.ravel())
    accuracy_KNN = nbor.score(test_x,test_t)

    return [accuracy_LR, accuracy_KNN]



data = pd.read_csv("adult.data", names=features)

output = []

def runFxn(data):
    data = data.sample(frac = 1)
    data = pre_proc(data)
    # print(len(data))
    columns = data.columns
    trainx, traint, testx, testt = prepare_dataset(data)
    feature_weighing = conditional_features(trainx,traint)
    res = predict(feature_weighing, traint, testx)
    acc,pre,rec,f1 = metrics(res,testt)


    # print("\nMetrics for naive Bayes: \n\n")
    # print("The accuracy of the naive Bayes classifier is ",acc*100,"%.")
    # print("The precision of the naive Bayes classifier is ",pre*100,"%.")
    # print("The recall of the naive Bayes classifier is ",rec*100,"%.")

    # print("The f-1 score of the naive Bayes classifier is ",f1*100,"%.\n")


    feature_weighing_Laplace = conditional_features(trainx,traint,alpha=4.67)
    res_lap = predict(feature_weighing_Laplace, testt, testx)
    acc_lap,pre_lap,rec_lap,f1_lap = metrics(res_lap,testt)
    

    # print("\nAfter Laplace smoothing, we see the accuracy and precision of the classifier become",acc_lap*100,"% and",pre_lap*100,"%.\n")


    comp_results = other_methods(trainx,traint,testx,testt,columns)


    # print("With logistic regression, we get an accuracy of ", comp_results[0]*100,"%.")
    # print("With k-nearest neighbours, we get an accuracy of ", comp_results[1]*100,"%.")

    output.append([acc, pre, rec, f1, acc_lap, pre_lap, comp_results[0], comp_results[1]])
    return

for i in range(10):
    print("Random Sample", i+1)
    runFxn(data)

output = np.array(output)
output = output*100
averages = np.array(output.mean(axis=0),dtype='<U5').reshape((1,8))
variances = np.array(output.var(axis=0),dtype='<U5').reshape((1,8))
output = np.append(output,averages,axis=0)
output = np.append(output,variances,axis=0)
labels = np.array(["Random Split "+str(i)+'  ' for i in range(1,output.shape[0]-1)]+["Average "]+["Variance "]).reshape((output.shape[0],1))
output = np.concatenate((labels,output),axis=1)
output = pd.DataFrame(output, columns=["Split No./Metric","Accuracy","Precision","Recall","F-1 Score","Smoothed Acc.","Smoothed Pre.", "Logistic Reg. Acc.", "KNN Acc."])
print()
print(output.to_markdown())
print()
