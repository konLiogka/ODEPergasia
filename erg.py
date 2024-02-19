from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from pandas.plotting import parallel_coordinates as pc

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from apyori import apriori

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random


# ----------------------
# Subject A - Imputation
# ----------------------


# K-Nearest Value Imputation

# Read .csv file
table = pd.read_csv("marketing.csv")
# Replace '?' values with np.nan
table.replace("?", np.nan, inplace=True)

 

 
# Implement imputer type
imputer = KNNImputer(n_neighbors=10)
# Impute the csv file
table = pd.DataFrame(imputer.fit_transform(table), columns=table.columns).round()
# Export dataframe
table.to_csv("./imputed", sep="\t")


# Select features
marketing = table.loc[
    :, [ "Age", "Occupation", "Education", "EthnicClass","MaritalStatus","Income"]
]
print()

# Plots between some features
plt.figure(figsize=(20, 10))
plt.title("Plots between various features and Income")



plt.subplot(2, 2, 1)
rg = marketing["Age"].astype(int).max() + 1
sns.boxplot(
    marketing,
    x="Age",
    y="Income",
    hue=marketing["Sex"].map({1.0: "Male", 2.0: "Female"}),
    order=range(2, rg),
    palette="Set2",
)


plt.subplot(2, 2, 2)
rg = marketing["Occupation"].astype(int).max() + 1
sns.barplot(marketing, x="Occupation", y="Income", order=range(1, rg))


plt.subplot(2, 2, 3)
rg = marketing["EthnicClass"].astype(int).max() + 1
sns.violinplot(
    marketing,
    x="EthnicClass",
    y="Income",
    order=range(1, rg),
    hue=marketing["Sex"].map({1.0: "Male", 2.0: "Female"}),
    palette="Set3",
)


plt.subplot(2, 2, 4)
sns.kdeplot(marketing, x="Income", hue="Age", palette="Set2")


plt.show()





# ----------------------
# Subject B - Clustering
# ----------------------


def plotting(model, marketing, K):
    # Create new dataframe with the predicted clusters
    df = pd.DataFrame(marketing)
    df["cluster"] = model.labels_
 

    # Print 5 first entries to check the features
    print(df.head())
    # Create figure

    # Show parallel coordinates diagram
    figures = K // 4
    mod = K % 4
    clu = 0
    # Randomize oclours based on our K clusters
    colorList = [
        "#" + "".join(random.choice("0123456789ABCDEF") for _ in range(6))
        for _ in range(K)
    ]
    # Plot K//4 figures that each have 4 subplots
    for j in range(figures):
        plt.figure(figsize=(15, 7))
        for i in range(4):
            plt.subplot(4, 1, i + 1)
            pc(df[df["cluster"] == clu], "cluster", color=colorList[clu])
            clu += 1

    # If there is a remainder, make another figure.
    if mod > 0:
        plt.figure(figsize=(10, 5))
        for i in range(mod):
            plt.subplot(4, 1, i + 1)
            pc(df[df["cluster"] == clu], "cluster", color=colorList[clu])
            clu += 1

    plt.show()


def kmeanscl(marketing):
    # 1. Elbow method for determining cluster number for KMeans

    # Distance of values from center of cluster, must minimize
    inertias = []
    # Range of K
    K = range(1, 40,2)
    # Calculate inertias for each K
    for i in K:
        model = KMeans(n_clusters=i, n_init="auto")
        model.fit(marketing)
        inertias.append(model.inertia_)
    # Plot the graph
    plt.figure(figsize=(10, 7))

    plt.plot(K, inertias, marker="o", c="blue")
    plt.xlabel("K Values")
    plt.ylabel("Inertias")
    plt.title("Elbow method for determining K")
    plt.grid(True)

    plt.show()
    print("Input cluster number: ")
    N = int(input())

    model = KMeans(n_clusters=N, n_init="auto")
    model.fit(marketing)


    plotting(model, marketing, N)


def aggrcl(marketing):
    # 2. Aggromerative clustering with dendrogram

    # Function for calculating the dendrogram leafs and nodes
    # It takes the model and additional arguments for customizing the dendrogram
    def plt_dendrogram(model, **kwargs):
        # counts of sample for each node
        counts = np.zeros(model.children_.shape[0])
        # total number of data points
        n_samples = len(model.labels_)

        # Loop for finding leaf nodes.
        # Iterating over every child of the model
        for i, merge in enumerate(model.children_):
            # number of samples under current node
            current_count = 0
            # child represents a leaf node or another merged node. Merged is the current child node
            for child in merge:
                # If current child is less than the samples, its a leaf node.
                if child < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child - n_samples]
            # Total number of samples under each node
            counts[i] = current_count
        # Creates linkage matrix by horizontally stacking the chikdren, counts and disitances
        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)
        # Plotting the dendrogram
        dendrogram(linkage_matrix, **kwargs)

    # Model used
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(marketing)

    # Plotting the figure

    plt.title("Hierarchical Clustering Dendrogram")
    plt_dendrogram(model, truncate_mode="level", p=3)
    plt.xlabel("Number of points in node")

    plt.show()
    print("Input cluster number: ")
    N = int(input())

    model = AgglomerativeClustering(
        n_clusters=N, metric="euclidean", linkage="complete"
    )
    model = model.fit(marketing)

    plotting(model, marketing, N)


 
def dbscancl(marketing):
    # 3. DBSCAN Clustering
    N=35

    # Check otimal eps value based on kdistances with N neighbors
    model = NearestNeighbors(n_neighbors=N)
    model = model.fit(marketing)
    distances, _ = model.kneighbors(marketing)

    distances = np.sort(distances[:, -1])
    plt.plot(distances)
    plt.grid(True)
    plt.title("NearestNeighbors K distances for determining Epsilon")
    plt.show()
    while True:
        try:
            epsIn=float(input("Input  epsilon value: "))
            break 
        except ValueError:
            print("Invalid Input")




    plt.figure(figsize=(15, 6))
    model = DBSCAN(eps=epsIn, min_samples=N,metric='euclidean', algorithm='ball_tree')
    model.fit(marketing)
    labels = model.labels_
    labels = set(labels[labels != -1]) # Exclude noise
    N = len(labels)

    plotting(model, marketing, N)


# --------------------------------------------------

 
 
# Loop for clustering methods
while True:
    print(
    "Select Clustering Method: \n",
    "1. K-Means \n",
    "2. Hierarchical \n",
    "3. DBSCAN \n",
    "4. Quit Clustering\n",
    )
    while True:
        try:
            answer = int(input())
            if answer in [1, 2, 3, 4]:
                break
            else:
                print("Invalid input enter 1, 2, 3 or 4")
        except ValueError:
            print("Invalid Input")

    match answer:
        case 1:
            kmeanscl(marketing)
        case 2:
            aggrcl(marketing)
        case 3:
            dbscancl(marketing)
        case 4:
            break


# ----------------------
# Subject C - Classification
# ----------------------



# Initialise dataframe without labels
X = marketing.drop(columns=['Income'], axis=1)
if 'cluster' in marketing.columns:
    marketing = marketing.drop(columns=['cluster'], axis=1)
   
# Initialise labels
y = marketing['Income']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12
)

# Ovrsampling only trained data
oversampler = SMOTE(sampling_strategy={8:2500, 9:2500})
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)
 
 # Function for training models
def classification(classifier):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    class_report = classification_report(y_test, y_pred, zero_division=1)
    print("OverallnClassification Report:\n", class_report)
 

# Loop for classification
while True:
     
    print(
        "Select Classification Method: \n",
        "1. Naive Bayes \n",
        "2. kNN \n",
        "3. Decision Tree \n",
        "4. Random Forest \n",
        "5. Quit Classification\n",
    )
    while True:
        try:
            answer = int(input())
            if answer in [1, 2, 3, 4, 5]:
                break
            else:
                print("Invalid input enter 1, 2, 3, 4 or 5")
        except ValueError:
            print("Invalid Input")

    match answer:
        case 1:
            # 1. Naive Bayes
            classifier = GaussianNB()
            classification(classifier)

        case 2:
            # 2. k Nearest Neighbors
            while True:
                try:
                  neighbors=int(input('Input neighbors: '))
                  if neighbors in  range(200):
                    break
                  else:
                    print("Too many neighbors")
                except ValueError:
                    print("Invalid Input")

            classifier = KNeighborsClassifier(n_neighbors=neighbors)
            classification(classifier)

        case 3:
            # 3. Decision Tree
            classifier = tree.DecisionTreeClassifier()
            classification(classifier)

        case 4:
            # 4. Random Forest (K Decision Trees)
            while True:
                try:
                    depth=int(input('Input depth: '))
                    if depth in  range(200):
                      break
                    else:
                      print("Too much depth")
                except ValueError:
                    print("Invalid Input")

            classifier = RandomForestClassifier(max_depth=depth, random_state=10)
            classification(classifier)
        case 5:
            break



# ----------------------
# Subject D - Apriori
# ----------------------
 
#Na balw oles ta features
        
marketing=table
 

mapC = {1 :'Male',  2 : 'Female'}
marketing['Sex']=marketing['Sex'].map(mapC)

mapC = {1:'14-17',2:'18-24',3:'25-34',4:'35-44',5:'45-54',6:'55-64',7:'75+'}
marketing['Age']=marketing['Age'].map(mapC)

mapC = {1:'Grade 8 or less',2:'Grade 9 to 11',3:'High School Grad',4:'1-3 yrs of College',5:'College Grad',6:'Grad Study'}
marketing['Education']=marketing['Education'].map(mapC)

mapC = {1:'Professional/Managerial',2:'Sales Worker',3:'Factory Worker/Laborer',4:'Cleric/Service',5:'Homemaker',6:'Student, HS or College',7:'Military', 8:'Retired', 9:'Unemployed'}
marketing['Occupation']=marketing['Occupation'].map(mapC)

mapC = {1:'American Indian',2:'Asian',3:'Black',4:'East Indian',5:'Hispanic',6:'Pacific English',7:'White',8:'Other'}
marketing['EthnicClass']=marketing['EthnicClass'].map(mapC)

mapC = {1:'Less than 10k',2:'10k - 15k',3:'15k to 20k',4:'20k to 25k',5:'25k to 30k',6:'30k to 40k',7:'40k to 50k',8:'50k to 75k',9:'75k or more'}
marketing['Income']=marketing['Income'].map(mapC)

mapC = {1:'Married',2:'Living Together',3:'Divorced',4:'Widowed',5:'Single'}
marketing['MaritalStatus']=marketing['MaritalStatus'].map(mapC)

mapC = {1:'One member',2:'Two members',3:'Three members',4:'Four members',5:'Five members', 6:'Six members',7:'Seven members',8:'Eight members',9:'Nine memberss or more'}
marketing['HouseholdMembers']=marketing['HouseholdMembers'].map(mapC)

mapC = {1:'House',2:'Condo',3:'Apartment',4:'Mobile Home',5:'Other home type'}
marketing['TypeOfHome']=marketing['TypeOfHome'].map(mapC)

mapC = {1:'Less than 1 yr in Sf',2:'1-3 yrs in Sf',3:'4-6 yrs in Sf',4:'7-10 yrs in Sf',5:'More than 10 yrs in Sf'}
marketing['YearsInSf']=marketing['YearsInSf'].map(mapC)

mapC = {1:'English',2:'Spanish',3:'Other'}
marketing['Language']=marketing['Language'].map(mapC)

mapC = {1:'None under 18',2:'1 under 18',3:'2 under 18',4:'3 under 18',5:'4 under 18',6:'5 under 18',7:'6 under 18',8:'7 under 18',9:'9 or more'}
marketing['Under18']=marketing['Under18'].map(mapC)

mapC = {1:'Not Married (dualincome)',2:'Yes, dual income',3:'No dual income'}
marketing['DualIncome']=marketing['DualIncome'].map(mapC)

mapC = {1:'Own',2:'Rent',3:'Lives with family'}
marketing['HouseholdStatus']=marketing['HouseholdStatus'].map(mapC)


# COnverting dt to list for using apriori algorithm
newM = marketing.astype(str)
newM = newM.values.tolist()

# Returns pair of feature name and value as a list, like a json file
column_to_feature = {i: col for i, col in enumerate(marketing.columns)}

# Loop for apriori, for testing any lift value
while True:
   
    while True:
            try:
                lift=float(input('Input lift value(0 or =<1 to quit): '))
                if lift >1:
                    break
                else:
                    sys.exit()
            except ValueError:
                print("Invalid Input")


    # Use Apriori algorithm
    rules =  apriori(newM,min_lift=lift)
    rules_list = [rule for rule in rules]
    for rule in rules_list:
        antecedent = [column_to_feature.get(item, item) for item in rule.ordered_statistics[0].items_base]
        consequent = [column_to_feature.get(item, item) for item in rule.ordered_statistics[0].items_add]
        print(f"\nRule: {antecedent} => {consequent}")


  