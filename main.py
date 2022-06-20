from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

# Učitavanja podataka sa remote repositorija (GitHub)
dataset = pd.read_csv(
    r'https://raw.githubusercontent.com/DMejra/Dataset-for-e-commerce/main/online_shoppers_intention.csv')


# Printanje podataka dataseta
print(dataset)

# Predobrada podataka
le = preprocessing.LabelEncoder()


# Izvlačenje podataka u kolone (redove)
Administrative = dataset.iloc[:, 0].values
Administrative_Duration = dataset.iloc[:, 1].values
Informational = dataset.iloc[:, 2].values
Informational_Duration = dataset.iloc[:, 3].values
ProductRelated = dataset.iloc[:, 4].values
ProductRelated_Duration = dataset.iloc[:, 5].values
BounceRates = dataset.iloc[:, 6].values
ExitRates = dataset.iloc[:, 7].values
PageValues = dataset.iloc[:, 8].values
SpecialDay = dataset.iloc[:, 9].values
Month = dataset.iloc[:, 10].values
Region = dataset.iloc[:, 11].values
VisitorType = dataset.iloc[:, 13].values
Weekend = dataset.iloc[:, 15].values
Revenue = dataset.iloc[:, 16].values


# Transformacija kolona koristeći ugrađenu funkciju "fit_transform" - vrši kodiranje kolona
Administrative_encoded = le.fit_transform(Administrative)
Administrative_Duration_encoded = le.fit_transform(Administrative_Duration)
Informational_encoded = le.fit_transform(Informational)
Informational_Duration_encoded = le.fit_transform(Informational_Duration)
ProductRelated_encoded = le.fit_transform(ProductRelated)
ProductRelated_Duration_encoded = le.fit_transform(ProductRelated_Duration)
BounceRates_encoded = le.fit_transform(BounceRates)
ExitRates_encoded = le.fit_transform(ExitRates)
PageValues_encoded = le.fit_transform(PageValues)
SpecialDay_encoded = le.fit_transform(SpecialDay)
Month_encoded = le.fit_transform(Month)
Region_encoded = le.fit_transform(Region)
VisitorType_encoded = le.fit_transform(VisitorType)
Weekend_encoded = le.fit_transform(Weekend)
Revenue_encoded = le.fit_transform(Revenue)


# Postavljanje modela zavisno od željenog algoritma
model = GaussianNB()
#model = DecisionTreeClassifier(criterion="entropy")
#model = KNeighborsClassifier(n_neighbors=30, p=2, metric='euclidean')


X_train, X_test, y_train, y_test = train_test_split(
    list(zip(Administrative_encoded, Administrative_Duration_encoded, Informational_encoded, Informational_Duration_encoded,
             ProductRelated_encoded, ProductRelated_Duration_encoded, BounceRates_encoded, ExitRates_encoded,
             PageValues_encoded, SpecialDay_encoded, Month_encoded,Region_encoded,VisitorType_encoded,Weekend_encoded)),
    Revenue_encoded,
    test_size=0.2, random_state=0)


# Treniranje modela
model.fit(list(X_train), y_train)

# Ispisivanje vrijednosti osnovnih metrika
y_pred = model.predict(X_test)
print("Model accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("Model F1 score: ", f1_score(y_test, y_pred))

# Ispisivanje matrice konfuzija
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Uključivanje biblioteke "seaborn" u cilju prikazivanje stvarne slike
import seaborn as sns

group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten() / np.sum(cm)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names, group_counts, group_percentages)]

labels = np.asarray(labels).reshape(2, 2)

ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Reds')

ax.set_title('Seaborn Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')


ax.xaxis.set_ticklabels(['False', 'True'])
ax.yaxis.set_ticklabels(['False', 'True'])

# Vizuelni prikaz matrice konfuzija
plt.show()
