#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#%% 
#Membaca file citrus.csv
data = pd.read_csv('citrus.csv')
print(data.head(20))

# %%
# Filter data numerik
data_numerik = data.select_dtypes(include=['int64', 'float64']) 

# %%
# Visualisasi heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data_numerik.corr(), annot=True, cmap='coolwarm', fmt='.6f')
plt.title('Correlation Heatmap')
plt.show()

# %%
# Split training dan testing data
from sklearn.model_selection import train_test_split
x = data.drop(columns=['name'], axis=1)
y = data['name']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# %%
#Latihan model dan prediksi
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# %%
# Latih model
model = DecisionTreeClassifier(random_state=10)
model.fit(x_train, y_train)

# %%
# Prediksi data uji
y_pred = model.predict(x_test)

# %%
# Print classification report
print(classification_report(y_test, y_pred))

# %%
# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7,7))

sns.set(font_scale=1.4) # for label size
sns.heatmap(cm, ax=ax,annot=True, annot_kws={"size": 16}) # font size

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

# %%
features = ['diameter', 'weight', 'red', 'green', 'blue']

#%%
# Visualisasi pohon
from sklearn import tree
fig, ax = plt.subplots(figsize=(12, 12))
tree.plot_tree(model, feature_names=features)
plt.show()

# %%
# Contoh data baru
new_data = pd.DataFrame({
    'diameter': [10.0],
    'weight': [150.0],
    'red': [164],
    'green': [78],
    'blue': [10]
})

prediction = model.predict(new_data)
print(f"Predicted class for the new data: {prediction[0]}")

# %%
