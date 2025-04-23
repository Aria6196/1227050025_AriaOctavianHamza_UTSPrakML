### Struktur 
citrus.csv - Dataset yang berisi fitur dan label citrus.
1227050025_AriaOctavianHamza_UTS.py - Script Python berisi seluruh pipeline machine learning.
README.md - Penjelasan tahapan dan langkah proyek ini.

### Membaca Dataset
###### Menampilkan data awal dan memeriksa tipe-tipe data.
data = pd.read_csv('citrus.csv')

### Visualisasi Korelasi
###### Melihat hubungan antar fitur numerik dalam bentuk heatmap untuk memahami korelasi antar fitur.
sns.heatmap(data_numerik.corr(), annot=True, cmap='coolwarm')

### Split Dataset
###### Dataset dibagi menjadi data latih dan data uji dengan rasio 80:20.
x = data.drop(columns=['name'])
y = data['name']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

### Melatih Model
###### Model Decision Tree dilatih menggunakan data latih.
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

### Evaluasi Model
###### Menampilkan classification report, confusion matrix, dan akurasi dari prediksi model.
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

### Visualisasi Confusion Matrix
###### Memvisualisasikan pohon keputusan yang terbentuk dari training data.
tree.plot_tree(model, feature_names=features)

### Prediksi Data Baru
###### Menggunakan model untuk memprediksi kelas citrus dari data baru.
new_data = pd.DataFrame({...})
prediction = model.predict(new_data)
