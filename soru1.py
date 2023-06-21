import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Veri setini yükle
data = pd.read_csv('data1.csv', sep=';')

# Ondalık ayracı düzelt
data = data.replace(',', '.', regex=True)

# Öznitelikler ve etiketleri ayır
X = data.drop('class', axis=1).astype(float)
y = data['class']

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PCA ile öznitelik çıkarımı
pca = PCA(n_components=10)  # Yeni öznitelik sayısı
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# KNN sınıflandırması
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, y_train)
y_pred = knn.predict(X_test_pca)

# Karşıtlık matrisi (Confusion Matrix) ve doğruluk (Accuracy) değeri
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Karşıtlık Matrisi:")
print(cm)
print("Doğruluk (Accuracy) Değeri: {:.2f}".format(accuracy))
