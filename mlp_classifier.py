from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# MNIST veri setini yükle
transform = transforms.ToTensor()
testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Veriyi düzleştir
X = np.array([img.view(-1).numpy() for img, _ in testset])
y = np.array([label for _, label in testset])

# Train-test ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLP sınıflandırıcı
clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=30, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Rapor ve confusion matrix
print("MLP Classification Report:")
print(classification_report(y_test, y_pred))

os.makedirs("results/plots", exist_ok=True)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=range(10), yticklabels=range(10))
plt.title("Confusion Matrix - MLP")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("results/plots/confusion_matrix_mlp.png")
plt.show()
