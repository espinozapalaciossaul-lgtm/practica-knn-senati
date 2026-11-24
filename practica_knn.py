import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# ====================== 1. SATISFACCIÓN DEL CLIENTE ======================
print("=" * 60)
print("1. PREDICCIÓN DE SATISFACCIÓN DEL CLIENTE")
print("=" * 60)

np.random.seed(42)
n = 300

# Crear datos
data_cliente = pd.DataFrame({
    'calidad_servicio': np.random.randint(1, 11, n),
    'tiempo_respuesta_min': np.random.randint(5, 120, n),
    'precio_percibido': np.random.randint(1, 11, n),
    'atencion_personalizada': np.random.randint(1, 11, n),
    'recomendaria': np.random.randint(0, 2, n)
})

# Crear variable objetivo
data_cliente['satisfecho'] = np.where(
    (data_cliente['calidad_servicio'] >= 7) &
    (data_cliente['tiempo_respuesta_min'] <= 45) &
    (data_cliente['atencion_personalizada'] >= 6),
    1, 0
)

# Añadir ruido de forma segura
ruido = np.random.choice(n, 30, replace=False)
data_cliente.loc[ruido, 'satisfecho'] = 1 - data_cliente.loc[ruido, 'satisfecho']

X1 = data_cliente.drop('satisfecho', axis=1)
y1 = data_cliente['satisfecho']

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.25, random_state=42, stratify=y1
)

scaler1 = StandardScaler()
X1_train_scaled = scaler1.fit_transform(X1_train)
X1_test_scaled = scaler1.transform(X1_test)

knn1 = KNeighborsClassifier(n_neighbors=5)
knn1.fit(X1_train_scaled, y1_train)
y1_pred = knn1.predict(X1_test_scaled)

print(f"Accuracy: {accuracy_score(y1_test, y1_pred):.4f}")
print("\nReporte de clasificación:")
print(classification_report(y1_test, y1_pred, target_names=['No satisfecho', 'Satisfecho']))

# ====================== 2. INVERSIÓN EN SOFTWARE ======================
print("\n" + "=" * 60)
print("2. PREDICCIÓN DE INVERSIÓN EN SOFTWARE")
print("=" * 60)

data_empresa = pd.DataFrame({
    'tamaño_empresa': np.random.randint(5, 500, n),
    'nivel_automatizacion_actual': np.random.randint(1, 11, n),
    'presupuesto_disponible': np.random.randint(10, 500, n),
    'roi_esperado': np.random.uniform(5, 40, n),
    'problemas_procesos': np.random.randint(1, 11, n)
})

data_empresa['invierte'] = np.where(
    (data_empresa['tamaño_empresa'] >= 50) &
    (data_empresa['nivel_automatizacion_actual'] <= 5) &
    (data_empresa['roi_esperado'] >= 15) &
    (data_empresa['problemas_procesos'] >= 6),
    1, 0
)

# Ruido seguro
ruido2 = np.random.choice(n, 25, replace=False)
data_empresa.loc[ruido2, 'invierte'] = 1 - data_empresa.loc[ruido2, 'invierte']

X2 = data_empresa.drop('invierte', axis=1)
y2 = data_empresa['invierte']

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.25, random_state=42, stratify=y2
)

scaler2 = StandardScaler()
X2_train_scaled = scaler2.fit_transform(X2_train)
X2_test_scaled = scaler2.transform(X2_test)

knn2 = KNeighborsClassifier(n_neighbors=7)
knn2.fit(X2_train_scaled, y2_train)
y2_pred = knn2.predict(X2_test_scaled)

print(f"Accuracy: {accuracy_score(y2_test, y2_pred):.4f}")
print("\nReporte de clasificación:")
print(classification_report(y2_test, y2_pred, target_names=['No invierte', 'Invierte']))

# ====================== GRÁFICOS ======================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm1 = confusion_matrix(y1_test, y1_pred)
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Satisfecho', 'Satisfecho'],
            yticklabels=['No Satisfecho', 'Satisfecho'])
axes[0].set_title('Satisfacción del Cliente')
axes[0].set_ylabel('Real')
axes[0].set_xlabel('Predicho')

cm2 = confusion_matrix(y2_test, y2_pred)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['No Invierte', 'Invierte'],
            yticklabels=['No Invierte', 'Invierte'])
axes[1].set_title('Inversión en Software')
axes[1].set_ylabel('Real')
axes[1].set_xlabel('Predicho')

plt.tight_layout()
plt.savefig('resultados_knn.png', dpi=150)
plt.show()

print("\n✅ Listo. Gráfico guardado como resultados_knn.png")