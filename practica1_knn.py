import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="SENATI / KNN Lab", layout="wide")

st.title("Laboratorio de predicción inteligente")
st.caption("Clasificación supervisada mediante los 3 vecinos más cercanos.")

# ==================== DATOS SINTÉTICOS (80 registros) ====================
np.random.seed(42)
n = 80

df = pd.DataFrame({
    'valoracion': np.random.randint(1, 6, n),
    'tiempo_espera': np.random.randint(5, 60, n),
    'problemas_reportados': np.random.randint(0, 5, n)
})

# Regla para crear el target (satisfecho)
df['satisfecho'] = np.where(
    (df['valoracion'] >= 4) & 
    (df['tiempo_espera'] <= 25) & 
    (df['problemas_reportados'] <= 1),
    1, 0
)

# Añadir un poco de ruido
ruido = np.random.choice(n, 8, replace=False)
df.loc[ruido, 'satisfecho'] = 1 - df.loc[ruido, 'satisfecho']

X = df[['valoracion', 'tiempo_espera', 'problemas_reportados']]
y = df['satisfecho']

# Entrenar modelo
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)

# Métricas
y_pred = knn.predict(X_scaled)
acc = accuracy_score(y, y_pred)

# ==================== INTERFAZ ====================
col1, col2, col3 = st.columns(3)
col1.metric("Exactitud", f"{acc*100:.1f}%")
col2.metric("Registros", n)
col3.metric("Vecinos (K)", 3)

st.markdown("---")

left, right = st.columns([1, 1])

with left:
    st.subheader("Simulador de predicción")
    valoracion = st.slider("Valoración (1-5)", 1, 5, 5)
    tiempo = st.slider("Tiempo de espera (min)", 5, 60, 12)
    problemas = st.slider("Problemas reportados", 0, 5, 0)

    if st.button("Ejecutar predicción KNN", type="primary", use_container_width=True):
        entrada = scaler.transform([[valoracion, tiempo, problemas]])
        pred = knn.predict(entrada)[0]
        distancias, indices = knn.kneighbors(entrada)

        st.markdown("### RESULTADO DE LA PREDICCIÓN")
        if pred == 1:
            st.success("**Sí**")
            st.write("Cliente satisfecho")
        else:
            st.error("**No**")
            st.write("Cliente no satisfecho")

        st.markdown("### Votación de los vecinos")
        for i, (idx, dist) in enumerate(zip(indices[0], distancias[0])):
            voto = "Sí" if y.iloc[idx] == 1 else "No"
            st.write(f"Vecino {i+1}: **{voto}** - distancia {dist:.3f}")

with right:
    st.subheader("Evaluación del modelo")
    st.caption("Matriz de confusión - conjunto de prueba (25%)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    knn.fit(X_train_s, y_train)
    y_pred_test = knn.predict(X_test_s)

    cm = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No', 'Sí'], yticklabels=['No', 'Sí'])
    ax.set_xlabel('Predicho')
    ax.set_ylabel('Real')
    st.pyplot(fig)