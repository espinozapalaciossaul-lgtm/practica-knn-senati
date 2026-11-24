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

tab1, tab2 = st.tabs(["01 - Satisfacción del cliente", "02 - Inversión en software"])

# ==================== TAB 1: SATISFACCIÓN DEL CLIENTE ====================
with tab1:
    np.random.seed(42)
    n = 80
    df1 = pd.DataFrame({
        'valoracion': np.random.randint(1, 6, n),
        'tiempo_espera': np.random.randint(5, 60, n),
        'problemas_reportados': np.random.randint(0, 5, n)
    })
    df1['satisfecho'] = np.where(
        (df1['valoracion'] >= 4) & (df1['tiempo_espera'] <= 25) & (df1['problemas_reportados'] <= 1),
        1, 0
    )
    ruido1 = np.random.choice(n, 6, replace=False)
    df1.loc[ruido1, 'satisfecho'] = 1 - df1.loc[ruido1, 'satisfecho']

    X1 = df1[['valoracion', 'tiempo_espera', 'problemas_reportados']]
    y1 = df1['satisfecho']
    scaler1 = StandardScaler()
    X1_scaled = scaler1.fit_transform(X1)
    knn1 = KNeighborsClassifier(n_neighbors=3)
    knn1.fit(X1_scaled, y1)
    acc1 = accuracy_score(y1, knn1.predict(X1_scaled))

    c1, c2, c3 = st.columns(3)
    c1.metric("Exactitud", f"{acc1*100:.1f}%")
    c2.metric("Registros", n)
    c3.metric("Vecinos (K)", 3)

    left, right = st.columns(2)
    with left:
        st.subheader("Simulador de predicción")
        val = st.slider("Valoración (1-5)", 1, 5, 5, key="val")
        tiempo = st.slider("Tiempo de espera (min)", 5, 60, 12, key="tiempo")
        problemas = st.slider("Problemas reportados", 0, 5, 0, key="prob")

        if st.button("Ejecutar predicción KNN", key="btn1", type="primary"):
            entrada = scaler1.transform([[val, tiempo, problemas]])
            pred = knn1.predict(entrada)[0]
            dist, idx = knn1.kneighbors(entrada)
            st.markdown("### RESULTADO DE LA PREDICCIÓN")
            if pred == 1:
                st.success("**Sí**\n\nCliente satisfecho")
            else:
                st.error("**No**\n\nCliente no satisfecho")
            st.markdown("### Votación de los vecinos")
            for i, (d, ix) in enumerate(zip(dist[0], idx[0])):
                voto = "Sí" if y1.iloc[ix] == 1 else "No"
                st.write(f"Vecino {i+1}: **{voto}** - distancia {d:.3f}")

    with right:
        st.subheader("Evaluación del modelo")
        Xtr, Xte, ytr, yte = train_test_split(X1, y1, test_size=0.25, random_state=42, stratify=y1)
        sc = StandardScaler()
        knn_temp = KNeighborsClassifier(n_neighbors=3)
        knn_temp.fit(sc.fit_transform(Xtr), ytr)
        ypred = knn_temp.predict(sc.transform(Xte))
        cm = confusion_matrix(yte, ypred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No', 'Sí'], yticklabels=['No', 'Sí'])
        ax.set_xlabel('Predicho')
        ax.set_ylabel('Real')
        st.pyplot(fig)

# ==================== TAB 2: INVERSIÓN EN SOFTWARE ====================
with tab2:
    np.random.seed(123)
    df2 = pd.DataFrame({
        'presupuesto_miles': np.random.randint(50, 300, n),
        'num_empleados': np.random.randint(10, 200, n),
        'procesos_manuales': np.random.randint(1, 20, n)
    })
    df2['invierte'] = np.where(
        (df2['presupuesto_miles'] >= 100) & 
        (df2['num_empleados'] >= 40) & 
        (df2['procesos_manuales'] >= 8),
        1, 0
    )
    ruido2 = np.random.choice(n, 4, replace=False)
    df2.loc[ruido2, 'invierte'] = 1 - df2.loc[ruido2, 'invierte']

    X2 = df2[['presupuesto_miles', 'num_empleados', 'procesos_manuales']]
    y2 = df2['invierte']
    scaler2 = StandardScaler()
    X2_scaled = scaler2.fit_transform(X2)
    knn2 = KNeighborsClassifier(n_neighbors=3)
    knn2.fit(X2_scaled, y2)
    acc2 = accuracy_score(y2, knn2.predict(X2_scaled))

    c1, c2, c3 = st.columns(3)
    c1.metric("Exactitud", f"{acc2*100:.1f}%")
    c2.metric("Registros", n)
    c3.metric("Vecinos (K)", 3)

    left, right = st.columns(2)
    with left:
        st.subheader("Simulador de predicción")
        presup = st.number_input("Presupuesto (miles $)", min_value=50, max_value=500, value=1200, step=10)
        empleados = st.number_input("Número de empleados", min_value=5, max_value=500, value=80)
        procesos = st.number_input("Procesos manuales", min_value=0, max_value=30, value=15)

        if st.button("Ejecutar predicción KNN", key="btn2", type="primary"):
            entrada = scaler2.transform([[presup, empleados, procesos]])
            pred = knn2.predict(entrada)[0]
            dist, idx = knn2.kneighbors(entrada)
            st.markdown("### RESULTADO DE LA PREDICCIÓN")
            if pred == 1:
                st.success("**Sí**\n\nEl empresario invierte")
            else:
                st.error("**No**\n\nEl empresario no invierte")
            st.markdown("### Votación de los vecinos")
            for i, (d, ix) in enumerate(zip(dist[0], idx[0])):
                voto = "Sí" if y2.iloc[ix] == 1 else "No"
                st.write(f"Vecino {i+1}: **{voto}** - distancia {d:.3f}")

    with right:
        st.subheader("Evaluación del modelo")
        Xtr, Xte, ytr, yte = train_test_split(X2, y2, test_size=0.25, random_state=42, stratify=y2)
        sc = StandardScaler()
        knn_temp = KNeighborsClassifier(n_neighbors=3)
        knn_temp.fit(sc.fit_transform(Xtr), ytr)
        ypred = knn_temp.predict(sc.transform(Xte))
        cm = confusion_matrix(yte, ypred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                    xticklabels=['No', 'Sí'], yticklabels=['No', 'Sí'])
        ax.set_xlabel('Predicho')
        ax.set_ylabel('Real')
        st.pyplot(fig)

    # Gráfico de dispersión
    st.subheader("Datos de entrenamiento - Inversión en software")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    colores = ['red' if x == 0 else 'blue' for x in y2]
    ax2.scatter(df2['presupuesto_miles'], df2['num_empleados'], c=colores, alpha=0.7)
    ax2.set_xlabel("presupuesto_miles")
    ax2.set_ylabel("empleados")
    ax2.legend(['No', 'Sí'], title="Invierte")
    st.pyplot(fig2)