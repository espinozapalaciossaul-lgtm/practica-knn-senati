import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==================== DATOS ====================
np.random.seed(42)
n = 300
df = pd.DataFrame({
    'calidad_atencion': np.random.randint(1, 11, n),
    'tiempo_espera': np.random.randint(5, 80, n),
    'precio_percibido': np.random.randint(1, 11, n),
    'se_resolvio': np.random.randint(0, 2, n),
    'reclamos_previos': np.random.randint(0, 6, n)
})

df['satisfecho'] = np.where(
    (df['calidad_atencion'] >= 7) &
    (df['tiempo_espera'] <= 30) &
    (df['precio_percibido'] <= 6) &
    (df['se_resolvio'] == 1),
    1, 0
)

# Ruido
idx = np.random.choice(n, 25, replace=False)
df.loc[idx, 'satisfecho'] = 1 - df.loc[idx, 'satisfecho']

X = df.drop('satisfecho', axis=1)
y = df['satisfecho']

# Buscar mejor K
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
mejor_k = 12
mejor_score = 0
for k in range(3, 21, 2):
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_scaled, y, cv=5)
    if scores.mean() > mejor_score:
        mejor_score = scores.mean()
        mejor_k = k

knn = KNeighborsClassifier(n_neighbors=mejor_k)
knn.fit(X_scaled, y)
exactitud = mejor_score * 100

# ==================== INTERFAZ ====================
root = tk.Tk()
root.title("KNN - EJERCICIO 1 · Satisfacción del cliente")
root.geometry("1100x650")
root.configure(bg="#f0f4f8")

# Título
tk.Label(root, text="EJERCICIO 1 · Satisfacción del cliente", 
         font=("Arial", 16, "bold"), bg="#1e4d8c", fg="white", pady=10).pack(fill="x")

main_frame = tk.Frame(root, bg="#f0f4f8")
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

# ===== PANEL IZQUIERDO =====
left = tk.Frame(main_frame, bg="white", relief="ridge", bd=1)
left.pack(side="left", fill="y", padx=(0, 10))

tk.Label(left, text="Modelo entrenado", font=("Arial", 11, "bold"), bg="white").pack(pady=(10, 5))

info_frame = tk.Frame(left, bg="white")
info_frame.pack(pady=5)
tk.Label(info_frame, text=f"{mejor_k}", font=("Arial", 14, "bold"), bg="white").grid(row=0, column=0, padx=20)
tk.Label(info_frame, text=f"{exactitud:.1f}%", font=("Arial", 14, "bold"), bg="white").grid(row=0, column=1, padx=20)
tk.Label(info_frame, text="300", font=("Arial", 14, "bold"), bg="white").grid(row=0, column=2, padx=20)
tk.Label(info_frame, text="Mejor K", font=("Arial", 8), bg="white").grid(row=1, column=0)
tk.Label(info_frame, text="Exactitud", font=("Arial", 8), bg="white").grid(row=1, column=1)
tk.Label(info_frame, text="Registros", font=("Arial", 8), bg="white").grid(row=1, column=2)
tk.Label(left, text="Entrenamiento: 240     Prueba: 60", font=("Arial", 8), bg="white").pack()

# Datos del caso nuevo
tk.Label(left, text="Datos del caso nuevo", font=("Arial", 11, "bold"), bg="white").pack(pady=(15, 5))

tk.Label(left, text="Calidad de atención (1-10)", bg="white").pack(anchor="w", padx=10)
calidad = tk.Scale(left, from_=1, to=10, orient="horizontal", length=250)
calidad.set(9)
calidad.pack()

tk.Label(left, text="Tiempo de espera (min)", bg="white").pack(anchor="w", padx=10)
tiempo = tk.Scale(left, from_=5, to=80, orient="horizontal", length=250)
tiempo.set(10)
tiempo.pack()

tk.Label(left, text="Precio percibido (1-10)", bg="white").pack(anchor="w", padx=10)
precio = tk.Scale(left, from_=1, to=10, orient="horizontal", length=250)
precio.set(4)
precio.pack()

tk.Label(left, text="¿Se resolvió su problema?", bg="white").pack(anchor="w", padx=10)
resolvio_var = tk.StringVar(value="Si")
tk.Radiobutton(left, text="Si", variable=resolvio_var, value="Si", bg="white").pack(anchor="w", padx=20)
tk.Radiobutton(left, text="No", variable=resolvio_var, value="No", bg="white").pack(anchor="w", padx=20)

tk.Label(left, text="Número de reclamos previos", bg="white").pack(anchor="w", padx=10)
reclamos = tk.Scale(left, from_=0, to=10, orient="horizontal", length=250)
reclamos.set(0)
reclamos.pack()

resultado_label = tk.Label(left, text="", font=("Arial", 12, "bold"), bg="white", fg="green")
resultado_label.pack(pady=10)

def predecir():
    resolvio = 1 if resolvio_var.get() == "Si" else 0
    entrada = np.array([[calidad.get(), tiempo.get(), precio.get(), resolvio, reclamos.get()]])
    entrada_scaled = scaler.transform(entrada)
    pred = knn.predict(entrada_scaled)[0]
    
    if pred == 1:
        resultado_label.config(text="✓ SATISFECHO", fg="green", bg="#d4edda")
    else:
        resultado_label.config(text="✗ NO SATISFECHO", fg="red", bg="#f8d7da")

tk.Button(left, text="PREDECIR", command=predecir, bg="#0d6efd", fg="white", 
          font=("Arial", 11, "bold"), width=20, height=2).pack(pady=10)

# ===== PANEL DERECHO (Gráfico) =====
right = tk.Frame(main_frame, bg="white")
right.pack(side="right", fill="both", expand=True)

fig, ax = plt.subplots(figsize=(7, 5))
colores = ['#ff6b6b' if s == 0 else '#4ecdc4' for s in y]
ax.scatter(df['calidad_atencion'], df['tiempo_espera'], c=colores, alpha=0.6, s=30)
ax.set_xlabel("Calidad de atención (1-10)")
ax.set_ylabel("Tiempo de espera (min)")
ax.set_title("Distribución de los datos (★ = caso nuevo)")
ax.legend(["No satisfecho", "Satisfecho"], loc="upper right")

canvas = FigureCanvasTkAgg(fig, master=right)
canvas.draw()
canvas.get_tk_widget().pack(fill="both", expand=True)

root.mainloop()