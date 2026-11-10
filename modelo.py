import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

FEATURES = ["monto", "ratio_monto", "hora", "pais_distinto",
            "dispositivo_nuevo", "tx_ultima_hora", "antiguedad_dias"]

def generar_datos(n=6000, fraude_pct=0.06, seed=42):
    """Crea transacciones legitimas y fraudulentas (datos sinteticos)."""
    rng = np.random.default_rng(seed)
    n_f = int(n * fraude_pct)
    n_l = n - n_f
            
    prom_l = rng.lognormal(4.2, 0.5, n_l)
    monto_l = prom_l * rng.lognormal(0, 0.4, n_l)
    hora_l = np.clip(rng.normal(15, 4, n_l), 0, 23).astype(int)
    legit = pd.DataFrame({
        "monto": monto_l, "promedio": prom_l, "hora": hora_l,
        "pais_distinto": rng.binomial(1, 0.03, n_l),
        "dispositivo_nuevo": rng.binomial(1, 0.05, n_l),
        "tx_ultima_hora": rng.poisson(0.5, n_l),
        "antiguedad_dias": rng.uniform(30, 3000, n_l),
        "fraude": 0})

    prom_f = rng.lognormal(4.2, 0.5, n_f)
    monto_f = prom_f * rng.lognormal(1.5, 0.8, n_f)
    noche = rng.random(n_f) < 0.6
    hora_f = np.where(noche, rng.choice([0, 1, 2, 3, 4, 5], n_f), rng.integers(6, 24, n_f))
    fraude = pd.DataFrame({
        "monto": monto_f, "promedio": prom_f, "hora": hora_f,
        "pais_distinto": rng.binomial(1, 0.6, n_f),
        "dispositivo_nuevo": rng.binomial(1, 0.7, n_f),
        "tx_ultima_hora": rng.poisson(3, n_f),
        "antiguedad_dias": rng.exponential(200, n_f) + 1,
        "fraude": 1})

    df = pd.concat([legit, fraude], ignore_index=True).sample(frac=1, random_state=seed)
    df["ratio_monto"] = df["monto"] / df["promedio"]
    return df.reset_index(drop=True)


def entrenar():
    df = generar_datos()
    train, test = train_test_split(df, test_size=0.3, stratify=df["fraude"], random_state=1)
    modelo = RandomForestClassifier(n_estimators=150, class_weight="balanced",
                                    random_state=1, n_jobs=-1)
    modelo.fit(train[FEATURES], train["fraude"])
    return modelo, test.reset_index(drop=True)


def metricas(modelo, df, umbral=0.5):
    """Fraudes detectados, no detectados, falsos positivos, precision, etc."""
    prob = modelo.predict_proba(df[FEATURES])[:, 1]
    pred = (prob >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(df["fraude"], pred, labels=[0, 1]).ravel()
    return {
        "Fraudes detectados": int(tp),
        "Fraudes no detectados": int(fn),
        "Falsos positivos": int(fp),
        "Legitimas correctas": int(tn),
        "Precision general": float(round((tp + tn) / len(df), 4)),
        "Sensibilidad (recall)": float(round(tp / (tp + fn), 4)) if (tp + fn) else 0.0,
    }


def puntaje(modelo, tx: dict):
    fila = dict(tx)
    fila["ratio_monto"] = fila["monto"] / max(fila["promedio"], 1)
    X = pd.DataFrame([fila])[FEATURES]
    return float(modelo.predict_proba(X)[0, 1])

def prueba_datos_manipulados(modelo, test, umbral=0.5):
    """El defraudador oculta pais, dispositivo y baja el monto para no ser detectado."""
    legit = test[test["fraude"] == 0]
    fr = test[test["fraude"] == 1].copy()
    fr["pais_distinto"] = 0
    fr["dispositivo_nuevo"] = 0
    fr["monto"] = fr["promedio"] * 1.2
    fr["ratio_monto"] = 1.2
    return metricas(modelo, pd.concat([legit, fr]), umbral)


def prueba_fraude_nuevo(modelo, test, umbral=0.5, seed=7):
    """Nuevo patron: muchas compras muy pequenas seguidas (card testing)."""
    rng = np.random.default_rng(seed)
    legit = test[test["fraude"] == 0]
    n = int((test["fraude"] == 1).sum())
    prom = rng.lognormal(4.2, 0.5, n)
    monto = rng.uniform(1, 5, n)
    nuevo = pd.DataFrame({
        "monto": monto, "promedio": prom, "ratio_monto": monto / prom,
        "hora": rng.integers(0, 24, n), "pais_distinto": 0, "dispositivo_nuevo": 0,
        "tx_ultima_hora": rng.integers(15, 30, n),
        "antiguedad_dias": rng.uniform(30, 3000, n), "fraude": 1})
    return metricas(modelo, pd.concat([legit, nuevo]), umbral)

def prueba_volumen(modelo, test, n_tx=10000):
    """Muchas transacciones a la vez + latencia por transaccion individual."""
    lote = test.sample(n_tx, replace=True, random_state=3)
    t0 = time.perf_counter()
    modelo.predict_proba(lote[FEATURES])
    t_lote = time.perf_counter() - t0

    muestra = test.sample(100, random_state=4)
    tiempos = []
    for _, fila in muestra.iterrows():
        t0 = time.perf_counter()
        modelo.predict_proba(pd.DataFrame([fila[FEATURES]]))
        tiempos.append((time.perf_counter() - t0) * 1000)
    return {
        "Transacciones en lote": n_tx,
        "Tiempo del lote (s)": round(t_lote, 3),
        "Transacciones por segundo": int(n_tx / t_lote),
        "Latencia promedio por transaccion (ms)": round(float(np.mean(tiempos)), 2),
        "Latencia maxima (ms)": round(float(np.max(tiempos)), 2),
    }
