# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np

app = FastAPI()

class CreanceItem(BaseModel):
    Date: str
    Centre_Nom: str
    Type_de_Cr_ances: str
    Montant: float

@app.post("/detect-anomalies")
async def detect_anomalies(data: List[CreanceItem]):
    df = pd.DataFrame([item.dict() for item in data])
    df["Date"] = pd.to_datetime(df["Date"])
    df["Montant"] = pd.to_numeric(df["Montant"], errors="coerce").fillna(0)

    df_grouped = df.groupby(['Date', 'Centre_Nom', 'Type_de_Cr_ances'])['Montant'].sum().reset_index()
    df_pivot = df_grouped.pivot(index=['Date', 'Centre_Nom'], columns='Type_de_Cr_ances', values='Montant').fillna(0).reset_index()

    anomalies = []
    for centre in df_pivot['Centre_Nom'].unique():
        centre_data = df_pivot[df_pivot['Centre_Nom'] == centre].sort_values('Date')

        for type_col in ['RC', 'RO', 'Patients']:
            if type_col not in centre_data.columns:
                continue

            series = centre_data[type_col]
            rolling_mean = series.rolling(window=7).mean()
            rolling_std = series.rolling(window=7).std()

            z_scores = (series - rolling_mean) / rolling_std
            z_scores = z_scores.fillna(0)

            threshold = 2.0
            anomalies_detected = z_scores[abs(z_scores) > threshold]

            for idx in anomalies_detected.index:
                anomalies.append({
                    "Centre": centre,
                    "Type_de_Creance": type_col,
                    "Date": str(centre_data.iloc[idx]['Date'].date()),
                    "Montant": float(series.iloc[idx]),
                    "Z_score": float(z_scores.iloc[idx])
                })

    return {"anomalies": anomalies}
