#Importamos librerias
import pandas as pd
import numpy as np
import json
import pickle as pkl

# --- Función 1: answer_questions ---
def answer_questions(df_train, df_users, df_products):
    """
    Resuelve las preguntas Q1-Q7 de forma optimizada, sin hacer merges masivos.
    Devuelve un diccionario con las respuestas formateadas según el ejemplo.
    """
    target = {}

    # Q1: Encontrar el partnumber con color_id=3 y discount=1, ordenado por family
    q1_result = (
        df_products[(df_products['color_id'] == 3) & (df_products['discount'] == 1)]
        .sort_values('family')
        .iloc[0]['partnumber']
    )
    target['query_1'] = {"partnumber": int(q1_result)}

    # Q2: Encontrar el país con más usuarios con M<500 y luego el usuario con menor F, mayor R, menor user_id
    filtered_users = df_users[df_users['M'] < 500]
    top_country = (
        filtered_users['country'].value_counts().idxmax()
    )
    user_info = (
        filtered_users[filtered_users['country'] == top_country]
        .sort_values(['F', 'R', 'user_id'], ascending=[True, False, True])
        .iloc[0]
    )
    target['query_2'] = {"user_id": int(user_info['user_id'])}

    # Q3: Promediar visitas antes de añadir al carrito
    visits_before_cart = (
        df_train[df_train['add_to_cart'] == 1]
        .groupby('partnumber')['session_id'].count()
        .mean()
    )
    target['query_3'] = {"average_previous_visits": round(visits_before_cart, 2)}

    # Q4: Tipo de dispositivo más frecuente al comprar productos con descuento
    filtered_train = df_train[df_train['add_to_cart'] == 1]
    filtered_products = df_products[df_products['discount'] == 1]
    merged_data = pd.merge(
        filtered_train[['partnumber', 'device_type']],
        filtered_products[['partnumber']],
        on='partnumber',
        how='inner'
    )
    most_frequent_device = merged_data['device_type'].value_counts().idxmax()
    target['query_4'] = {"device_type": int(most_frequent_device)}

    # Q5: Usuario con más interacciones desde device_type=3 entre los top 3 F por país
    top_users_by_country = (
        df_users.groupby(['country', 'user_id'])['F']
        .first()
        .reset_index()
        .sort_values(['country', 'F'], ascending=[True, False])
        .groupby('country')
        .head(3)
    )

    device_3_interactions = (
        df_train[
            (df_train['device_type'] == 3) & 
            (df_train['user_id'].isin(top_users_by_country['user_id']))
        ]
        .groupby('user_id')['partnumber']
        .nunique()
    )

    most_active_user = device_3_interactions.idxmax()
    target['query_5'] = {"user_id": int(most_active_user)}

    # Q6: Familias únicas en interacciones fuera del país de residencia
    user_countries = df_users.groupby('user_id')['country'].unique()
    foreign_interactions = df_train.merge(
        pd.DataFrame({'user_id': user_countries.index, 'resident_countries': user_countries.values}),
        on='user_id'
    )

    foreign_interactions = foreign_interactions[
        ~foreign_interactions.apply(lambda x: x['country'] in x['resident_countries'], axis=1)
    ]

    merged_products = pd.merge(
        foreign_interactions[['partnumber']],
        df_products[['partnumber', 'family']],
        on='partnumber',
        how='inner'
    )

    unique_families = merged_products['family'].nunique()
    target['query_6'] = {"unique_families": int(unique_families)}

    # Q7: Página más frecuente por familia en los primeros 7 días de junio
    june_data = df_train[
        (df_train['date'].str[:7] == '2023-06') & 
        (df_train['date'].str[8:10].astype(int) <= 7) &
        (df_train['add_to_cart'] == 1)
    ]

    merged_data = pd.merge(
        june_data[['partnumber', 'pagetype']],
        df_products[['partnumber', 'family']],
        on='partnumber',
        how='inner'
    )

    most_frequent_pagetype = (
        merged_data.groupby('family')['pagetype']
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )

    target['query_7'] = most_frequent_pagetype

    return {"target": target}

# --- Función 2: save_predictions ---
def save_predictions(answers_dict, filename="predictions_1.json"):
    """
    Guarda el diccionario de respuestas en un archivo JSON.
    Convierte todos los valores no serializables a tipos básicos de Python.
    """

    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convierte únicamente si es un arreglo de NumPy
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]  # Si ya es lista, procesa elementos
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (tuple)):
            return [convert_to_serializable(i) for i in obj]  # Convierte tuplas a listas serializables
        else:
            return obj

    # Convertir todo el diccionario
    serializable_dict = convert_to_serializable(answers_dict)

    # Guardar en JSON
    with open(filename, 'w') as f:
        json.dump(serializable_dict, f, indent=4)

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Cargar datos
    df_train = pd.read_csv('/Users/fernandoarroyo/Desktop/Inditex/Data/train.csv')
    df_users = pd.read_csv('/Users/fernandoarroyo/Desktop/Inditex/Data/users.csv')
    df_products = pd.read_pickle('/Users/fernandoarroyo/Desktop/Inditex/Data/products.pkl')

    # Responder preguntas
    answers = answer_questions(df_train, df_users, df_products)

    # Guardar resultados
    save_predictions(answers)

    print("Predicciones guardadas en 'predictions_1.json'")