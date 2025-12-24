# Predict Model Notebook

## 1. Importaciones y configuraciones iniciales

import pandas as pd
import numpy as np
import json
import joblib
from lightfm import LightFM
import os
import pickle
import logging
from datetime import datetime
from train_model import HybridRecommender

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Definir rutas absolutas
BASE_PATH = "/Users/fernandoarroyo/Desktop/Inditex/hackathon-inditex-data-recommender"
DATA_PATH = os.path.join(BASE_PATH, "data")
MODEL_PATH = os.path.join(BASE_PATH, "models")
PREDICTIONS_PATH = os.path.join(BASE_PATH, "predictions/predictions_3.json")

# Funciones auxiliares

def load_model(model_path):
    """Carga el modelo LightFM entrenado desde un archivo."""
    print("Cargando el modelo entrenado...")
    model = joblib.load(model_path)
    print("Modelo cargado correctamente.")
    return model

def load_test_data(test_path):
    """Carga y prepara el dataset de test."""
    print("Cargando el dataset de test...")
    
    # Encontrar el archivo de test más reciente
    processed_path = os.path.dirname(test_path)
    files = os.listdir(processed_path)
    test_file = sorted([f for f in files if f.startswith('test_prepared')])[-1]
    test_path = os.path.join(processed_path, test_file)
    
    # Cargar datos
    df_test = pd.read_parquet(test_path)

    # Convertir columnas de fechas si existen
    if 'date' in df_test.columns:
        df_test['date'] = pd.to_datetime(df_test['date'])
    if 'timestamp_local' in df_test.columns:
        df_test['timestamp_local'] = pd.to_datetime(df_test['timestamp_local'])

    print("Dataset de test cargado y preparado:")
    print(df_test.head())
    return df_test

def validate_consistency(model, user_map, item_map):
    """Valida que los mapas y el modelo son consistentes."""
    print("Validando consistencia entre el modelo y los mapas...")
    expected_items = model.item_embeddings.shape[0]
    expected_users = model.user_embeddings.shape[0]
    
    if len(item_map) != expected_items:
        print(f"Advertencia: El mapa de ítems tiene {len(item_map)} ítems, pero el modelo espera {expected_items} ítems.")
        return False
    if len(user_map) != expected_users:
        print(f"Advertencia: El mapa de usuarios tiene {len(user_map)} usuarios, pero el modelo espera {expected_users} usuarios.")
        return False
    
    print("Consistencia validada entre mapas y el modelo.")
    return True

def generate_recommendations(model, df_test, user_map, item_map):
    """Genera las recomendaciones para los usuarios en el dataset de test."""
    print("Generando recomendaciones...")
    recs_dict = {"target": {}}
    
    # Invertir el mapeo de items para obtener partnumbers reales
    item_map_inv = {v: k for k, v in item_map.items()}
    
    # Obtener todos los partnumbers disponibles como lista de integers normales
    all_partnumbers = [int(x) for x in item_map.keys()]
    
    # Agrupar por session_id y tomar el primer user_id
    session_users = df_test.groupby('session_id')['user_id'].first()
    
    for session_id, user_id in session_users.items():
        try:
            user_index = user_map.get(str(user_id), None)  # Convertir user_id a string
            
            if user_index is None:
                # Para usuarios nuevos, seleccionar 5 productos aleatorios diferentes
                top_partnumbers = np.random.choice(all_partnumbers, size=5, replace=False)
            else:
                # Predecir scores para todos los items conocidos
                scores = model.predict(user_index, np.arange(len(item_map)))
                top_indices = np.argsort(-scores)[:5]  # Top 5 índices
                top_partnumbers = [int(item_map_inv[idx]) for idx in top_indices]  # Convertir a int normal
            
            # Convertir session_id y partnumbers a integers normales
            recs_dict["target"][int(session_id)] = [int(x) for x in top_partnumbers]
            
        except Exception as e:
            print(f"Error generando recomendaciones para session_id {session_id}: {str(e)}")
            # En caso de error, seleccionar 5 productos aleatorios diferentes
            top_partnumbers = np.random.choice(all_partnumbers, size=5, replace=False)
            recs_dict["target"][int(session_id)] = [int(x) for x in top_partnumbers]
    
    print("Recomendaciones generadas correctamente.")
    return recs_dict

def load_latest_model():
    """Carga el modelo más reciente."""
    try:
        model_files = [f for f in os.listdir(MODEL_PATH) if f.startswith('hybrid_recommender')]
        if not model_files:
            raise FileNotFoundError("No se encontró ningún modelo entrenado")
        
        latest_model = sorted(model_files)[-1]
        model_file = os.path.join(MODEL_PATH, latest_model)
        logging.info(f"Cargando modelo: {model_file}")
        
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        logging.info("Modelo cargado exitosamente")
        return model
    except Exception as e:
        logging.error(f"Error al cargar el modelo: {str(e)}")
        raise

def get_recommendations(model, session_id, test_df, products_df, users_df, n_recommendations=5):
    """Genera recomendaciones para una sesión."""
    session_data = test_df[test_df['session_id'] == session_id]
    user_id = session_data['user_id'].iloc[0] if not session_data.empty else -1
    
    # Obtener productos vistos en la sesión
    viewed_products = set(session_data['partnumber'].unique())
    
    # Obtener candidatos similares a los productos vistos
    candidate_products = set()
    for product_id in viewed_products:
        if product_id in model.valid_product_indices:
            product_idx = model.valid_product_indices.index(product_id)
            _, indices = model.item_similarity_model.kneighbors(
                model.product_embeddings[product_idx].reshape(1, -1)
            )
            candidate_products.update([model.valid_product_indices[i] for i in indices[0]])
    
    # Si no hay productos vistos o candidatos, usar productos populares
    if not candidate_products:
        candidate_products = set(sorted(
            model.product_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )[:100])
    
    # Crear features para ranking
    ranking_features = []
    ranking_products = []
    
    for product_id in candidate_products:
        if product_id not in viewed_products:  # No recomendar productos ya vistos
            try:
                features = model.create_features(
                    user_id,
                    session_id,
                    product_id,
                    test_df,
                    products_df,
                    users_df
                )
                ranking_features.append(features)
                ranking_products.append(product_id)
            except Exception as e:
                logging.warning(f"Error creando features para producto {product_id}: {str(e)}")
    
    if not ranking_features:
        # Si no hay features válidas, retornar productos populares
        popular_products = sorted(
            model.product_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [int(p[0]) for p in popular_products[:n_recommendations]]
    
    # Predecir scores
    X_predict = pd.DataFrame(ranking_features)
    scores = model.ranking_model.predict(X_predict)
    
    # Ordenar productos por score
    product_scores = list(zip(ranking_products, scores))
    product_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Retornar los top N productos
    return [int(p[0]) for p in product_scores[:n_recommendations]]

def generate_predictions():
    """Genera predicciones para todas las sesiones de test."""
    try:
        logging.info("Cargando datos...")
        
        # Cargar datos
        processed_path = os.path.join(DATA_PATH, "processed")
        files = os.listdir(processed_path)
        
        test_file = sorted([f for f in files if f.startswith('test_prepared')])[-1]
        products_file = sorted([f for f in files if f.startswith('products_prepared')])[-1]
        users_file = sorted([f for f in files if f.startswith('users_prepared')])[-1]
        
        test_df = pd.read_parquet(os.path.join(processed_path, test_file))
        products_df = pd.read_parquet(os.path.join(processed_path, products_file))
        users_df = pd.read_parquet(os.path.join(processed_path, users_file))
        
        logging.info("Datos cargados correctamente")
        
        # Cargar modelo
        model = load_latest_model()
        
        # Generar recomendaciones
        logging.info("Generando recomendaciones...")
        predictions = {"target": {}}  # Inicializar con la estructura correcta
        total_sessions = len(test_df['session_id'].unique())
        
        for i, session_id in enumerate(test_df['session_id'].unique()):
            try:
                recommendations = get_recommendations(
                    model,
                    session_id,
                    test_df,
                    products_df,
                    users_df
                )
                # Guardar como string:lista de ints
                predictions["target"][str(session_id)] = [int(x) for x in recommendations]
                
                if (i + 1) % 100 == 0:
                    logging.info(f"Procesadas {i+1}/{total_sessions} sesiones")
                    
            except Exception as e:
                logging.error(f"Error en sesión {session_id}: {str(e)}")
                # En caso de error, usar recomendaciones aleatorias
                all_products = products_df.index.tolist()
                predictions["target"][str(session_id)] = [
                    int(x) for x in np.random.choice(all_products, size=5, replace=False)
                ]
        
        # Guardar predicciones con formato correcto
        os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)
        with open(PREDICTIONS_PATH, 'w') as f:
            json.dump(predictions, f, indent=4)  # Añadir indent=4 para formato legible
        
        logging.info(f"Predicciones guardadas en {PREDICTIONS_PATH}")
        return predictions
        
    except Exception as e:
        logging.error(f"Error en generate_predictions: {str(e)}")
        raise

## 2. Ejecución principal
if __name__ == "__main__":
    try:
        logging.info("Iniciando proceso de predicción...")
        generate_predictions()
        logging.info("Proceso completado exitosamente")
    except Exception as e:
        logging.error(f"Error en el proceso principal: {str(e)}")
        raise

