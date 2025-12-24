# train_model.py
import os
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
import pickle
import logging
from datetime import datetime

# Rutas de los archivos
base_path = "/Users/fernandoarroyo/Desktop/Inditex/hackathon-inditex-data-recommender/data/processed/"
products_path = base_path + "products_prepared.csv"
train_path = base_path + "train_prepared.csv"
test_path = base_path + "test_prepared.csv"
users_path = base_path + "users_prepared.csv"

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class HybridRecommender:
    def __init__(self):
        self.item_similarity_model = NearestNeighbors(n_neighbors=50, metric='cosine')
        self.ranking_model = xgb.XGBRanker(
            objective='rank:ndcg',
            learning_rate=0.1,
            max_depth=6,
            n_estimators=100
        )
        self.product_embeddings = None
        self.product_popularity = None
        self.scaler = StandardScaler()
    
    def create_features(self, user_id, session_id, product_id, interactions_df, products_df, users_df):
        """Crea features para el modelo de ranking."""
        features = {}
        
        # Features de producto
        if 'product_cluster' in products_df.columns:
            features['product_cluster'] = products_df.loc[product_id, 'product_cluster']
        else:
            features['product_cluster'] = -1
        
        features['discount'] = products_df.loc[product_id, 'discount']
        
        # Features de popularidad
        features['popularity'] = self.product_popularity.get(product_id, 0)
        
        # Features de usuario (si existe)
        if user_id in users_df.index:
            features['user_rfm'] = users_df.loc[user_id, 'rfm_score']
            features['user_country'] = users_df.loc[user_id, 'country_encoded']
        else:
            features['user_rfm'] = 0
            features['user_country'] = -1
        
        # Features de sesión
        session_interactions = interactions_df[interactions_df['session_id'] == session_id]
        features['session_length'] = len(session_interactions)
        
        # Features de interacción
        if 'pagetype' in interactions_df.columns:
            mode_values = session_interactions['pagetype'].mode()
            features['pagetype'] = mode_values.iloc[0] if not mode_values.empty else -1
        else:
            features['pagetype'] = -1
        
        if 'device_type' in interactions_df.columns:
            mode_values = session_interactions['device_type'].mode()
            features['device_type'] = mode_values.iloc[0] if not mode_values.empty else -1
        else:
            features['device_type'] = -1
        
        # Features temporales
        if 'hour' in interactions_df.columns:
            mode_values = session_interactions['hour'].mode()
            features['hour'] = mode_values.iloc[0] if not mode_values.empty else -1
        else:
            features['hour'] = -1
        
        if 'day_of_week' in interactions_df.columns:
            mode_values = session_interactions['day_of_week'].mode()
            features['day_of_week'] = mode_values.iloc[0] if not mode_values.empty else -1
        else:
            features['day_of_week'] = -1
        
        # Features de familia/categoría si existen
        for col in ['family', 'category', 'subcategory']:
            if col in products_df.columns:
                features[f'product_{col}'] = products_df.loc[product_id, col]
            else:
                features[f'product_{col}'] = -1
        
        return pd.Series(features)

    def fit(self, train_df, products_df, users_df):
        """Entrena el modelo híbrido."""
        logging.info("Iniciando entrenamiento del modelo híbrido...")
        
        # Reducir uso de memoria convirtiendo tipos de datos
        train_df = train_df.astype({
            'session_id': 'category',
            'user_id': 'int32',
            'partnumber': 'int32',
            'add_to_cart': 'int8'
        })
        
        # Procesar solo una muestra para el entrenamiento inicial
        unique_sessions = train_df['session_id'].unique()
        sample_size = min(50000, len(unique_sessions))  # Limitar a 50k sesiones
        sampled_sessions = np.random.choice(unique_sessions, sample_size, replace=False)
        train_df_sample = train_df[train_df['session_id'].isin(sampled_sessions)]
        
        logging.info("Procesando embeddings de productos...")
        valid_embeddings = []
        valid_indices = []
        
        # Procesar solo los productos que aparecen en el conjunto de entrenamiento
        relevant_products = set(train_df_sample['partnumber'].unique())
        
        for idx, (product_id, embedding) in enumerate(zip(products_df.index, products_df['embedding'].values)):
            if product_id in relevant_products:
                try:
                    if isinstance(embedding, (np.ndarray, list)) and len(embedding) == 1280:
                        valid_embeddings.append(embedding)
                        valid_indices.append(idx)
                except Exception as e:
                    continue
        
        if not valid_embeddings:
            logging.error("No se encontraron embeddings válidos")
            return self
        
        # Convertir a array de numpy y normalizar
        self.product_embeddings = self.scaler.fit_transform(np.vstack(valid_embeddings))
        self.item_similarity_model.fit(self.product_embeddings)
        self.valid_product_indices = products_df.index[valid_indices].tolist()
        
        # Calcular popularidad solo para productos relevantes
        self.product_popularity = (
            train_df_sample.groupby('partnumber')['add_to_cart']
            .mean()
            .to_dict()
        )
        
        # Preparar datos para el modelo de ranking
        logging.info("Preparando datos para el modelo de ranking...")
        train_features = []
        train_labels = []
        group_sizes = []
        
        # Procesar por lotes para reducir uso de memoria
        batch_size = 1000
        for i in range(0, len(sampled_sessions), batch_size):
            batch_sessions = sampled_sessions[i:i+batch_size]
            batch_data = train_df_sample[train_df_sample['session_id'].isin(batch_sessions)]
            
            for session_id in batch_sessions:
                session_data = batch_data[batch_data['session_id'] == session_id]
                user_id = session_data['user_id'].iloc[0]
                
                session_features = []
                session_labels = []
                
                for _, row in session_data.iterrows():
                    if row['partnumber'] in self.valid_product_indices:
                        features = self.create_features(
                            user_id, 
                            session_id, 
                            row['partnumber'],
                            batch_data,  # Usar solo datos del lote
                            products_df,
                            users_df
                        )
                        session_features.append(features)
                        session_labels.append(row['add_to_cart'])
                
                if session_features:
                    train_features.extend(session_features)
                    train_labels.extend(session_labels)
                    group_sizes.append(len(session_features))
            
            logging.info(f"Procesados {i+len(batch_sessions)}/{len(sampled_sessions)} sesiones")
        
        if not train_features:
            logging.error("No se pudieron crear features de entrenamiento")
            return self
        
        # Convertir a numpy arrays
        X_train = pd.DataFrame(train_features)
        y_train = np.array(train_labels)
        
        # Entrenar modelo de ranking con parámetros optimizados
        logging.info("Entrenando modelo de ranking...")
        self.ranking_model.fit(
            X_train, 
            y_train,
            group=group_sizes,
            verbose=True
        )
        
        logging.info("Entrenamiento completado")
        return self

    def save(self, model_path):
        """Guarda el modelo entrenado."""
        os.makedirs(model_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with open(os.path.join(model_path, f'hybrid_recommender_{timestamp}.pkl'), 'wb') as f:
            pickle.dump(self, f)
        
        logging.info(f"Modelo guardado en {model_path}")

def train_model(data_path="data", model_path="models"):
    """Función principal para entrenar el modelo."""
    # Cargar datos procesados
    logging.info("Cargando datos procesados...")
    processed_path = os.path.join(data_path, "processed")
    
    # Encontrar los archivos más recientes
    files = os.listdir(processed_path)
    train_file = sorted([f for f in files if f.startswith('train_prepared')])[-1]
    products_file = sorted([f for f in files if f.startswith('products_prepared')])[-1]
    users_file = sorted([f for f in files if f.startswith('users_prepared')])[-1]
    
    # Cargar datos
    train_df = pd.read_parquet(os.path.join(processed_path, train_file))
    products_df = pd.read_parquet(os.path.join(processed_path, products_file))
    users_df = pd.read_parquet(os.path.join(processed_path, users_file))
    
    # Entrenar modelo
    model = HybridRecommender()
    model.fit(train_df, products_df, users_df)
    
    # Guardar modelo
    model.save(model_path)
    
    return model

if __name__ == "__main__":
    train_model()
