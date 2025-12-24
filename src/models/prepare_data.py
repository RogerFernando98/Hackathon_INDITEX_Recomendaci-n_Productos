import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
from datetime import datetime
import logging
from scipy import sparse

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def prepare_data(base_path="data"):
    """
    Prepara los datos para el sistema de recomendación.
    """
    # Definir rutas
    raw_path = os.path.join(base_path, "raw")
    processed_path = os.path.join(base_path, "processed")
    os.makedirs(processed_path, exist_ok=True)
    
    logging.info("Cargando datasets...")
    
    try:
        # Cargar datos
        train = pd.read_csv(os.path.join(raw_path, "train.csv"))
        test = pd.read_csv(os.path.join(raw_path, "test.csv"))
        products = pd.read_pickle(os.path.join(raw_path, "products.pkl"))
        users = pd.read_csv(os.path.join(raw_path, "users.csv"))

        # Preparar productos
        logging.info("Preparando datos de productos...")
        products_prepared = prepare_products(products)
        
        # Preparar usuarios
        logging.info("Preparando datos de usuarios...")
        users_prepared = prepare_users(users)
        
        # Preparar interacciones
        logging.info("Preparando datos de interacciones...")
        train_prepared = prepare_interactions(train, is_train=True)
        test_prepared = prepare_interactions(test, is_train=False)
        
        # Crear features para el modelo
        logging.info("Creando features adicionales...")
        
        # Features de popularidad
        popularity_features = create_popularity_features(train_prepared)
        
        # Features de sesión
        session_features = create_session_features(train_prepared)
        
        # Features de similitud de productos
        product_similarity = create_product_similarity_features(products_prepared)
        
        # Guardar datos procesados
        logging.info("Guardando datos procesados...")
        save_processed_data(
            processed_path,
            products_prepared,
            users_prepared,
            train_prepared,
            test_prepared,
            popularity_features,
            session_features,
            product_similarity
        )
        
        logging.info("Procesamiento de datos completado exitosamente")
        
    except Exception as e:
        logging.error(f"Error en el procesamiento de datos: {str(e)}")
        raise

def prepare_products(products):
    """Prepara los datos de productos."""
    products = products.copy()
    
    # Convertir discount a int
    products['discount'] = products['discount'].astype(int)
    
    # Procesar embeddings
    if 'embedding' in products.columns:
        # Normalizar embeddings
        embeddings = np.vstack(products['embedding'].fillna(method='ffill').fillna(method='bfill'))
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Crear clusters de productos
        kmeans = KMeans(n_clusters=20, random_state=42)
        products['product_cluster'] = kmeans.fit_predict(embeddings_scaled)
    
    # Codificar variables categóricas
    categorical_cols = ['cod_section', 'family']
    for col in categorical_cols:
        if col in products.columns:
            products[f'{col}_encoded'] = products[col].astype('category').cat.codes
    
    return products

def prepare_users(users):
    """Prepara los datos de usuarios."""
    users = users.copy()
    
    # Calcular RFM score normalizado
    users['rfm_score'] = StandardScaler().fit_transform(
        (users['F'] * users['M'] - users['R']).values.reshape(-1, 1)
    )
    
    # Codificar país
    users['country_encoded'] = users['country'].astype('category').cat.codes
    
    return users

def prepare_interactions(df, is_train=True):
    """Prepara los datos de interacciones."""
    df = df.copy()
    
    # Convertir fechas
    date_cols = ['date', 'timestamp_local']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
    
    # Extraer features temporales
    df['hour'] = df['timestamp_local'].dt.hour
    df['day_of_week'] = df['timestamp_local'].dt.dayofweek
    
    # Manejar valores nulos
    df['user_id'] = df['user_id'].fillna(-1).astype(int)
    
    if is_train:
        df['add_to_cart'] = df['add_to_cart'].fillna(0).astype(int)
    
    return df

def create_popularity_features(train_df):
    """Crea features basadas en popularidad."""
    return {
        'product_popularity': train_df.groupby('partnumber')['add_to_cart'].mean().to_dict(),
        'product_interactions': train_df.groupby('partnumber').size().to_dict()
    }

def create_session_features(train_df):
    """Crea features basadas en sesiones."""
    return {
        'session_length': train_df.groupby('session_id').size().to_dict(),
        'session_cart_ratio': train_df.groupby('session_id')['add_to_cart'].mean().to_dict()
    }

def create_product_similarity_features(products_df):
    """Crea matriz de similitud entre productos."""
    if 'embedding' not in products_df.columns:
        return None
    
    # Filtrar y validar embeddings
    valid_embeddings = []
    valid_indices = []
    
    logging.info("Procesando embeddings de productos...")
    for idx, embedding in enumerate(products_df['embedding'].values):
        try:
            # Verificar que el embedding sea válido y tenga la dimensión correcta
            if isinstance(embedding, (np.ndarray, list)) and len(embedding) == 1280:
                valid_embeddings.append(embedding)
                valid_indices.append(idx)
            else:
                logging.warning(f"Embedding inválido encontrado en el índice {idx}. Dimensión: {len(embedding) if isinstance(embedding, (np.ndarray, list)) else 'no es array'}")
        except Exception as e:
            logging.warning(f"Error procesando embedding en índice {idx}: {str(e)}")
    
    if not valid_embeddings:
        logging.error("No se encontraron embeddings válidos")
        return None
    
    try:
        # Convertir a array de numpy y crear matriz de similitud
        embeddings = np.vstack(valid_embeddings)
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        # Crear diccionario con mappings
        return {
            'embeddings': embeddings,
            'similarity_matrix': similarity_matrix,
            'valid_indices': valid_indices,
            'product_ids': products_df.index[valid_indices].tolist()
        }
    except Exception as e:
        logging.error(f"Error creando matriz de similitud: {str(e)}")
        return None

def save_processed_data(path, products_prepared, users_prepared, train_prepared, test_prepared, 
                       popularity_features, session_features, product_similarity):
    """Guarda los datos procesados."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar DataFrames en formato parquet
    dataframes = {
        'products_prepared': products_prepared,
        'users_prepared': users_prepared,
        'train_prepared': train_prepared,
        'test_prepared': test_prepared
    }
    
    # Guardar features en formato npy
    features = {
        'popularity_features': popularity_features,
        'session_features': session_features
    }
    
    # Guardar DataFrames
    for name, df in dataframes.items():
        if df is not None and isinstance(df, pd.DataFrame):
            output_path = os.path.join(path, f"{name}_{timestamp}.parquet")
            try:
                df.to_parquet(output_path, engine='pyarrow')
                logging.info(f"Guardado DataFrame {name} en {output_path}")
            except Exception as e:
                logging.error(f"Error guardando DataFrame {name}: {str(e)}")
                # Intentar guardar en CSV si falla parquet
                try:
                    csv_path = os.path.join(path, f"{name}_{timestamp}.csv")
                    df.to_csv(csv_path, index=False)
                    logging.info(f"Guardado DataFrame {name} en {csv_path} (formato CSV)")
                except Exception as e2:
                    logging.error(f"Error guardando DataFrame {name} en CSV: {str(e2)}")
    
    # Guardar features
    for name, feature in features.items():
        if feature is not None:
            output_path = os.path.join(path, f"{name}_{timestamp}.npy")
            try:
                np.save(output_path, feature)
                logging.info(f"Guardado feature {name} en {output_path}")
            except Exception as e:
                logging.error(f"Error guardando feature {name}: {str(e)}")
    
    # Guardar product_similarity por separado debido a su tamaño
    if product_similarity is not None:
        try:
            # Guardar cada componente por separado
            embeddings_path = os.path.join(path, f"product_embeddings_{timestamp}.npy")
            similarity_path = os.path.join(path, f"product_similarity_{timestamp}.npz")
            
            # Guardar embeddings
            np.save(embeddings_path, product_similarity['embeddings'])
            
            # Guardar matriz de similitud en formato sparse
            similarity_matrix = sparse.csr_matrix(product_similarity['similarity_matrix'])
            sparse.save_npz(similarity_path, similarity_matrix)
            
            # Guardar índices
            indices_path = os.path.join(path, f"product_indices_{timestamp}.npy")
            np.save(indices_path, np.array(product_similarity['valid_indices']))
            
            logging.info(f"Guardados componentes de product_similarity en {path}")
        except Exception as e:
            logging.error(f"Error guardando product_similarity: {str(e)}")

if __name__ == "__main__":
    prepare_data()