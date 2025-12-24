import pandas as pd
import requests
import time
from tqdm import tqdm
import os
import json

def get_user_data(user_id, max_retries=3, timeout=10):
    """
    Obtiene los datos de un usuario específico de la API con reintentos.
    """
    for attempt in range(max_retries):
        try:
            url = f"https://zara-boost-hackathon.nuwe.io/user/{user_id}"
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            time.sleep(1)  # Esperar 1 segundo entre reintentos
        except Exception as e:
            if attempt == max_retries - 1:  # Si es el último intento
                print(f"Error final obteniendo datos para user_id {user_id}: {str(e)}")
            time.sleep(2)  # Esperar más tiempo después de un error
    return None

def save_progress(data, processed_users, output_path, progress_file):
    """
    Guarda el progreso actual y los usuarios procesados.
    """
    # Guardar datos actuales
    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
    
    # Guardar lista de usuarios procesados
    with open(progress_file, 'w') as f:
        json.dump(list(processed_users), f)

def load_progress(progress_file):
    """
    Carga los usuarios ya procesados.
    """
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return set(json.load(f))
    return set()

def collect_all_users_data(train_path, output_path, batch_size=1000):
    """
    Recolecta datos de usuarios con guardado periódico y capacidad de continuar.
    """
    # Configurar archivos de progreso
    progress_file = output_path.replace('.csv', '_progress.json')
    
    # Cargar usuarios únicos
    print("Cargando usuarios únicos del dataset de entrenamiento...")
    df_train = pd.read_csv(train_path)
    unique_users = df_train['user_id'].unique()
    
    # Cargar progreso anterior
    processed_users = load_progress(progress_file)
    print(f"Usuarios ya procesados: {len(processed_users)}")
    
    # Filtrar usuarios pendientes
    pending_users = [u for u in unique_users if u not in processed_users]
    print(f"Usuarios pendientes: {len(pending_users)}")
    
    # Inicializar listas
    all_data = []
    current_batch = []
    
    # Obtener datos para cada usuario
    print("Obteniendo datos de usuarios de la API...")
    for user_id in tqdm(pending_users):
        user_data = get_user_data(user_id)
        if user_data:
            for i in range(len(user_data['values']['country'])):
                row = {
                    'user_id': user_id,
                    'country': user_data['values']['country'][i],
                    'R': user_data['values']['R'][i],
                    'F': user_data['values']['F'][i],
                    'M': user_data['values']['M'][i]
                }
                current_batch.append(row)
        
        processed_users.add(user_id)
        
        # Guardar progreso cada batch_size usuarios
        if len(current_batch) >= batch_size:
            all_data.extend(current_batch)
            save_progress(all_data, processed_users, output_path, progress_file)
            current_batch = []
            
        time.sleep(0.2)  # Pausa más larga entre peticiones
    
    # Guardar datos finales
    if current_batch:
        all_data.extend(current_batch)
    save_progress(all_data, processed_users, output_path, progress_file)
    
    return pd.DataFrame(all_data)

if __name__ == "__main__":
    # Definir rutas
    train_path = "/Users/fernandoarroyo/Desktop/Inditex/hackathon-inditex-data-recommender/data/raw/train.csv"
    output_path = "/Users/fernandoarroyo/Desktop/Inditex/hackathon-inditex-data-recommender/data/raw/users.csv"
    
    # Obtener datos
    print("Iniciando recolección de datos de usuarios...")
    df_users = collect_all_users_data(train_path, output_path)
    
    print(f"\nProceso completado.")
    print(f"Total de registros: {len(df_users)}")
    print("\nPrimeras filas del dataset:")
    print(df_users.head())
    print("\nInformación del dataset:")
    print(df_users.info())