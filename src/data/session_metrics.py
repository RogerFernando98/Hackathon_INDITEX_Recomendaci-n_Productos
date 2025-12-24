import pandas as pd
import os


def get_session_metrics(df: pd.DataFrame, user_id: int) -> pd.DataFrame:
    """
    Given a pandas DataFrame in the format of the train dataset and a user_id, return the following metrics for every session_id of the user:
        - user_id (int) : the given user id.
        - session_id (int) : the session id.
        - total_session_time (float) : The time passed between the first and last interactions, in seconds. Rounded to the 2nd decimal.
        - cart_addition_ratio (float) : Percentage of the added products out of the total products interacted with. Rounded to the 2nd decimal.

    Parameters
    ----------
    df : pandas DataFrame
       DataFrame of the data to be used for the agent.
    user_id : int
        Id of the client.

    Returns
    -------
    Pandas DataFrame with metrics for all sessions of the given user, sorted by user_id and session_id.
    """
    # Verificar que el DataFrame no esté vacío y tenga las columnas necesarias
    required_columns = ['user_id', 'session_id', 'timestamp_local', 'add_to_cart']
    if df.empty or not all(col in df.columns for col in required_columns):
        return pd.DataFrame(columns=['user_id', 'session_id', 'total_session_time', 'cart_addition_ratio'])

    # Filtrar el DataFrame para el user_id proporcionado
    user_data = df[df['user_id'] == user_id].copy()

    # Si no hay datos para el usuario, retornar un DataFrame vacío con las columnas requeridas
    if user_data.empty:
        return pd.DataFrame(columns=['user_id', 'session_id', 'total_session_time', 'cart_addition_ratio'])

    # Asegurar que timestamp_local es datetime
    if not pd.api.types.is_datetime64_any_dtype(user_data['timestamp_local']):
        user_data['timestamp_local'] = pd.to_datetime(user_data['timestamp_local'])

    # Calcular métricas por sesión
    session_metrics = []
    for session_id, session_data in user_data.groupby('session_id'):
        # Calcular tiempo total de sesión
        if len(session_data) > 1:
            total_time = (session_data['timestamp_local'].max() - 
                         session_data['timestamp_local'].min()).total_seconds()
        else:
            total_time = 0.0
            
        # Calcular ratio de adición al carrito
        total_interactions = len(session_data)
        cart_additions = session_data['add_to_cart'].sum()
        cart_ratio = (cart_additions / total_interactions * 100) if total_interactions > 0 else 0.0
        
        session_metrics.append({
            'user_id': user_id,
            'session_id': session_id,
            'total_session_time': round(total_time, 2),
            'cart_addition_ratio': round(cart_ratio, 2)
        })

    # Crear DataFrame con los resultados
    result = pd.DataFrame(session_metrics)
    
    # Si no hay métricas, retornar DataFrame vacío con las columnas correctas
    if result.empty:
        return pd.DataFrame(columns=['user_id', 'session_id', 'total_session_time', 'cart_addition_ratio'])
    
    # Ordenar por user_id y session_id
    result = result.sort_values(by=['user_id', 'session_id'])
    
    # Asegurar el orden correcto de las columnas
    result = result[['user_id', 'session_id', 'total_session_time', 'cart_addition_ratio']]
    
    return result

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Cargar datos de ejemplo usando ruta relativa
    # Obtener la ruta base del proyecto
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_path = os.path.join(project_root, "data", "raw", "train.csv")
    
    # Cargar datos de ejemplo
    sample_data = pd.read_csv(data_path, parse_dates=['timestamp_local'])

    # Llamar a la función con un user_id
    user_id = 179371
    result = get_session_metrics(sample_data, user_id)

    # Mostrar el resultado
    print(result)