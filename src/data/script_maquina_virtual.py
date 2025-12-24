import pandas as pd
import aiohttp
import asyncio
from tqdm import tqdm
import os
import logging
from datetime import datetime
import json
import time  # Añadido el import de time

import pandas as pd
import aiohttp
import asyncio
from tqdm import tqdm
import os
import logging
from datetime import datetime
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('user_data_collection.log'),
        logging.StreamHandler()
    ]
)

async def get_user_data(session, user_id):
    """
    Obtiene datos de un usuario usando aiohttp.
    """
    url = f"https://zara-boost-hackathon.nuwe.io/user/{user_id}"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
            if response.status == 200:
                return user_id, await response.json()
    except Exception as e:
        logging.debug(f"Error para user_id {user_id}: {str(e)}")
    return user_id, None

async def process_batch(session, user_ids):
    """
    Procesa un lote de usuarios de forma asíncrona.
    """
    tasks = [get_user_data(session, user_id) for user_id in user_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_data = []
    for result in results:
        if isinstance(result, tuple) and result[1]:
            user_id, user_data = result
            for i in range(len(user_data['values']['country'])):
                processed_data.append({
                    'user_id': user_id,
                    'country': user_data['values']['country'][i],
                    'R': user_data['values']['R'][i],
                    'F': user_data['values']['F'][i],
                    'M': user_data['values']['M'][i]
                })
    return processed_data

async def collect_users_data(start_user=1, end_user=300000, batch_size=100):
    """
    Recolecta datos de usuarios usando async/await.
    """
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"users_data_{timestamp}.csv")
    temp_path = os.path.join(output_dir, f"temp_data_{timestamp}.json")
    
    all_data = []
    successful_requests = 0
    start_time = time.time()
    
    # Configurar cliente HTTP con límites aumentados
    conn = aiohttp.TCPConnector(limit=100, force_close=False)
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        for batch_start in tqdm(range(start_user, end_user + 1, batch_size)):
            batch_end = min(batch_start + batch_size, end_user + 1)
            batch_users = range(batch_start, batch_end)
            
            batch_results = await process_batch(session, batch_users)
            
            if batch_results:
                all_data.extend(batch_results)
                successful_requests += len(batch_results)
                
                # Guardar progreso temporal en JSON
                if len(all_data) % (batch_size * 5) == 0:
                    with open(temp_path, 'w') as f:
                        json.dump(all_data, f)
                    
                    elapsed_time = time.time() - start_time
                    speed = successful_requests / elapsed_time
                    logging.info(f"Progreso: {successful_requests} registros. "
                               f"Velocidad: {speed:.2f} registros/segundo. "
                               f"Último ID: {batch_end}")
            
            await asyncio.sleep(0.1)  # Pequeña pausa entre lotes
    
    # Guardar resultados finales
    if all_data:
        df_final = pd.DataFrame(all_data)
        df_final.to_csv(output_path, index=False)
        logging.info(f"Datos guardados en {output_path}")
        logging.info(f"Total registros: {len(df_final)}")
        
        # Eliminar archivo temporal
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return pd.DataFrame(all_data)

if __name__ == "__main__":
    logging.info("Iniciando recolección de datos de usuarios...")
    
    # Instalar aiohttp si no está instalado
    try:
        import aiohttp
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "aiohttp"])
        import aiohttp
    
    # Ejecutar la recolección de datos
    df_users = asyncio.run(collect_users_data())
    
    logging.info(f"Proceso completado. Total de registros: {len(df_users)}")
    if not df_users.empty:
        logging.info("\nPrimeras filas del dataset:")
        logging.info(df_users.head().to_string())