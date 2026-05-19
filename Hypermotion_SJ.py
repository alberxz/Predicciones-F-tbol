import requests
import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV # El juez que evalúa el sobreajuste
import time
import warnings

# Silenciamos las advertencias de pandas/sklearn para mantener la consola limpia
warnings.filterwarnings('ignore')

# =====================================================================
# MÓDULO 1: INGESTA DE DATOS (EL SENSOR GLOBAL)
# =====================================================================
def descargar_matriz_laliga(max_intentos=3):
    print("\n[1/4] Inicializando secuencias. Descargando telemetría de LaLiga Hypermotion...")
    url = "https://www.laliga.com/estadisticas/laliga-hypermotion/equipo"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept-Language': 'es-ES,es;q=0.9'
    }
    
    for intento in range(max_intentos):
        try:
            respuesta = requests.get(url, headers=headers, timeout=15)
            respuesta.raise_for_status()
            soup = BeautifulSoup(respuesta.text, 'html.parser')
            script = soup.find('script', id='__NEXT_DATA__')
            
            if script:
                datos = json.loads(script.string)
                equipos = datos.get('props', {}).get('pageProps', {}).get('statsData', {}).get('team_stats', [])
                if equipos:
                    print(f"      > [OK] Datos capturados: {len(equipos)} equipos.")
                    return equipos
        except Exception as e:
            print(f"      > [!] Intento {intento+1} fallido. Reintentando...")
            time.sleep(2)
            
    raise Exception("Fallo crítico en la extracción de datos.")

# =====================================================================
# MÓDULO 2: PREPARACIÓN DEL INPUT DECK (FILTROS Y TARGETS)
# =====================================================================
def estructurar_fisica_de_particulas(equipos_crudos):
    print("[2/4] Aislado de variables y definición de los 7 targets de colisión...")
    datos = []
    
    for equipo in equipos_crudos:
        nombre = equipo['nick_name']
        stats = {item['name']: item['stat'] for item in equipo['stats']}
        
        pj = stats.get('games_played', 1)
        if pj == 0: continue
            
        fila = {
            'Equipo': nombre,
            'target_goals': stats.get('goals', 0) / pj,
            'target_shots': stats.get('total_shots', 0) / pj,
            'target_on_target': stats.get('shots_on_target_inc_goals', 0) / pj,
            'target_corners': stats.get('corners_won', 0) / pj,
            'target_cards': (stats.get('yellow_cards', 0) + stats.get('total_red_cards', 0)) / pj,
            'target_woodwork': stats.get('hit_woodwork', 0) / pj,
            'target_penalties': stats.get('penalties_taken', 0) / pj
        }
        
        columnas_trampa = ['points', 'position', 'won', 'drawn', 'lost', 'games_played', 'games_pending']
        
        for clave, valor in stats.items():
            if clave not in columnas_trampa and 'goal' not in clave and isinstance(valor, (int, float)):
                if 'percentage' not in clave and 'accuracy' not in clave and 'ppda' not in clave:
                    fila[clave] = valor / pj
                else:
                    fila[clave] = valor
                    
        datos.append(fila)
        
    df = pd.DataFrame(datos).replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

# =====================================================================
# MÓDULO 3: XGBOOST AUTÓNOMO CON GRID SEARCH Y SESGO BAYESIANO
# =====================================================================
def calcular_lambdas_xgboost_autotuning(df, equipos_crudos, local, visitante):
    print("[3/4] Entrenando IA... Ejecutando GridSearch (Validación Cruzada Anti-Sobreajuste)...")
    
    targets = [c for c in df.columns if c.startswith('target_')]
    features = df.drop(columns=['Equipo'] + targets)
    
    def encontrar_equipo(buscado, lista):
        for n in lista:
            if buscado.lower() in n.lower(): return n
        return None

    nom_L = encontrar_equipo(local, df['Equipo'].tolist())
    nom_V = encontrar_equipo(visitante, df['Equipo'].tolist())
    
    if not nom_L or not nom_V:
        raise Exception(f"Equipo no encontrado. Disponibles: {df['Equipo'].tolist()}")

    stats_L = df[df['Equipo'] == nom_L][features.columns]
    stats_V = df[df['Equipo'] == nom_V][features.columns]
    
    # --- CALCULADORA BAYESIANA DINÁMICA ---
    def calcular_sesgo_bayesiano(nombre_equipo, condicion='home'):
        equipo_data = next((e for e in equipos_crudos if nombre_equipo.lower() in e['nick_name'].lower()), None)
        prior = 1.10 if condicion == 'home' else 0.90
        if not equipo_data: return prior
        
        dic = {item['name']: item['stat'] for item in equipo_data['stats']}
        victorias_cond = sum(v for k, v in dic.items() if condicion in k.lower() and 'won' in k.lower())
        goles_cond = sum(v for k, v in dic.items() if condicion in k.lower() and 'goal' in k.lower() and 'conceded' not in k.lower())
        
        victorias_totales = max(dic.get('won', 0.1), 0.1)
        goles_totales = max(dic.get('goals', 0.1), 0.1)
        pj_totales = max(dic.get('games_played', 1), 1)
        
        N_condicion = pj_totales / 2.0
        rendimiento_vic = victorias_cond / (victorias_totales / 2.0)
        rendimiento_gol = goles_cond / (goles_totales / 2.0)
        rendimiento_puro = (rendimiento_vic + rendimiento_gol) / 2.0 
        
        K = 10.0
        peso_realidad = N_condicion / (N_condicion + K)
        peso_historico = K / (N_condicion + K)
        
        multiplicador_final = (peso_realidad * rendimiento_puro) + (peso_historico * prior)
        return max(0.5, multiplicador_final)

    sesgo_L = calcular_sesgo_bayesiano(nom_L, 'home')
    sesgo_V = calcular_sesgo_bayesiano(nom_V, 'away')
    
    print(f"      > [MOTOR BAYESIANO] {nom_L} (L): Multiplicador Espacial = {sesgo_L:.3f}x")
    print(f"      > [MOTOR BAYESIANO] {nom_V} (V): Multiplicador Espacial = {sesgo_V:.3f}x")
    print("      > [GRID SEARCH] Iniciando búsqueda de hiperparámetros. Esto puede tardar unos segundos...")

    lambdas = {nom_L: {}, nom_V: {}}
    
    # LA CUADRÍCULA DE PARÁMETROS: El código probará estas 12 combinaciones por cada métrica
    param_grid = {
        'max_depth': [2, 3],              # Árboles enanos para evitar memorización
        'learning_rate': [0.05, 0.1],     # Velocidad de corrección de gradiente
        'n_estimators': [100, 150, 200]   # Iteraciones máximas
    }
    
    for t in targets:
        metrica = t.replace('target_', '')
        
        # Configuramos el modelo base con 'subsample' al 80% (venda en los ojos anti-sobreajuste)
        xgb_base = XGBRegressor(random_state=42, objective='reg:squarederror', subsample=0.8, colsample_bytree=0.8)
        
        # EL JUEZ (GridSearchCV): cv=3 significa que divide la liga en 3 partes, entrena con 2 y se examina con 1.
        grid_search = GridSearchCV(
            estimator=xgb_base, 
            param_grid=param_grid, 
            cv=3,                           # 3-Fold Cross Validation
            scoring='neg_mean_squared_error', 
            n_jobs=-1,                      # Usa todos los núcleos de tu procesador
            verbose=0
        )
        
        grid_search.fit(features, df[t])
        mejor_modelo = grid_search.best_estimator_
        
        # Mostramos por pantalla qué parámetros ha elegido el código para sobrevivir al sobreajuste
        if metrica in ['goals', 'cards']:
            print(f"        ~ {metrica.upper()}: Configuración ganadora -> {grid_search.best_params_}")
        
        lambda_L = float(mejor_modelo.predict(stats_L)[0])
        lambda_V = float(mejor_modelo.predict(stats_V)[0])
        
        # APLICACIÓN DE LA FÍSICA ESPACIAL PURA
        if metrica in ['goals', 'shots', 'on_target', 'corners']:
            lambdas[nom_L][metrica] = lambda_L * sesgo_L
            lambdas[nom_V][metrica] = lambda_V * sesgo_V
        elif metrica == 'cards':
            lambdas[nom_L][metrica] = max(0.1, lambda_L * (2.0 - sesgo_L))
            lambdas[nom_V][metrica] = max(0.1, lambda_V * (2.0 - sesgo_V))
        else:
            lambdas[nom_L][metrica] = lambda_L
            lambdas[nom_V][metrica] = lambda_V
            
    return lambdas, nom_L, nom_V

# =====================================================================
# MÓDULO 4: MOTOR MONTE CARLO Y MERCADOS FINANCIEROS
# =====================================================================
def ejecutar_transporte_montecarlo(lambdas, local, visitante, historias=100000):
    print(f"\n[4/4] Transportando {historias:,} historias con forzado termodinámico...\n")
    
    tally_L = {k: np.random.poisson(lam, historias) for k, lam in lambdas[local].items()}
    tally_V = {k: np.random.poisson(lam, historias) for k, lam in lambdas[visitante].items()}
    
    # --- LA LEY DE CONSERVACIÓN FÍSICA ---
    for eq in [tally_L, tally_V]:
        eq['on_target'] = np.minimum(eq['on_target'], eq['shots'])
        tiros_fuera = eq['shots'] - eq['on_target']
        eq['woodwork'] = np.minimum(eq['woodwork'], tiros_fuera)
        eq['goals'] = np.minimum(eq['goals'], eq['on_target'])

    print("=" * 85)
    print(f" ⚽ GEMELO DIGITAL: {local} (L) vs {visitante} (V)")
    print("=" * 85)
    print(f"{'Variable de Cascada (Media por partido)':<42} | {local:<12} | {visitante:<12}")
    print("-" * 85)
    
    metricas_mostrar = {
        'goals': 'Goles',
        'shots': 'Tiros Totales',
        'on_target': 'Tiros a puerta',
        'corners': 'Saques de Esquina',
        'cards': 'Tarjetas Recibidas',
        'woodwork': 'Balones al Palo',
        'penalties': 'Penaltis a Favor'
    }
    
    for clave, nombre in metricas_mostrar.items():
        media_L = np.mean(tally_L[clave])
        media_V = np.mean(tally_V[clave])
        print(f"{nombre:<42} | {media_L:<12.2f} | {media_V:<12.2f}")
        
    print("-" * 85)
    print(" 📊 PROBABILIDADES CUÁNTICAS Y CUOTAS DE VALOR (Payout 91.5%)")
    print("-" * 85)
    
    def calcular_cuota(probabilidad):
        if probabilidad <= 0: return 0.00
        return 91.5 / probabilidad

    p_1 = np.mean(tally_L['goals'] > tally_V['goals']) * 100
    p_x = np.mean(tally_L['goals'] == tally_V['goals']) * 100
    p_2 = np.mean(tally_L['goals'] < tally_V['goals']) * 100
    p_over = np.mean((tally_L['goals'] + tally_V['goals']) >= 3) * 100 
    p_ambos = np.mean((tally_L['goals'] > 0) & (tally_V['goals'] > 0)) * 100
    
    print(" [MERCADO DE GOLES]")
    print(f" [1] Victoria Local:          {p_1:>6.2f} %  -> Cuota Exigible: {calcular_cuota(p_1):>5.2f}")
    print(f" [X] Empate:                  {p_x:>6.2f} %  -> Cuota Exigible: {calcular_cuota(p_x):>5.2f}")
    print(f" [2] Victoria Visitante:      {p_2:>6.2f} %  -> Cuota Exigible: {calcular_cuota(p_2):>5.2f}")
    print(f" [*] Más de 2.5 Goles:        {p_over:>6.2f} %  -> Cuota Exigible: {calcular_cuota(p_over):>5.2f}")
    print(f" [*] Ambos Marcan:            {p_ambos:>6.2f} %  -> Cuota Exigible: {calcular_cuota(p_ambos):>5.2f}")
    
    p_corn_1 = np.mean(tally_L['corners'] > tally_V['corners']) * 100
    p_corn_x = np.mean(tally_L['corners'] == tally_V['corners']) * 100
    p_corn_2 = np.mean(tally_L['corners'] < tally_V['corners']) * 100
    p_corn_over = np.mean((tally_L['corners'] + tally_V['corners']) >= 9) * 100 
    
    print("\n [MERCADO DE CÓRNERS]")
    print(f" [1] Más Córners Local:       {p_corn_1:>6.2f} %  -> Cuota Exigible: {calcular_cuota(p_corn_1):>5.2f}")
    print(f" [X] Empate a Córners:        {p_corn_x:>6.2f} %  -> Cuota Exigible: {calcular_cuota(p_corn_x):>5.2f}")
    print(f" [2] Más Córners Visitante:   {p_corn_2:>6.2f} %  -> Cuota Exigible: {calcular_cuota(p_corn_2):>5.2f}")
    print(f" [*] Más de 8.5 Córners:      {p_corn_over:>6.2f} %  -> Cuota Exigible: {calcular_cuota(p_corn_over):>5.2f}")

    p_card_1 = np.mean(tally_L['cards'] > tally_V['cards']) * 100
    p_card_x = np.mean(tally_L['cards'] == tally_V['cards']) * 100
    p_card_2 = np.mean(tally_L['cards'] < tally_V['cards']) * 100
    p_card_over = np.mean((tally_L['cards'] + tally_V['cards']) >= 6) * 100 
    
    print("\n [MERCADO DE TARJETAS]")
    print(f" [1] Más Tarjetas Local:      {p_card_1:>6.2f} %  -> Cuota Exigible: {calcular_cuota(p_card_1):>5.2f}")
    print(f" [X] Empate a Tarjetas:       {p_card_x:>6.2f} %  -> Cuota Exigible: {calcular_cuota(p_card_x):>5.2f}")
    print(f" [2] Más Tarjetas Visitante:  {p_card_2:>6.2f} %  -> Cuota Exigible: {calcular_cuota(p_card_2):>5.2f}")
    print(f" [*] Más de 5.5 Tarjetas:     {p_card_over:>6.2f} %  -> Cuota Exigible: {calcular_cuota(p_card_over):>5.2f}")

    p_sot_1 = np.mean(tally_L['on_target'] > tally_V['on_target']) * 100
    p_sot_2 = np.mean(tally_L['on_target'] < tally_V['on_target']) * 100
    p_sot_over = np.mean((tally_L['on_target'] + tally_V['on_target']) >= 8) * 100 
    
    print("\n [MERCADO DE TIROS A PUERTA]")
    print(f" [1] Más a Puerta Local:      {p_sot_1:>6.2f} %  -> Cuota Exigible: {calcular_cuota(p_sot_1):>5.2f}")
    print(f" [2] Más a Puerta Visitante:  {p_sot_2:>6.2f} %  -> Cuota Exigible: {calcular_cuota(p_sot_2):>5.2f}")
    print(f" [*] Más de 7.5 a Puerta:     {p_sot_over:>6.2f} %  -> Cuota Exigible: {calcular_cuota(p_sot_over):>5.2f}")
    print("=" * 85)

# =====================================================================
# EJECUCIÓN (SISTEMA DE ARRANQUE)
# =====================================================================
if __name__ == "__main__":
    try:
        crudos = descargar_matriz_laliga()
        matriz = estructurar_fisica_de_particulas(crudos)
        
        EQUIPO_LOCAL = "CD Leganés" 
        EQUIPO_VISITANTE = "SD Huesca"
        
        lambdas_partido, nom_L, nom_V = calcular_lambdas_xgboost_autotuning(matriz, crudos, EQUIPO_LOCAL, EQUIPO_VISITANTE)
        ejecutar_transporte_montecarlo(lambdas_partido, nom_L, nom_V, historias=100000)
        
    except Exception as e:
        print(f"\n[!] ERROR: {e}")
