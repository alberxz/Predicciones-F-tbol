import requests
import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor
import time

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
            
        # DEFINICIÓN DE LOS 7 TARGETS (Lo que queremos predecir) normalizados por partido
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
                # Normalizamos métricas absolutas (excepto porcentajes y ratios)
                if 'percentage' not in clave and 'accuracy' not in clave and 'ppda' not in clave:
                    fila[clave] = valor / pj
                else:
                    fila[clave] = valor
                    
        datos.append(fila)
        
    df = pd.DataFrame(datos).replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

# =====================================================================
# MÓDULO 3: EL CEREBRO NO LINEAL (MÚLTIPLES BOSQUES ESTOCÁSTICOS)
# =====================================================================
def calcular_lambdas_por_bosque(df, local, visitante):
    print("[3/4] Entrenando 7 Bosques Random Forest simultáneos (uno por métrica)...")
    
    targets = [c for c in df.columns if c.startswith('target_')]
    features = df.drop(columns=['Equipo'] + targets)
    
    # Motor de búsqueda flexible (por si escribimos "Barsa" en vez de "FC Barcelona")
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
    
    lambdas = {nom_L: {}, nom_V: {}}
    
    # Entrenamos un bosque distinto para cada evento de la cascada
    for t in targets:
        metrica = t.replace('target_', '')
        
        bosque = RandomForestRegressor(n_estimators=250, max_depth=5, random_state=42)
        bosque.fit(features, df[t])
        
        # El bosque lee las 100 variables del Levante y predice su tasa bruta
        lambda_L = float(bosque.predict(stats_L)[0])
        lambda_V = float(bosque.predict(stats_V)[0])
        
        # Efecto Espacial (Ventaja Local / Desventaja Visitante)
        if metrica in ['goals', 'shots', 'on_target', 'corners']:
            lambdas[nom_L][metrica] = lambda_L * 1.10  # +10% producción local
            lambdas[nom_V][metrica] = lambda_V * 0.90  # -10% producción visitante
        elif metrica == 'cards':
            lambdas[nom_L][metrica] = lambda_L * 0.85  # -15% castigo local
            lambdas[nom_V][metrica] = lambda_V * 1.15  # +15% castigo visitante
        else: # Palos y penaltis (Más dependientes del azar puro, no aplicamos sesgo espacial fuerte)
            lambdas[nom_L][metrica] = lambda_L
            lambdas[nom_V][metrica] = lambda_V
            
    return lambdas, nom_L, nom_V

# =====================================================================
# MÓDULO 4: MOTOR MONTE CARLO Y FORZADO FÍSICO
# =====================================================================
def ejecutar_transporte_montecarlo(lambdas, local, visitante, historias=100000):
    print(f"[4/4] Transportando {historias:,} historias con forzado termodinámico...\n")
    
    tally_L = {k: np.random.poisson(lam, historias) for k, lam in lambdas[local].items()}
    tally_V = {k: np.random.poisson(lam, historias) for k, lam in lambdas[visitante].items()}
    
    # --- LA LEY DE CONSERVACIÓN FÍSICA DEL FÚTBOL ---
    for eq in [tally_L, tally_V]:
        # 1. Los tiros a puerta no pueden ser mayores que los tiros totales
        eq['on_target'] = np.minimum(eq['on_target'], eq['shots'])
        # 2. Los palos no pueden ser mayores que los tiros que NO van a puerta
        tiros_fuera = eq['shots'] - eq['on_target']
        eq['woodwork'] = np.minimum(eq['woodwork'], tiros_fuera)
        # 3. Los goles no pueden ser mayores que los tiros a puerta
        eq['goals'] = np.minimum(eq['goals'], eq['on_target'])

    # --- RENDERIZADO DEL DASHBOARD FINAL ---
    print("=" * 75)
    print(f" ⚽ GEMELO DIGITAL: {local} (L) vs {visitante} (V)")
    print("=" * 75)
    print(f"{'Variable de Cascada (Media por partido)':<42} | {local:<12} | {visitante:<12}")
    print("-" * 75)
    
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
        
    print("-" * 75)
    print(" 📊 PROBABILIDADES CUÁNTICAS DE SUCESOS (Mercado 1X2)")
    print("-" * 75)
    
    p_1 = np.mean(tally_L['goals'] > tally_V['goals']) * 100
    p_x = np.mean(tally_L['goals'] == tally_V['goals']) * 100
    p_2 = np.mean(tally_L['goals'] < tally_V['goals']) * 100
    p_over = np.mean((tally_L['goals'] + tally_V['goals']) >= 3) * 100
    p_ambos = np.mean((tally_L['goals'] > 0) & (tally_V['goals'] > 0)) * 100
    
    print(f" [1] Victoria Local ({local}):  {p_1:>6.2f} %")
    print(f" [X] Empate:                 {p_x:>6.2f} %")
    print(f" [2] Victoria Visitante ({visitante}):{p_2:>6.2f} %")
    print(f" [*] Más de 2.5 Goles en total:      {p_over:>6.2f} %")
    print(f" [*] Ambos equipos marcan:    {p_ambos:>6.2f} %")
    print("=" * 75)

# =====================================================================
# EJECUCIÓN (SISTEMA DE ARRANQUE)
# =====================================================================
if __name__ == "__main__":
    try:
        # 1. Cargamos el combustible (Datos y Matrices)
        crudos = descargar_matriz_laliga()
        matriz = estructurar_fisica_de_particulas(crudos)
        
        # 2. Definimos los materiales objetivo
        # Nota: Asegúrate de que los nombres coincidan aproximadamente con los de LaLiga
        EQUIPO_LOCAL = "CD Leganés" 
        EQUIPO_VISITANTE = "SD Huesca"
        
        # 3. Lanzamos la simulación
        lambdas_partido, nom_L, nom_V = calcular_lambdas_por_bosque(matriz, EQUIPO_LOCAL, EQUIPO_VISITANTE)
        ejecutar_transporte_montecarlo(lambdas_partido, nom_L, nom_V, historias=100000)
        
    except Exception as e:
        print(f"\n[!] ERROR: {e}")