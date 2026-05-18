import requests
import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor
import time

# =====================================================================
# MÓDULO 1: ESCÁNER GLOBAL (MACRO Y MICRO DATOS)
# =====================================================================
def descargar_ecosistema_completo(max_intentos=3):
    print("\n[1/4] Inicializando Escáner Global de LaLiga...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    url_equipos = "https://www.laliga.com/estadisticas/laliga-easports/equipo"
    urls_jugadores = [
        "https://www.laliga.com/estadisticas/laliga-easports/goleadores",
        "https://www.laliga.com/estadisticas/laliga-easports/asistencias",
        "https://www.laliga.com/estadisticas/laliga-easports/tarjetas-amarillas",
        "https://www.laliga.com/estadisticas/laliga-easports/recuperaciones",
        "https://www.laliga.com/estadisticas/laliga-easports/despejes"
    ]
    
    def extraer_json(url):
        for _ in range(max_intentos):
            try:
                r = requests.get(url, headers=headers, timeout=15)
                soup = BeautifulSoup(r.text, 'html.parser')
                script = soup.find('script', id='__NEXT_DATA__')
                if script: return json.loads(script.string)
            except: time.sleep(2)
        return None

    def buscar_nodos_jugadores(datos, lista):
        if isinstance(datos, dict):
            if 'name' in datos and 'stats' in datos and isinstance(datos['stats'], list):
                lista.append(datos)
            for v in datos.values(): buscar_nodos_jugadores(v, lista)
        elif isinstance(datos, list):
            for i in datos: buscar_nodos_jugadores(i, lista)

    d_eq = extraer_json(url_equipos)
    equipos = d_eq.get('props', {}).get('pageProps', {}).get('statsData', {}).get('team_stats', []) if d_eq else []
    
    jugadores_db = {}
    for url in urls_jugadores:
        d_jug = extraer_json(url)
        if d_jug:
            nodos = []
            buscar_nodos_jugadores(d_jug, nodos)
            for jug in nodos:
                nombre = jug.get('name', 'Desc').lower()
                eq = jug.get('team', {}).get('nick_name', '') if isinstance(jug.get('team'), dict) else str(jug.get('team', ''))
                if nombre not in jugadores_db: jugadores_db[nombre] = {'equipo': eq, 'stats': {}}
                for item in jug.get('stats', []):
                    if 'name' in item and 'stat' in item:
                        jugadores_db[nombre]['stats'][item['name']] = item['stat']

    print(f"      > [OK] Ecosistema capturado: {len(equipos)} equipos y {len(jugadores_db)} perfiles.")
    return equipos, jugadores_db

# =====================================================================
# MÓDULO 2: ESTADO ESTACIONARIO (MACRO)
# =====================================================================
def preparar_matriz_equipos(equipos_crudos):
    datos = []
    for equipo in equipos_crudos:
        nombre = equipo['nick_name']
        stats = {item['name']: item['stat'] for item in equipo['stats']}
        pj = stats.get('games_played', 1)
        if pj == 0: continue
            
        fila = {
            'Equipo': nombre,
            'target_games_played': pj,
            'target_goals': stats.get('goals', 0) / pj,
            'target_shots': stats.get('total_shots', 0) / pj,
            'target_on_target': stats.get('shots_on_target_inc_goals', 0) / pj,
            'target_corners': stats.get('corners_won', 0) / pj,
            'target_cards': (stats.get('yellow_cards', 0) + stats.get('total_red_cards', 0)) / pj,
            'target_woodwork': stats.get('hit_woodwork', 0) / pj,
            'target_penalties': stats.get('penalties_taken', 0) / pj
        }
        
        columnas_trampa = ['points', 'position', 'won', 'drawn', 'lost', 'games_played', 'goals']
        for k, v in stats.items():
            if k not in columnas_trampa and isinstance(v, (int, float)):
                fila[k] = v / pj if 'percentage' not in k and 'accuracy' not in k else v
        datos.append(fila)
        
    return pd.DataFrame(datos).fillna(0)

# =====================================================================
# MÓDULO 3: PERFILADOR AUTOMÁTICO (WAR DINÁMICO ESTRICTO)
# =====================================================================
def analizar_perfiles_jugadores(df_equipos, jugadores_db, bajas):
    print("\n[2/4] Ejecutando Perfilador Automático (Extracción de Llaves Exactas)...")
    
    impactos = {}

    for nombre_baja, equipo_baja in bajas:
        nombre_baja = nombre_baja.lower()
        jugador = next((info for n, info in jugadores_db.items() if nombre_baja in n), None)
        nombre_of = next((n for n in jugadores_db.keys() if nombre_baja in n), nombre_baja)

        if not jugador:
            print(f"      > [!] '{nombre_baja.title()}' no encontrado. Impacto nulo.")
            continue

        st = jugador['stats']
        idx_eq_list = df_equipos.index[df_equipos['Equipo'].str.contains(equipo_baja, case=False)].tolist()
        if not idx_eq_list: continue
        idx = idx_eq_list[0]
        
        mins = st.get('total_mins_played', st.get('mins_played', 1))
        if mins == 0: mins = 1
        partidos_jug = max(1.0, mins / 90.0)
        
        print(f"      > [-] Escaneando a '{nombre_of.title()}' ({equipo_baja})...")

        # --- EXTRACCIÓN REAL DEL JUGADOR (Con llaves exactas comprobadas) ---
        pg_goles = st.get('total_goals', 0) / partidos_jug
        pg_asist = st.get('total_assists', 0) / partidos_jug
        pg_tiros = st.get('total_scoring_att', st.get('total_attempt', 0)) / partidos_jug

        pg_robos = st.get('total_interception', 0) / partidos_jug
        pg_entradas = st.get('total_tackle', 0) / partidos_jug
        pg_despejes = st.get('total_clearance', 0) / partidos_jug

        # --- EXTRACCIÓN REAL DEL EQUIPO (Para sacar proporciones) ---
        eq_goles = max(df_equipos.at[idx, 'target_goals'], 0.1)
        # Búsqueda dinámica para asistencias, tiros y defensa del equipo
        eq_asist = max(next((df_equipos.at[idx, c] for c in df_equipos.columns if 'assist' in c), 1.0), 0.1)
        eq_tiros = max(df_equipos.at[idx, 'target_shots'], 0.1)
        
        eq_robos = max(next((df_equipos.at[idx, c] for c in df_equipos.columns if 'interception' in c), 10.0), 0.1)
        eq_entradas = max(next((df_equipos.at[idx, c] for c in df_equipos.columns if 'tackle' in c), 15.0), 0.1)
        eq_despejes = max(next((df_equipos.at[idx, c] for c in df_equipos.columns if 'clearance' in c), 15.0), 0.1)

        # --- CÁLCULO DE PESOS PUROS ---
        peso_goles = min(0.60, pg_goles / eq_goles)
        peso_asist = min(0.60, pg_asist / eq_asist)
        peso_tiros = min(0.60, pg_tiros / eq_tiros)
        poder_ofensivo = (peso_goles + peso_asist + peso_tiros) / 3.0

        peso_robos = min(0.60, pg_robos / eq_robos)
        peso_entradas = min(0.60, pg_entradas / eq_entradas)
        peso_despejes = min(0.60, pg_despejes / eq_despejes)
        poder_defensivo = (peso_robos + peso_entradas + peso_despejes) / 3.0

        # --- INYECCIÓN ALGORÍTMICA (WAR) ---
        suplente_ofensivo = (1.0 - poder_ofensivo) / 10.0
        multiplicador_ataque_propio = max(0.40, 1.0 - poder_ofensivo + suplente_ofensivo)

        suplente_defensivo = (1.0 - poder_defensivo) / 10.0
        perdida_defensa_neta = poder_defensivo - suplente_defensivo
        multiplicador_ataque_rival = min(1.50, 1.0 + perdida_defensa_neta)

        print(f"          ~ Perfil Detectado: [Ofensiva: {poder_ofensivo*100:5.1f}% | Defensiva: {poder_defensivo*100:5.1f}%]")
        print(f"          ~ Consecuencia FÍSICA: Su equipo ataca al {multiplicador_ataque_propio*100:.1f}%. El rival ataca al {multiplicador_ataque_rival*100:.1f}%.")

        impactos[equipo_baja] = {
            'ataque_propio_mult': multiplicador_ataque_propio,
            'ataque_rival_mult': multiplicador_ataque_rival
        }

    return impactos

# =====================================================================
# MÓDULO 4: BOSQUES ALEATORIOS + INYECCIÓN CRUZADA
# =====================================================================
def predecir_partido_inteligente(df_entrenamiento, impactos, local, visitante, historias=100000):
    print("\n[3/4] Entrenando Bosques Base y aplicando Inyección Cruzada de Roles...")
    
    targets = [c for c in df_entrenamiento.columns if c.startswith('target_')]
    features = df_entrenamiento.drop(columns=['Equipo', 'target_games_played'] + targets, errors='ignore')
    
    nom_L = next((n for n in df_entrenamiento['Equipo'] if local.lower() in n.lower()), None)
    nom_V = next((n for n in df_entrenamiento['Equipo'] if visitante.lower() in n.lower()), None)

    stats_L = df_entrenamiento[df_entrenamiento['Equipo'] == nom_L][features.columns]
    stats_V = df_entrenamiento[df_entrenamiento['Equipo'] == nom_V][features.columns]
    
    lambdas = {nom_L: {}, nom_V: {}}
    
    # 1. ENTRENAMIENTO ESTACIONARIO
    for t in targets:
        metrica = t.replace('target_', '')
        bosque = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
        bosque.fit(features, df_entrenamiento[t]) 
        
        lam_L = float(bosque.predict(stats_L)[0])
        lam_V = float(bosque.predict(stats_V)[0])
        
        lambdas[nom_L][metrica] = lam_L * 1.10 if metrica in ['goals', 'shots', 'on_target', 'corners'] else lam_L
        lambdas[nom_V][metrica] = lam_V * 0.90 if metrica in ['goals', 'shots', 'on_target', 'corners'] else lam_V

    # 2. INYECCIÓN CRUZADA DE BAJAS
    for eq_target, efectos in impactos.items():
        es_local = (nom_L is not None) and (eq_target.lower() in nom_L.lower())
        es_visit = (nom_V is not None) and (eq_target.lower() in nom_V.lower())
        
        if es_local:
            lambdas[nom_L]['goals'] *= efectos['ataque_propio_mult']
            lambdas[nom_L]['shots'] *= efectos['ataque_propio_mult']
            lambdas[nom_L]['on_target'] *= efectos['ataque_propio_mult']
            lambdas[nom_L]['corners'] *= efectos['ataque_propio_mult']
            
            lambdas[nom_V]['goals'] *= efectos['ataque_rival_mult']
            lambdas[nom_V]['shots'] *= efectos['ataque_rival_mult']
            lambdas[nom_V]['on_target'] *= efectos['ataque_rival_mult']
            
        elif es_visit:
            lambdas[nom_V]['goals'] *= efectos['ataque_propio_mult']
            lambdas[nom_V]['shots'] *= efectos['ataque_propio_mult']
            lambdas[nom_V]['on_target'] *= efectos['ataque_propio_mult']
            lambdas[nom_V]['corners'] *= efectos['ataque_propio_mult']
            
            lambdas[nom_L]['goals'] *= efectos['ataque_rival_mult']
            lambdas[nom_L]['shots'] *= efectos['ataque_rival_mult']
            lambdas[nom_L]['on_target'] *= efectos['ataque_rival_mult']

    print(f"[4/4] Transporte Monte Carlo en curso ({historias:,} iteraciones)...\n")
    
    tally_L = {k: np.random.poisson(lam, historias) for k, lam in lambdas[nom_L].items()}
    tally_V = {k: np.random.poisson(lam, historias) for k, lam in lambdas[nom_V].items()}
    
    # FORZADO DE EVENTOS
    for eq in [tally_L, tally_V]:
        eq['on_target'] = np.minimum(eq['on_target'], eq['shots'])
        eq['woodwork'] = np.minimum(eq['woodwork'], eq['shots'] - eq['on_target'])
        eq['goals'] = np.minimum(eq['goals'], eq['on_target'])

    print("=" * 75)
    print(f" ⚽ GEMELO DIGITAL: {nom_L} (L) vs {nom_V} (V)")
    print("=" * 75)
    print(f"{'Variable de Cascada (Media por partido)':<42} | {nom_L:<12} | {nom_V:<12}")
    print("-" * 75)
    
    metricas = {
        'goals': 'Goles', 
        'shots': 'Tiros Totales', 
        'on_target': 'Tiros a puerta', 
        'corners': 'Saques de Esquina',
        'cards': 'Tarjetas Recibidas',
        'woodwork': 'Balones al Palos',
        'penalties': 'Penaltis a Favor'
    }
    for clave, nombre in metricas.items():
        print(f"{nombre:<42} | {np.mean(tally_L[clave]):<12.2f} | {np.mean(tally_V[clave]):<12.2f}")
        
    print("-" * 75)
    print(" 📊 PROBABILIDADES CUÁNTICAS DE SUCESOS (Mercado 1X2)")
    print("-" * 75)
    print(f" [1] Victoria Local:  {np.mean(tally_L['goals'] > tally_V['goals']) * 100:>6.2f} %")
    print(f" [X] Empate:  {np.mean(tally_L['goals'] == tally_V['goals']) * 100:>6.2f} %")
    print(f" [2] Victoria Visitante: {np.mean(tally_L['goals'] < tally_V['goals']) * 100:>6.2f} %")
    print(f" [*] Más de 2.5 Goles:{np.mean((tally_L['goals'] + tally_V['goals']) >= 3) * 100:>6.2f} %")
    print(f" [*] Ambos Marcan:    {np.mean((tally_L['goals'] > 0) & (tally_V['goals'] > 0)) * 100:>6.2f} %")
    print("=" * 75)

# =====================================================================
# EJECUCIÓN MAESTRA
# =====================================================================
if __name__ == "__main__":
    try:
        crudos_eq, db_jug = descargar_ecosistema_completo()
        df_entrenamiento = preparar_matriz_equipos(crudos_eq)
        
        EQUIPO_LOCAL = "Levante"
        EQUIPO_VISITANTE = "Mallorca"
        
        bajas = [
            ("ninguno", EQUIPO_VISITANTE),
            ("ninguno", EQUIPO_VISITANTE)
        ]
        
        impactos = analizar_perfiles_jugadores(df_entrenamiento, db_jug, bajas)
        predecir_partido_inteligente(df_entrenamiento, impactos, EQUIPO_LOCAL, EQUIPO_VISITANTE)
        
    except Exception as e:
        print(f"\n[!] ERROR: {e}")