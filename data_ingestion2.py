import fastf1
import pandas as pd
import numpy as np
import time

# Configuração do Cache
fastf1.Cache.enable_cache('f1_cache') 

def get_seconds(timedelta_val):
    """Converte Timedelta do Pandas/FastF1 para segundos (float)."""
    if pd.isna(timedelta_val):
        return np.nan
    return timedelta_val.total_seconds()

def process_session(session, season_year):
    """
    Extrai as features de uma sessão específica (uma corrida).
    Retorna uma lista de dicionários (linhas do dataset).
    """
    session_rows = []
    
    # Metadados do evento
    # Usamos .get() no objeto Series para evitar KeyErrors e usamos o season_year passado explicitamente
    try:
        event_name = session.event.get('EventName', 'Unknown Event')
        round_number = session.event.get('RoundNumber', 0)
        location = session.event.get('Location', 'Unknown Location')
        season = season_year # Usa o ano passado por parâmetro, garantido
    except Exception as e:
        print(f"  [Aviso] Erro ao ler metadados do evento: {e}")
        # Mesmo com erro de metadados, tentamos continuar se tivermos o ano
        event_name = 'Unknown'
        round_number = 0
        location = 'Unknown'
        season = season_year
    
    # Tenta obter a data do evento
    try:
        if hasattr(session.event, 'EventDate'):
             event_date = session.event.EventDate.strftime('%Y-%m-%d')
        else:
             event_date = str(session.event.get('EventDate', 'Unknown'))
    except:
        event_date = 'Unknown'

    laps = session.laps
    results = session.results

    if results.empty:
        print("  [Aviso] Resultados vazios para esta sessão.")
        return []

    for driver_code, res in results.iterrows():
        driver_name = res.get("FullName", "Unknown")
        driver_number = res.get("DriverNumber", "Unknown")
        team_name = res.get("TeamName", "Unknown")
        
        # Coletar as voltas do piloto
        try:
            drv_laps = laps.pick_driver(driver_code)
        except:
            drv_laps = pd.DataFrame() 
        
        # Features de Corrida
        grid_position = res.get("GridPosition", np.nan)
        final_pos = res.get("Position", np.nan)
        points = res.get("Points", 0.0)
        status = res.get("Status", "Unknown")
        
        # --- Tempos de Volta ---
        avg_lap_time = np.nan
        best_lap_time = np.nan
        std_lap_time = np.nan
        clean_avg_lap_time = np.nan
        n_stints = np.nan
        compounds_used = ""
        laps_completed = 0
        n_pitstops = 0
        total_pit_time = np.nan

        if not drv_laps.empty:
            laps_completed = len(drv_laps)
            
            # 1. Clean Average
            try:
                clean_laps = drv_laps.pick_quicklaps() 
                if not clean_laps.empty:
                    clean_avg_lap_time = get_seconds(clean_laps["LapTime"].mean())
                    std_lap_time = get_seconds(clean_laps["LapTime"].std())
            except:
                pass # Falha ao filtrar quicklaps, mantém NaN
            
            # 2. General Average & Best Lap
            avg_lap_time = get_seconds(drv_laps["LapTime"].mean())
            best_lap_time = get_seconds(drv_laps["LapTime"].min())

            # --- Pit Stops ---
            pit_stops_data = drv_laps[drv_laps['PitInTime'].notna()]
            n_pitstops = len(pit_stops_data)
            
            if not pit_stops_data.empty and 'PitOutTime' in drv_laps.columns:
                 durations = drv_laps.loc[drv_laps['PitInTime'].notna(), 'PitOutTime'] - \
                             drv_laps.loc[drv_laps['PitInTime'].notna(), 'PitInTime']
                 total_pit_time = get_seconds(durations.sum())
            else:
                 total_pit_time = 0.0 if n_pitstops == 0 else np.nan

            # Stints
            if 'Stint' in drv_laps.columns:
                n_stints = drv_laps['Stint'].nunique()

            # Compostos
            if 'Compound' in drv_laps.columns:
                compounds = drv_laps['Compound'].dropna().unique()
                compounds_used = ", ".join(compounds)

        # Tempo total de prova (Results)
        total_time_sec = get_seconds(res.get("Time", np.nan))

        row = {
            "season": season,
            "round": round_number,
            "event_name": event_name,
            "location": location,
            "event_date": event_date,
            "driver_code": driver_code,
            "driver_number": driver_number,
            "driver_name": driver_name,
            "team_name": team_name,
            "grid_position": grid_position,
            "final_position": final_pos,
            "status": status,
            "points": points,
            "total_time_sec": total_time_sec,
            "fastest_lap_time_sec": best_lap_time, 
            "laps_completed": laps_completed,
            "avg_lap_time_sec": avg_lap_time,
            "best_lap_time_sec": best_lap_time,
            "std_lap_time_sec": std_lap_time,
            "clean_avg_lap_time_sec": clean_avg_lap_time,
            "n_pitstops": n_pitstops,
            "total_pit_time_sec": total_pit_time,
            "n_stints": n_stints,
            "compounds_used": compounds_used
        }

        session_rows.append(row)
    
    return session_rows

def build_full_dataset(start_year, end_year):
    """
    Itera por todos os anos e rodadas, carregando do cache com proteção de rate limit.
    """
    all_data = []

    for year in range(start_year, end_year + 1):
        print(f"\n=== Iniciando Temporada {year} ===")
        
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            print(f"Erro ao obter calendário de {year}: {e}")
            continue

        race_events = schedule[schedule['RoundNumber'] > 0]

        for _, event in race_events.iterrows():
            round_num = event['RoundNumber']
            event_name = event['EventName']
            
            print(f"  -> Round {round_num}: {event_name}...", end=" ")
            
            # Pausa preventiva entre requisições para evitar 429
            time.sleep(3)
            
            success = False
            attempts = 0
            max_attempts = 3
            
            while not success and attempts < max_attempts:
                try:
                    # Carrega a sessão
                    session = fastf1.get_session(year, round_num, 'R')
                    
                    # Tenta carregar os dados
                    session.load(telemetry=False, weather=False, messages=False)
                    
                    # Processa passando o ano explicitamente
                    rows = process_session(session, year)
                    all_data.extend(rows)
                    print(f"OK ({len(rows)} pilotos)")
                    success = True
                    
                except Exception as e:
                    attempts += 1
                    error_msg = str(e)
                    
                    if "429" in error_msg or "Too Many Requests" in error_msg:
                        wait_time = 10 * attempts
                        print(f"\n     [RATE LIMIT] Esperando {wait_time}s antes de tentar novamente...", end=" ")
                        time.sleep(wait_time)
                    else:
                        print(f"FALHOU. Motivo: {e}")
                        break
            
            if not success and attempts == max_attempts:
                print("ABORTADO após várias tentativas.")

    return pd.DataFrame(all_data)

# --- Execução Principal ---

if __name__ == "__main__":
    START_YEAR = 2021
    END_YEAR = 2025

    print(f"Iniciando construção do dataset de {START_YEAR} a {END_YEAR}...")
    print("Nota: Pausas foram adicionadas para respeitar limites da API.")
    
    full_df = build_full_dataset(START_YEAR, END_YEAR)
    
    if not full_df.empty:
        full_df = full_df.sort_values(by=['season', 'round', 'final_position'])
        
        print("\nDataset completo gerado!")
        print(full_df.head())
        print(f"Total de registros: {len(full_df)}")
        
        filename = f'f1_dataset_{START_YEAR}_{END_YEAR}.csv'
        full_df.to_csv(filename, index=False)
        print(f"Salvo em: {filename}")
    else:
        print("Nenhum dado foi gerado.")