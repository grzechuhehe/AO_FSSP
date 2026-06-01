import numpy as np
import pandas as pd
import os
import re

class FSSPInstance:
    """Klasa reprezentująca instancję problemu Flow Shop Scheduling."""
    def __init__(self, name, n, m, p_matrix, ub=None):
        self.name = name
        self.n = n  # liczba zadań
        self.m = m  # liczba maszyn
        self.p = p_matrix  # macierz czasów operacji (m x n)
        self.ub = ub  # najlepsze znane rozwiązanie (Upper Bound)

    def __repr__(self):
        return f"FSSPInstance(name='{self.name}', n={self.n}, m={self.m}, ub={self.ub})"

def parse_taillard_file(filepath):
    """Parsuje plik tekstowy Taillarda i zwraca listę instancji."""
    instances = []
    with open(filepath, 'r') as f:
        content = f.read()

    # Podział na bloki instancji
    blocks = re.split(r'number of jobs, number of machines, initial seed, upper bound and lower bound :', content)
    
    for block in blocks[1:]:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
            
        # Parametry: n, m, seed, UB, LB
        params = list(map(int, lines[0].split()))
        n, m = params[0], params[1]
        
        # Macierz czasów (pomijamy linię "processing times :")
        matrix_lines = []
        for line in lines[2:]:
            if line.strip():
                matrix_lines.append(list(map(int, line.split())))
            if len(matrix_lines) == m:
                break
                
        p_matrix = np.array(matrix_lines, dtype=np.int32)
        instances.append({'n': n, 'm': m, 'p': p_matrix})
        
    return instances

def load_all_data(directory):
    """Wczytuje wszystkie instancje i dopasowuje wyniki referencyjne z CSV."""
    ub_csv_path = os.path.join(directory, "Taillard_UB_Schedules OBrunner.csv")
    df_ub = pd.read_csv(ub_csv_path)
    
    # Słownik pomocniczy do dopasowania (n, m) -> lista UB i nazw
    reference_data = {}
    for _, row in df_ub.iterrows():
        key = (int(row['n']), int(row['m']))
        if key not in reference_data:
            reference_data[key] = []
        reference_data[key].append({'name': row['Name'], 'ub': int(row['UB'])})
        
    ref_counters = {k: 0 for k in reference_data.keys()}
    all_instances = {}

    # Lista plików do wczytania (tylko tai*.txt)
    files = sorted([f for f in os.listdir(directory) if f.startswith('tai') and f.endswith('.txt')])
    
    for filename in files:
        filepath = os.path.join(directory, filename)
        parsed_list = parse_taillard_file(filepath)
        
        for p_data in parsed_list:
            key = (p_data['n'], p_data['m'])
            if key in reference_data:
                idx = ref_counters[key]
                ref = reference_data[key][idx]
                
                inst = FSSPInstance(
                    name=ref['name'],
                    n=p_data['n'],
                    m=p_data['m'],
                    p_matrix=p_data['p'],
                    ub=ref['ub']
                )
                all_instances[inst.name] = inst
                ref_counters[key] += 1
                
    return all_instances

def calculate_makespan(p_matrix, permutation):
    """
    Oblicza Cmax dla zadanej permutacji (0-indexed).
    p_matrix: macierz (m x n)
    permutation: lista/array indeksów zadań
    """
    m, n = p_matrix.shape
    # times[i] przechowuje aktualny czas zakończenia na i-tej maszynie
    times = np.zeros(m, dtype=np.int32)
    
    for job_idx in permutation:
        # Pierwsza maszyna
        times[0] += p_matrix[0, job_idx]
        # Kolejne maszyny
        for i in range(1, m):
            # Zadanie może zacząć się na maszynie i tylko jeśli:
            # 1. Skończyło się na maszynie i-1 (times[i-1])
            # 2. Maszyna i jest wolna (times[i])
            times[i] = max(times[i-1], times[i]) + p_matrix[i, job_idx]
            
    return times[-1]

def get_full_schedule(p_matrix, permutation):
    """
    Zwraca szczegółowy harmonogram: start_times i end_times dla każdej operacji.
    Obie macierze mają wymiar (m x n).
    """
    m, n = p_matrix.shape
    start_times = np.zeros((m, n), dtype=np.int32)
    end_times = np.zeros((m, n), dtype=np.int32)
    
    # Przechowujemy czas zakończenia ostatniego zadania na każdej maszynie
    last_machine_time = np.zeros(m, dtype=np.int32)
    # Przechowujemy czas zakończenia zadania na poprzedniej maszynie
    last_job_end_time = 0
    
    for j, job_idx in enumerate(permutation):
        for i in range(m):
            # Czas startu to max z (kiedy maszyna wolna, kiedy zadanie gotowe z poprz. maszyny)
            ready_on_machine = last_machine_time[i]
            ready_from_prev_machine = end_times[i-1, j] if i > 0 else 0
            
            start = max(ready_on_machine, ready_from_prev_machine)
            end = start + p_matrix[i, job_idx]
            
            start_times[i, j] = start
            end_times[i, j] = end
            last_machine_time[i] = end
            
    return start_times, end_times

if __name__ == "__main__":
    # Szybki test
    path = "/mnt/c/Users/huber/PycharmProjects/Test/FSSP/"
    all_inst = load_all_data(path)
    print(f"Załadowano {len(all_inst)} instancji.")
    if "Ta001" in all_inst:
        t1 = all_inst["Ta001"]
        print(f"Test Ta001: n={t1.n}, m={t1.m}, UB={t1.ub}")
        # Przykładowa permutacja (0, 1, ..., 19)
        test_perm = np.arange(t1.n)
        print(f"Makespan (0..19): {calculate_makespan(t1.p, test_perm)}")
