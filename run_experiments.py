import pandas as pd
import numpy as np
import os
from fssp_core import load_all_data
from tabu_search import TabuSearch, neh_heuristic
import matplotlib.pyplot as plt

def run_parameter_study(instance, results_dir):
    """Badanie wpływu typu sąsiedztwa i kadencji Tabu."""
    print(f"--- Badanie parametrów dla {instance.name} ---")
    
    results = []
    tenures = [5, 10, 20, 40]
    neighborhoods = ['swap', 'insert']
    
    for n_type in neighborhoods:
        for tenure in tenures:
            print(f"Testowanie: {n_type}, tenure={tenure}...")
            ts = TabuSearch(instance, tabu_tenure=tenure, max_iterations=200, neighborhood_type=n_type)
            res = ts.solve()
            
            results.append({
                'neighborhood': n_type,
                'tenure': tenure,
                'makespan': res['best_cmax'],
                'prd': res['prd'],
                'time': res['time']
            })
            
            # Zapisz historię zbieżności do osobnego pliku dla wykresów
            history_df = pd.DataFrame({'iteration': range(len(res['history'])), 'makespan': res['history']})
            history_df.to_csv(os.path.join(results_dir, f"conv_{n_type}_t{tenure}.csv"), index=False)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(results_dir, "parameter_study.csv"), index=False)
    print("Wyniki badania parametrów zapisane.")

def run_main_benchmark(all_instances, results_dir):
    """Uruchomienie algorytmu na zestawie różnych instancji."""
    print(f"--- Główny benchmark (skalowalność) ---")
    
    # Wybieramy po kilka instancji z każdego rozmiaru dla reprezentatywności
    selected_names = [
        "Ta001", "Ta005", "Ta010", # 20x5
        "Ta011", "Ta015", "Ta020", # 20x10
        "Ta021", "Ta025", "Ta030", # 20x20
        "Ta031", "Ta035", "Ta040"  # 50x5
    ]
    
    results = []
    
    for name in selected_names:
        if name not in all_instances: continue
        
        inst = all_instances[name]
        print(f"Przetwarzanie {name} ({inst.n}x{inst.m})...")
        
        # Używamy zoptymalizowanych parametrów (np. insert, tenure=20)
        ts = TabuSearch(inst, tabu_tenure=20, max_iterations=300, neighborhood_type='insert')
        res = ts.solve()
        
        results.append({
            'name': name,
            'n': inst.n,
            'm': inst.m,
            'ub': inst.ub,
            'our_cmax': res['best_cmax'],
            'prd': res['prd'],
            'time': res['time']
        })
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(results_dir, "main_benchmark.csv"), index=False)
    print("Wyniki benchmarku zapisane.")

if __name__ == "__main__":
    base_path = "/mnt/c/Users/huber/PycharmProjects/Test/FSSP/"
    results_path = os.path.join(base_path, "results")
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        
    print("Wczytywanie danych...")
    all_inst = load_all_data(base_path)
    
    # 1. Badanie parametrów na średniej instancji (np. Ta021 - 20x20)
    if "Ta021" in all_inst:
        run_parameter_study(all_inst["Ta021"], results_path)
    
    # 2. Główny benchmark na różnych rozmiarach
    run_main_benchmark(all_inst, results_path)
    
    print("\n--- FAZA 3 ZAKOŃCZONA ---")
    print(f"Wszystkie dane statystyczne znajdują się w folderze: {results_path}")
