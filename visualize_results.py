import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from fssp_core import load_all_data, get_full_schedule

def plot_convergence(results_dir, output_file):
    """Generuje wykres zbieżności dla różnych parametrów."""
    plt.figure(figsize=(10, 6))
    
    files = [f for f in os.listdir(results_dir) if f.startswith('conv_')]
    
    for f in sorted(files):
        df = pd.read_csv(os.path.join(results_dir, f))
        label = f.replace('conv_', '').replace('.csv', '').replace('_', ' ')
        plt.plot(df['iteration'], df['makespan'], label=label)
    
    plt.title('Zbieżność algorytmu Tabu Search')
    plt.xlabel('Iteracja')
    plt.ylabel('Makespan (Cmax)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_file)
    plt.close()
    print(f"Wykres zbieżności zapisany w: {output_file}")

def plot_gantt(instance, permutation, output_file):
    """Generuje profesjonalny Diagram Gantta."""
    start_times, end_times = get_full_schedule(instance.p, permutation)
    m, n = instance.p.shape
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Kolory dla zadań
    colors = plt.cm.tab20(np.linspace(0, 1, n))
    
    for i in range(m):
        for j in range(n):
            job_idx = permutation[j]
            duration = end_times[i, j] - start_times[i, j]
            ax.broken_barh([(start_times[i, j], duration)], 
                           (i - 0.4, 0.8), 
                           facecolors=colors[job_idx],
                           edgecolor='black',
                           alpha=0.8)
            # Numer zadania wewnątrz bloku
            if duration > (end_times.max() * 0.02): # tylko jeśli blok nie jest za mały
                ax.text(start_times[i, j] + duration/2, i, str(job_idx+1), 
                        ha='center', va='center', color='white', fontweight='bold', fontsize=8)

    ax.set_xlabel('Czas')
    ax.set_ylabel('Maszyna')
    ax.set_yticks(range(m))
    ax.set_yticklabels([f'M{i+1}' for i in range(m)])
    ax.set_title(f'Diagram Gantta - Instancja {instance.name} (Makespan: {end_times.max()})')
    ax.invert_yaxis()
    plt.grid(True, axis='x', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Diagram Gantta zapisany w: {output_file}")

def plot_prd_comparison(results_dir, output_file):
    """Wykres słupkowy błędu PRD dla różnych instancji."""
    df = pd.read_csv(os.path.join(results_dir, "main_benchmark.csv"))
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df['name'], df['prd'], color='skyblue', edgecolor='navy')
    
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.title('Błąd względny (PRD) względem najlepszych znanych rozwiązań (UB)')
    plt.xlabel('Instancja')
    plt.ylabel('PRD [%]')
    plt.xticks(rotation=45)
    
    # Dodaj wartości nad słupkami
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Wykres PRD zapisany w: {output_file}")

if __name__ == "__main__":
    base_path = "/mnt/c/Users/huber/PycharmProjects/Test/FSSP/"
    results_path = os.path.join(base_path, "results")
    plots_path = os.path.join(base_path, "plots")
    
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
        
    # 1. Wykres zbieżności
    if os.path.exists(results_path):
        plot_convergence(results_path, os.path.join(plots_path, "convergence.png"))
        plot_prd_comparison(results_path, os.path.join(plots_path, "prd_comparison.png"))
    
    # 2. Diagram Gantta (dla pierwszej instancji z benchmarku)
    all_inst = load_all_data(base_path)
    if "Ta001" in all_inst:
        from tabu_search import neh_heuristic
        inst = all_inst["Ta001"]
        # Używamy NEH dla ładnego wykresu
        best_perm = neh_heuristic(inst)
        plot_gantt(inst, best_perm, os.path.join(plots_path, "gantt_ta001.png"))
    
    print("\n--- FAZA 4 ZAKOŃCZONA ---")
    print(f"Wszystkie wizualizacje znajdują się w folderze: {plots_path}")
