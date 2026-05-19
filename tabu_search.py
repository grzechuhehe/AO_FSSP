import numpy as np
import time
import copy
from fssp_core import calculate_makespan, load_all_data

def neh_heuristic(instance):
    """
    Heurystyka NEH (Nawaz-Enscore-Ham).
    Zwraca bardzo dobrą permutację początkową.
    """
    # 1. Oblicz sumaryczny czas procesowania dla każdego zadania
    total_times = np.sum(instance.p, axis=0)
    # 2. Posortuj zadania malejąco wg sumarycznego czasu
    order = np.argsort(total_times)[::-1]
    
    # 3. Zacznij od dwóch pierwszych zadań
    current_seq = [order[0]]
    
    for i in range(1, instance.n):
        next_job = order[i]
        best_cmax = float('inf')
        best_pos = -1
        
        # Przetestuj wszystkie możliwe pozycje wstawienia dla next_job
        for pos in range(len(current_seq) + 1):
            temp_seq = current_seq[:pos] + [next_job] + current_seq[pos:]
            cmax = calculate_makespan(instance.p, temp_seq)
            
            if cmax < best_cmax:
                best_cmax = cmax
                best_pos = pos
        
        current_seq.insert(best_pos, next_job)
        
    return current_seq

class TabuSearch:
    def __init__(self, instance, tabu_tenure=10, max_iterations=100, neighborhood_type='insert'):
        self.instance = instance
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.neighborhood_type = neighborhood_type # 'swap' lub 'insert'
        
        # Statystyki zbieżności dla sprawozdania
        self.history = [] 

    def get_neighbors(self, permutation):
        """Generuje sąsiedztwo dla zadanej permutacji."""
        neighbors = []
        n = len(permutation)
        
        if self.neighborhood_type == 'swap':
            for i in range(n):
                for j in range(i + 1, n):
                    neighbor = list(permutation)
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    # Ruch: (zadanie1, zadanie2, stara_poz_i, stara_poz_j)
                    move = ('swap', neighbor[i], neighbor[j])
                    neighbors.append((neighbor, move))
        
        elif self.neighborhood_type == 'insert':
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    neighbor = list(permutation)
                    job = neighbor.pop(i)
                    neighbor.insert(j, job)
                    # Ruch: (które zadanie, na jaką pozycję trafiło)
                    move = ('insert', job, j)
                    neighbors.append((neighbor, move))
                    
        return neighbors

    def solve(self, initial_sol=None):
        """Główna pętla Tabu Search."""
        if initial_sol is None:
            # Domyślnie używamy NEH, bo daje lepsze wyniki na start
            current_solution = neh_heuristic(self.instance)
        else:
            current_solution = list(initial_sol)
            
        current_cmax = calculate_makespan(self.instance.p, current_solution)
        
        best_solution = list(current_solution)
        best_cmax = current_cmax
        
        # Tabu list: move -> iteration_when_expires
        tabu_dict = {}
        
        start_time = time.time()
        
        for it in range(self.max_iterations):
            neighbors = self.get_neighbors(current_solution)
            
            best_neighbor = None
            best_neighbor_cmax = float('inf')
            best_move = None
            
            # Przeszukaj sąsiedztwo
            for neighbor, move in neighbors:
                neighbor_cmax = calculate_makespan(self.instance.p, neighbor)
                
                # Kryterium aspiracji: jeśli ruch jest Tabu, ale daje globalnie najlepszy wynik
                is_tabu = move in tabu_dict and tabu_dict[move] > it
                
                if not is_tabu or neighbor_cmax < best_cmax:
                    if neighbor_cmax < best_neighbor_cmax:
                        best_neighbor_cmax = neighbor_cmax
                        best_neighbor = neighbor
                        best_move = move
            
            # Jeśli nie znaleźliśmy żadnego ruchu (bardzo rzadkie), przerwij
            if best_neighbor is None:
                break
                
            # Aktualizacja rozwiązania bieżącego
            current_solution = best_neighbor
            current_cmax = best_neighbor_cmax
            
            # Aktualizacja najlepszego globalnego
            if current_cmax < best_cmax:
                best_cmax = current_cmax
                best_solution = list(current_solution)
            
            # Dodaj ruch do listy Tabu
            tabu_dict[best_move] = it + self.tabu_tenure
            
            # Zapisz historię do wykresu zbieżności
            self.history.append(best_cmax)
            
        execution_time = time.time() - start_time
        
        return {
            'best_cmax': best_cmax,
            'best_permutation': best_solution,
            'history': self.history,
            'time': execution_time,
            'prd': 100 * (best_cmax - self.instance.ub) / self.instance.ub if self.instance.ub else None
        }

if __name__ == "__main__":
    # Test działania na jednej instancji
    path = "/mnt/c/Users/huber/PycharmProjects/Test/FSSP/"
    instances = load_all_data(path)
    
    inst = instances["Ta001"]
    print(f"Rozpoczynam TS dla {inst.name} (UB={inst.ub})...")
    
    ts = TabuSearch(inst, tabu_tenure=15, max_iterations=50, neighborhood_type='insert')
    result = ts.solve()
    
    print(f"Wynik: {result['best_cmax']}")
    print(f"Błąd PRD: {result['prd']:.2f}%")
    print(f"Czas: {result['time']:.4f}s")
