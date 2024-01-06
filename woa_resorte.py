# --------------------------------------------------------------
# -------------- Universidad Carlos III de Madrid --------------
# ---- Máster en Ciencia y Tecnología Informática 2023/2024 ----
# ------ Inteligencia Artificial de Inspiración Biológica ------
# Resolución de problema mediante Whale Optimization Algorithm
# ---------------- Docente: Pedro Isasi Viñuela ----------------
# ---------------- Estudiante: Hiram Ochoa Vea -----------------
# --------------------- NIA: 100503653 -------------------------
# ------------------ 07 de enero del 2024 ----------------------
# --------------------------------------------------------------

import numpy as np
import time

# Constants
X_MIN, X_MAX = 0.05, 2.0
Y_MIN, Y_MAX = 0.25, 1.3
Z_MIN, Z_MAX = 2.0, 15.0
ITERATIONS = 500
WHALES = 100
B = 1.0 # Logarithmic spiral constant
R = 0.045 # Quadratic loss constant

# Position class
class Position:
    def __init__(self, x = 0, y = 0, z = 0):
        self.x = x
        self.y = y
        self.z = z

# Functions
def get_random_number(lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound)

def generate_initial_coord(min_coord, max_coord):
    rand_num = get_random_number(0.0, 1.0)
    return min_coord + rand_num * (max_coord - min_coord)

# Penalty functions
def g1(position):
    return 1 - ( ((position.y**2) * position.z) / (71785.0 * position.x**4) )

def g2(position):
    first_part = ( (4 * position.y**2) - (position.x * position.y) ) / ( 12566.0 * ((position.y * position.x**3) - position.x**4) )
    second_part = 1 / (5108.0 * position.x**2)

    return first_part + second_part

def g3(position):
    return 1 - ( (140.45 * position.x) / (position.y**2 * position.z) )

def g4(position):
    return ( (position.x + position.y) / 1.5 ) - 1.0

# WOA-related functions
def evaluate_objective_function(positions_list):
    results_list = []

    for position in positions_list:
        p1 = max( 0, g1(position) )**2
        p2 = max( 0, g2(position) )**2
        p3 = max( 0, g3(position) )**2
        p4 = max( 0, g4(position) )**2

        penalty_value = R*( p1 + p2 + p3 + p4 )

        obj_func_value = (position.z + 2) * position.y * position.x**2

        final_value = obj_func_value + penalty_value

        results_list.append( final_value )

    return results_list

def constrain_position(current_position):
    new_position = Position( current_position.x, current_position.y, current_position.z )

    new_position.x = max( X_MIN, min(current_position.x, X_MAX) )
    new_position.y = max( Y_MIN, min(current_position.y, Y_MAX) )
    new_position.z = max( Z_MIN, min(current_position.z, Z_MAX) )

    return new_position

def position_vector_module( position_vector ):
    return np.sqrt(position_vector.x**2 + position_vector.y**2 + position_vector.z**2)

def update_whale_position( whale, best_whale, random_whale, iteration ):
    a_i = a_j = a_k = 2 - 2 * iteration / ITERATIONS
    a = Position( a_i, a_j, a_k ) # Vector a

    r = Position( get_random_number(0.0, 1.0), get_random_number(0.0, 1.0), get_random_number(0.0, 1.0) ) # Vector r

    A = Position(
            2 * a.x * r.x - a.x,
            2 * a.y * r.y - a.y,
            2 * a.z * r.z - a.z
        ) # Vector A
    
    C = Position( 2 * r.x, 2 * r.y, 2 * r.z ) # Vector C

    p = get_random_number(0.0, 1.0)

    X_new = Position()

    if p < 0.5:
        # Encircling prey phase when |A| < 1 and Search for prey phase when otherwise.
        best_or_rand = best_whale if position_vector_module(A) < 1.0 else random_whale

        Dx = abs( C.x * best_or_rand.x - whale.x )
        Dy = abs( C.y * best_or_rand.y - whale.y )
        Dz = abs( C.z * best_or_rand.z - whale.z )

        X_new.x = best_or_rand.x - A.x * Dx
        X_new.y = best_or_rand.y - A.y * Dy
        X_new.z = best_or_rand.z - A.z * Dz
        # End of encircling prey / search for prey.
    else:
        L = get_random_number(-1.0, 1.0)

        D_prime_x = abs(best_whale.x - whale.x)
        D_prime_y = abs(best_whale.y - whale.y)
        D_prime_z = abs(best_whale.z - whale.z)

        X_new.x = D_prime_x * np.exp(B * L) * np.cos(2 * np.pi * L) + best_whale.x
        X_new.y = D_prime_y * np.exp(B * L) * np.cos(2 * np.pi * L) + best_whale.y
        X_new.z = D_prime_z * np.exp(B * L) * np.cos(2 * np.pi * L) + best_whale.z

    X_new = constrain_position( X_new )

    return X_new

def get_best_whale( whale_positions, whale_values ):
    best_whales = [(position, value) for position, value in zip(whale_positions, whale_values)]
    best_whales.sort(key=lambda x: x[1])

    best_whale = best_whales[0] # The one with the minimum value

    return {
        'position': best_whale[0],
        'value': best_whale[1]
    }

def select_random_whale_except( current_whale_num ):
    random_whale_num = -1

    while random_whale_num == current_whale_num:
        random_whale_num = np.random.randint( 0, WHALES )

    return random_whale_num

def print_whale_data( whale, iteration ):
    value = whale['value']
    position = whale['position']

    if iteration > 0:
        print(f"Iteración: {iteration}, Fitness: {value}, Valores: [{position.x}, {position.y}, {position.z}]")
        
    else: # Final result
        print()
        print(f"--- Peso mínimo del resorte de tensión/compresión: {value}")
        print(f"--- Díametro del alambre (d) = {position.x}")
        print(f"--- Diámetro medio de la bobina (D) = {position.y}")
        print(f"--- Número de bobinas activas (N) = {position.z}")
        print()

def woa_execute():
    # initial_position  -> Initial whale position. List size: 'WHALES' elements.
    # initial_value     -> Initial whale value. List size: 'WHALES' elements.

    start_time = time.time()

    initial_position = [Position(
        generate_initial_coord( X_MIN, X_MAX ),
        generate_initial_coord( Y_MIN, Y_MAX ),
        generate_initial_coord( Z_MIN, Z_MAX )
    ) for _ in range(WHALES)]

    initial_value = evaluate_objective_function( initial_position )

    best_whale = get_best_whale( initial_position, initial_value )

    current_position = initial_position
    current_value = initial_value

    # Iterations
    for i in range( 1, ITERATIONS ):
        best_whale_position = best_whale['position']

        for whale_id in range( WHALES ):
            random_whale_id = select_random_whale_except( whale_id )
            current_position[whale_id] = update_whale_position( current_position[whale_id], best_whale_position, current_position[random_whale_id], i )

        current_value = evaluate_objective_function( current_position )
        new_best_whale = get_best_whale( current_position, current_value )

        if ( new_best_whale['value'] < best_whale['value'] ):
            best_whale = new_best_whale

        print_whale_data( best_whale, i )

    best_whale = get_best_whale( current_position, current_value )

    print_whale_data( best_whale, iteration = -1 )

    end_time = time.time()
    print(f"Tiempo de procesamiento: {end_time - start_time} segundos")

woa_execute()