from var_ordering import VariableOrderingType, CellOrderingType, compute_tile_var_ordering, compute_neighborhood_var_ordering
from encode_wfc import WFCEncodingType, encode_wfc_formula, encode_wfc_neighborhoods_formula, add_reachability_global_constraint, add_padding_nowrap_constraint
from solve import solve, Solver
from output_solution import decode_solution_to_grid, visualize_grid_solution
from input_grids import ExampleInput, get_mario_input, get_zelda_input
import numpy as np
from time import time

input_type = ExampleInput.ZELDA
var_ordering_type = VariableOrderingType.CONTEXT_SENSITIVE
cell_ordering_type = CellOrderingType.COL_MAJOR
wfc_encoding_type = WFCEncodingType.NEIGHBORHOOD
solver = Solver.PICOSAT
random_seed = 1

# Whether to add a constraint that there must be a path from the top-left to the bottom-right
# that only moves right and down, using only tiles in `path_tiles_for_reachability_constraint`.
use_reachability_global_constraint = True
path_tiles_for_reachability_constraint = [5]
# Whether to add a constraint that the edges of the output must be a special padding tile.
# This is a useful trick for generating tilemaps that don't wrap around. Only works when
# the input has been padded with the C+1 tile (so just the mario example).
use_padding_constraint = False

if __name__ == "__main__":
  seed = random_seed
  N = 20
  input_grid, reconstruct_img_from_grid = get_zelda_input()
  C = len(np.unique(input_grid))
  tile_var_ordering = None

  t = time()

  if wfc_encoding_type == WFCEncodingType.TILE:
    tile_var_ordering = compute_tile_var_ordering(
      N, C, var_ordering_type, cell_ordering_type, input_grid, seed
    )
    formula = encode_wfc_formula(N, C, input_grid, tile_var_ordering)
    if use_padding_constraint:
      add_padding_nowrap_constraint(N, C, formula, tile_var_ordering)
  elif wfc_encoding_type == WFCEncodingType.NEIGHBORHOOD:
    neighborhood_var_ordering, tile_var_ordering = compute_neighborhood_var_ordering(
      N, C, var_ordering_type, cell_ordering_type, input_grid, seed=seed, was_input_padded=use_padding_constraint
    )
    formula = encode_wfc_neighborhoods_formula(
      N, C, input_grid, tile_var_ordering, neighborhood_var_ordering
    )
  else: 
    raise ValueError("Unsupported WFC encoding type")
  
  if use_reachability_global_constraint:
    add_reachability_global_constraint(N, formula, tile_var_ordering, path_tiles_for_reachability_constraint)

  print(f'Formula encoded in {time() - t}s. Solving...')
  t = time()

  solution = solve(formula, solver)
  if solution is not None:
    solution_grid = decode_solution_to_grid(N, C, solution, tile_var_ordering)
    print(f'Solution found in {time() - t}s! Opening image in new window...')
    visualize_grid_solution(C, solution_grid, grid_to_img_func=reconstruct_img_from_grid)
  else:
    print("No solution found. This should not happen, and probably indicates a bug in the encoding.")