from enum import Enum
from collections import Counter, defaultdict
import numpy as np
from typing import Callable, Tuple
from itertools import product


# A TileVarOrdering is a function from cell (x, y) and tile t to a unique integer.
TileVarOrdering = Callable[[Tuple[int, int], int], int]
# A NeighborhoodVarOrdering is a function from cell (x, y), tile t, and neighborhood (t_top, t_right, t_bottom, t_left) to a unique integer.
NeighborhoodVarOrdering = Callable[[Tuple[int, int], int, Tuple[int, int, int, int]], int]


class VariableOrderingType(Enum):
  TRIVIAL = 'TRIVIAL'
  UNIFORM = 'UNIFORM'
  TILE_FREQUENCY = 'TILE_FREQUENCY'
  CONTEXT_SENSITIVE = 'CONTEXT_SENSITIVE'

class CellOrderingType(Enum):
  ROW_MAJOR = 'ROW_MAJOR'
  COL_MAJOR = 'COL_MAJOR'
  RANDOM = 'RANDOM'
  ROW_MAJOR_BOTTOM_UP = 'ROW_MAJOR_BOTTOM_UP'

def compute_tile_var_ordering(
  N: int,
  C: int,
  variable_ordering_type: VariableOrderingType,
  cell_ordering_type: CellOrderingType,
  input_grid: np.ndarray,
  seed=1
) -> TileVarOrdering:
  if variable_ordering_type == VariableOrderingType.TRIVIAL:
    return trivial_var_ordering(N, C, cell_ordering_type)
  elif variable_ordering_type == VariableOrderingType.UNIFORM:
    return uniform_random_var_ordering(N, C, cell_ordering_type, seed=seed)
  elif variable_ordering_type == VariableOrderingType.TILE_FREQUENCY:
    tile_frequencies = compute_tile_frequencies(input_grid, C)
    return build_tile_freq_var_ordering(N, C, tile_frequencies, cell_ordering_type, seed=seed)
  elif variable_ordering_type == VariableOrderingType.CONTEXT_SENSITIVE:
    raise NotImplementedError("Context-sensitive variable ordering is only supported for the neighborhood encoding.")
  else:
    raise ValueError("Invalid variable_ordering_type")
  

def compute_neighborhood_var_ordering(
  N: int,
  C: int,
  variable_ordering_type: VariableOrderingType,
  cell_ordering_type: CellOrderingType,
  input_grid: np.ndarray,
  seed = 1,
  was_input_padded = False
) -> Tuple[NeighborhoodVarOrdering, TileVarOrdering]:
  if variable_ordering_type == VariableOrderingType.UNIFORM:
    return uniform_random_var_ordering_neighborhoods(N, C, input_grid, cell_ordering_type, seed=seed)
  elif variable_ordering_type == VariableOrderingType.TRIVIAL:
    return trivial_var_ordering_neighborhoods(N, C, input_grid)
  elif variable_ordering_type == VariableOrderingType.TILE_FREQUENCY:
    raise NotImplementedError("Tile frequency ordering is only supported for the tile-based encoding.")
  elif variable_ordering_type == VariableOrderingType.CONTEXT_SENSITIVE:
    return build_neighborhood_freq_var_ordering(N, C, input_grid, cell_ordering_type, was_input_padded=was_input_padded, seed=seed)
  else:
    raise ValueError("Invalid variable_ordering_type")
  

def compute_neighborhood_frequencies_sparse(input_tiles):
  """
  Compute the frequencies of all (tile, neighborhood) pairs, where the
  neighborhood of a tile is its four ordered adjacent tiles (top, right, bottom, left).
  Instead of storing them all in an array of length C^5, store them in a map
  that only contains the neighborhoods with nonzero frequency.
  """
  N = len(input_tiles)
  frequencies_ctr = defaultdict(int)
  for col, row in product(range(N), repeat=2):
    tile_value = input_tiles[row, col]
    n1, n2, n3, n4 = input_tiles[(row-1)%N, col], input_tiles[row, (col+1)%N], input_tiles[(row+1)%N, col], input_tiles[row, (col-1)%N]
    neighborhood = (n1, n2, n3, n4)
    frequencies_ctr[tile_value, neighborhood] += 1
  return frequencies_ctr


def compute_edge_label_frequencies_per_axis_2d_fixed(input_tiles, C):
  N = len(input_tiles)
  edge_labels = [
      (input_tiles[row, col], input_tiles[row, (col+1)%N] if axis == 'h' else input_tiles[(row+1)%N, col], axis)
      for axis in ('h', 'v')
      for (row, col) in product(range(N), repeat=2)
  ]
  frequencies_ctr = Counter(edge_labels)
  frequencies_arr = np.array([frequencies_ctr[c1, c2, axis] for axis in ('h', 'v') for c1 in range(C) for c2 in range(C)])
  return frequencies_arr


def compute_tile_frequencies(input, C=None):
  """Computes the frequencies of all tiles in ascending order."""
  frequencies_ctr = Counter(np.array(input).flatten())
  tiles = range(C) if C else sorted(frequencies_ctr.keys())
  frequencies_arr = np.array([frequencies_ctr[key] for key in tiles])
  return frequencies_arr


def build_tile_freq_var_ordering(N, C, frequencies_arr, cell_order=CellOrderingType.RANDOM, seed=None) -> TileVarOrdering:
  """
  Constructs an ordering over all boolean SAT variables `assign((x, y), tile)`
  representing "cell (x, y) is assigned [tile]". For each cell (x, y), randomly
  sample a permutation for `assign((x, y), t_1), ..., assign((x, y), t_c)`
  weighted by the frequencies of each tile in the input.

  Outputs a function ordering((x, y), tile) which returns the index of
  `assign((x, y), tile)` in the ordering.

  Allows customizing the order of cells with `cell_order='random'|'rowmajor'|'colmajor'`.
  """
  rng = np.random.default_rng(seed=seed)
  probabilities = frequencies_arr / np.sum(frequencies_arr)
  # for each tile in left-to-right, top-to-bottom order, sample a random permutation of the tiles
  # weighted by the frequencies
  order_of_tile_for_cell = defaultdict(dict)
  for x, y in product(range(N), repeat=2):
    tile_permutation_for_cell = rng.choice(C, C, replace=False, p=probabilities)
    for i, tile in enumerate(tile_permutation_for_cell):
        order_of_tile_for_cell[x, y][tile] = i

  cell_idx_permutation = list(int(i) for i in rng.choice(N * N, N * N, replace=False))

  def ordering(cell, tile, log=False):
    x, y = cell
    cell_idx = 0
    if cell_order == CellOrderingType.RANDOM:
      cell_idx = cell_idx_permutation[x * N + y]
    elif cell_order == CellOrderingType.ROW_MAJOR:
      cell_idx = y * N + x
    elif cell_order == CellOrderingType.COL_MAJOR:
      cell_idx = x * N + y
    elif cell_order == CellOrderingType.ROW_MAJOR_BOTTOM_UP:
      cell_idx = (N - 1 - y) * N + x
    else:
      raise ValueError('Unsupported cell ordering type')
    
    tile_order_for_cell = order_of_tile_for_cell[x, y][tile]

    if log:
      print(
        f"cell {cell}, tile {tile}, tile order for cell {tile_order_for_cell}"
      )
    return (cell_idx * C) + tile_order_for_cell + 1

  return ordering


def trivial_var_ordering(N, C, cell_order=CellOrderingType.ROW_MAJOR) -> TileVarOrdering:
  """
  Constructs a simple ordering over all boolean SAT variables in a specified
  order of (x, y) and then ascending order of the tile.

  Outputs a function ordering((x, y), tile) which returns the index of
  `assign((x, y), t)` in the ordering.
  """
  def ordering(cell, tile):
    x, y = cell
    cell_idx = 0
    if cell_order == CellOrderingType.COL_MAJOR:
      cell_idx = x * N + y
    elif cell_order == CellOrderingType.ROW_MAJOR:
      cell_idx = y * N + x
    elif cell_order == CellOrderingType.RANDOM:
      raise Exception("Random cell ordering not implemented for trivial var ordering.")
    elif cell_order == CellOrderingType.ROW_MAJOR_BOTTOM_UP:
      cell_idx = (N - 1 - y)*N + x
    else:
      raise Exception("Unknown cell ordering type.")
    
    return (cell_idx * C) + tile + 1

  return ordering


def uniform_random_var_ordering(N, C, cell_order=CellOrderingType.RANDOM, seed=None) -> TileVarOrdering:
  uniform_frequencies = [1] * C
  return build_tile_freq_var_ordering(
    N, C, uniform_frequencies, cell_order=cell_order, seed=seed
  )


def sample_variable_ordering_scores(N, C, neighborhood_probabilities, tile_probabilities, cell_order=CellOrderingType.RANDOM, was_input_padded=False, seed=None):
  """
  Outputs a function score(var) which accepts a variable of the form ('assign', (x, y), tile)
  or ('neighborhood', (x, y), t, (t_top, t_right, t_bottom, t_left)) and outputs a unique number.
  The score function defines the ordering of variables, such that variables are
  ordered by ascending score.

  Allows customizing the order of cells with `cell_order='random'|'rowmajor'|'colmajor'`.
  """
  rng = np.random.default_rng(seed=seed)
  cell_idx_permutation = list(int(i) for i in rng.choice(N*N, N*N, replace=False))

  def score(var):
    var_type, (x, y), *rest = var


    # Pick tiles in row-major order
    cell_idx = 0
    if cell_order == CellOrderingType.RANDOM:
      cell_idx = cell_idx_permutation[y*N + x]
    elif cell_order == CellOrderingType.COL_MAJOR:
      cell_idx = x*N + y
    elif cell_order == CellOrderingType.ROW_MAJOR:
      cell_idx = y*N + x
    elif cell_order == CellOrderingType.ROW_MAJOR_BOTTOM_UP:
      cell_idx = (N - 1 - y)*N + x
    else:
      raise ValueError('Unsupported cell ordering type')

    # The label score is computed using the Gumbel-max trick which approximates
    # sampling a random permutation. By taking log(p) - log(-log(random(0,1)))
    # for each item with probability p, we get a series of numbers that represents
    # a weighted permutation of the items (the permutation is in reverse/descending
    # order of the numbers).
    label_score = 0
    if var_type == 'n_var':

      c, neighborhood = rest
      neighborhood_prob = neighborhood_probabilities[c, neighborhood]
      if neighborhood_prob == 0:
        raise 'Should only be calling this for neighborhoods that appear in input'

      label_score = np.log10(neighborhood_prob) + -np.log10(-np.log10(rng.random()))

    elif var_type == 'x_var':

      c, = rest
      tile_prob = tile_probabilities[c]
      label_score = np.log10(tile_prob) + -np.log10(-np.log10(rng.random()))

    else:
      raise 'Unknown var type'

    # we want all x vars to come after all n vars for the same tile idx, so that
    # we only decide based on tile frequency if all neighborhood decisions conflict
    var_type_idx = 0 if var_type == 'n_var' else 1

    if not was_input_padded:
      # Note: we're negating the label scores because they represent a _reversed_ permutation.
      # We want the more frequent samples to come first in the ordering.
      return (cell_idx, var_type_idx, -label_score)

    ignore_padded_neighborhoods = 0
    if var_type == 'n_var':
      c, neighborhood = rest
      ignore_padded_neighborhoods = int(c == C-1 or C-1 in neighborhood)
    else:
      c, = rest
      ignore_padded_neighborhoods = int(c == C-1)

    return (ignore_padded_neighborhoods, cell_idx, var_type_idx, -label_score)

  return score

def build_neighborhood_freq_var_ordering(N, C, input_tiles, cell_order=CellOrderingType.RANDOM, was_input_padded=False, seed=None) -> NeighborhoodVarOrdering:
  """
  Constructs an ordering over all boolean SAT variables of the form `assign((x, y), tile)`
  representing "cell (x, y) is assigned `tile`", or `neighbor((x, y), neighborhood)`
  representing "cell (x, y) and its neighboring tiles are assigned `neighborhood`.

  For each cell (x, y):
  - Randomly sample a permutation for `assign((x, y), t_1), ..., assign((x, y), t_c)`
  weighted by the frequencies of each tile in the input.
  - Randomly sample a permutation for `neighbor((x, y), n_1), ..., neighbor((x, y), n_k)`
  weighted by the frequencies of each neighborhood in the input.

  Outputs a pair of functions `(assign((x,y), t), neighbor((x, y), t, (tt, tr, tb, tl)))`
  which return the index of their respective variables in the ordering.
  """
  # Compute probability of each (tile, neighborhood)
  neighborhood_frequencies_map = compute_neighborhood_frequencies_sparse(input_tiles)
  total_freq = sum(neighborhood_frequencies_map.values())
  neighborhood_probabilities = {
    neighborhood: cnt / total_freq
    for neighborhood, cnt in neighborhood_frequencies_map.items()
  }

  # Compute probability of each tile
  tile_frequencies_arr = compute_tile_frequencies(input_tiles, C)
  tile_probabilities = tile_frequencies_arr / np.sum(tile_frequencies_arr)

  score_for_var = sample_variable_ordering_scores(N, C, neighborhood_probabilities, tile_probabilities,
                                                  cell_order=cell_order, was_input_padded=was_input_padded, seed=seed)

  n_vars = [
      ('n_var', (x, y), c, neighborhood)
      for x, y in product(range(N), repeat=2)
      for c, neighborhood in neighborhood_probabilities.keys()
  ]
  x_vars = [
      ('x_var', (x, y), c)
      for x, y in product(range(N), repeat=2)
      for c in range(C)
  ]
  decision_vars = n_vars + x_vars
  decision_vars.sort(key=score_for_var)
  decision_var_order = {
      var: i for i, var in enumerate(decision_vars)
  }
  def neighborhood(center_pos, c, neighborhood):
    var_idx = int(decision_var_order['n_var', center_pos, c, neighborhood]) + 1
    return var_idx

  def assign(tile_pos, c):
    var_idx = int(decision_var_order['x_var', tile_pos, c]) + 1
    return var_idx

  return (neighborhood, assign)

def trivial_var_ordering_neighborhoods(N, C, input_tiles) -> NeighborhoodVarOrdering:
  neighborhoods = compute_neighborhood_frequencies_sparse(input_tiles).keys()
  n_vars = [
      ('n_var', (x, y), c, neighborhood)
      for x, y in product(range(N), repeat=2)
      for c, neighborhood in neighborhoods
  ]
  x_vars = [
      ('x_var', (x, y), c)
      for x, y in product(range(N), repeat=2)
      for c in range(C)
  ]
  decision_vars = n_vars + x_vars
  decision_vars.sort()
  decision_var_order = {
      var: i for i, var in enumerate(decision_vars)
  }

  def neighborhood(center_pos, c, neighborhood):
    var_idx = int(decision_var_order['n_var', center_pos, c, neighborhood]) + 1
    return var_idx

  def assign(tile_pos, c):
    var_idx = int(decision_var_order['x_var', tile_pos, c]) + 1
    return var_idx

  return neighborhood, assign

def uniform_random_var_ordering_neighborhoods(N, C, input_tiles, cell_order=CellOrderingType.RANDOM, seed=None) -> NeighborhoodVarOrdering:
  neighborhoods = compute_neighborhood_frequencies_sparse(input_tiles).keys()
  num_tiles = N**2

  uniform_neighborhood_probabilities = {
      neighborhood: 1 / len(neighborhoods)
      for neighborhood in neighborhoods
  }
  uniform_tile_probabilities = [1 / num_tiles] * num_tiles
  score_for_var = sample_variable_ordering_scores(N, C, uniform_neighborhood_probabilities, uniform_tile_probabilities, cell_order=cell_order, seed=seed)

  n_vars = [
      ('n_var', (x, y), c, neighborhood)
      for x, y in product(range(N), repeat=2)
      for c, neighborhood in neighborhoods
  ]
  x_vars = [
      ('x_var', (x, y), c)
      for x, y in product(range(N), repeat=2)
      for c in range(C)
  ]
  decision_vars = n_vars + x_vars
  decision_vars.sort(key=score_for_var)
  decision_var_order = {
      var: i for i, var in enumerate(decision_vars)
  }

  def neighborhood(center_pos, c, neighborhood):
    var_idx = int(decision_var_order['n_var', center_pos, c, neighborhood]) + 1
    return var_idx

  def assign(tile_pos, c):
    var_idx = int(decision_var_order['x_var', tile_pos, c]) + 1
    return var_idx

  return (neighborhood, assign)
