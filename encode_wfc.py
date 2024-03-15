import numpy as np
from itertools import combinations
from enum import Enum
from var_ordering import compute_neighborhood_frequencies_sparse, compute_edge_label_frequencies_per_axis_2d_fixed

class WFCEncodingType(Enum):
  TILE = 1
  NEIGHBORHOOD = 2

def encode_wfc_formula(N, C, input, var_ordering):
  """
  Outputs a SAT formula (as a list of lists of literals) representing
  an NxN WaveFunctionCollapse problem. The only constraints are that
  every cell must be assigned exactly one tile, and tiles can only
  be adjacent if they have an adjacency in the input.
  """
  assign = var_ordering
  formula = []

  for x in range(N):
    for y in range(N):
      # C1: each slot has ALO tile
      formula.append([assign((x, y), c) for c in range(C)])

      # C2: each slot has AMO tile
      for [c1, c2] in combinations(range(C), 2):
        formula.append([-assign((x, y), c1), -assign((x, y), c2)])

  # C3: tiles can only be adjacent if they have an adjacency in the input
  wrapped_input = np.pad(input, [(1,1),(1,1)], 'wrap')
  # h_adj[c1, c2] = 1 means c1 can be to the left of c2
  h_adj = np.zeros((C,C))
  # v_adj[c1, c2] = 1 means c1 can be above c2
  v_adj = np.zeros((C,C))

  for u,v in np.lib.stride_tricks.sliding_window_view(wrapped_input, window_shape=(1,2)).reshape((-1,2)):
    h_adj[u,v] = 1

  for u,v in np.lib.stride_tricks.sliding_window_view(wrapped_input, window_shape=(2,1)).reshape((-1,2)):
    v_adj[u,v] = 1

  for x in range(N):
    for y in range(N):
      h_neighbor = ((x + 1) % N, y)
      v_neighbor = (x, (y + 1) % N)
      for c in range(C):
        valid_h_colors = [c2 for c2 in range(C) if h_adj[c, c2]]
        valid_v_colors = [c2 for c2 in range(C) if v_adj[c, c2]]
        formula.append([-assign((x, y), c)] + [assign(h_neighbor, c2) for c2 in valid_h_colors])
        formula.append([-assign((x, y), c)] + [assign(v_neighbor, c2) for c2 in valid_v_colors])

  return formula


def encode_wfc_neighborhoods_formula(N, C, input, assign_tile, assign_neighborhood):
  """
  Outputs a SAT formula (as a list of lists of literals) representing
  an NxN WaveFunctionCollapse problem. The only constraints are that
  every cell must be assigned exactly one tile, and tiles can only
  be adjacent if they have an adjacency in the input.

  In this formulation, we have neighborhood variables as well as tile variables.
  You can also call this function with `add_reachability_global_constraint=tiles`,
  which enforces that there's a path from the top-left to the bottom-right using
  tiles from the `tiles` array that only moves right or down.
  """
  formula = []
  input_neighborhoods = list(compute_neighborhood_frequencies_sparse(input).keys())

  ADD = formula.append

  for x in range(N):
    for y in range(N):
      # C1: each slot has ALO color
      ADD([assign_tile((x, y), c) for c in range(C)])

      # C2: each slot has AMO color
      for c1, c2 in combinations(range(C), 2):
        ADD([-assign_tile((x, y), c1), -assign_tile((x, y), c2)])

  for x in range(N):
    for y in range(N):
      for c, (c1, c2, c3, c4) in input_neighborhoods:
        # C3: assign_neighborhood((x, y), c, [_, _, _, _]) => assign_tile((x, y), c)
        ADD([-assign_neighborhood((x, y), c, (c1, c2, c3, c4)), assign_tile((x, y), c)])
        # C4: assign_neighborhood((x, y), _, [c, _, _, _]) => assign_tile((x, y-1), c)
        ADD([-assign_neighborhood((x, y), c, (c1, c2, c3, c4)), assign_tile((x, (y-1) % N), c1)])
        # C5: assign_neighborhood((x, y), _, [_, c, _, _]) => assign_tile((x+1, y), c)
        ADD([-assign_neighborhood((x, y), c, (c1, c2, c3, c4)), assign_tile(((x + 1) % N, y), c2)])
        # C6: assign_neighborhood((x, y), _, [_, _, c, _]) => assign_tile((x, y+1), c)
        ADD([-assign_neighborhood((x, y), c, (c1, c2, c3, c4)), assign_tile((x, (y+1) % N), c3)])
        # C7: assign_neighborhood((x, y), _, [_, _, _, c]) => assign_tile((x-1, y), c)
        ADD([-assign_neighborhood((x, y), c, (c1, c2, c3, c4)), assign_tile(((x - 1) % N, y), c4)])

  # C8: tiles can only be adjacent if they have an adjacency in the input
  frequencies = compute_edge_label_frequencies_per_axis_2d_fixed(input, C)
  invalid_indices = np.where(frequencies == 0)
  invalid_labels = np.array(
      [(c1, c2, axis) for axis in ('h', 'v') for c1 in range(C) for c2 in range(C)]
  )[invalid_indices]
  for (c1_str, c2_str, axis) in invalid_labels:
    # since np arrays must be homogenous, c1 and c2 got casted to str
    c1 = int(c1_str)
    c2 = int(c2_str)
    for x in range(N):
      for y in range(N):
        if axis == 'h':
          ADD([-assign_tile((x, y), c1), -assign_tile(((x+1) % N, y), c2)])
        else: # axis == 'v'
          ADD([-assign_tile((x, y), c1), -assign_tile((x, (y+1) % N), c2)])

  return formula

def add_reachability_global_constraint(N, formula, assign_tile, allowed_tiles):
  ADD = formula.append
  num_vars_before_path_tile = max(abs(v) for clause in formula for v in clause)

  # whether pos has a tile in the set of `allowed_tiles`
  def is_path_tile(pos):
    x, y = pos
    return x*N + y + 1 + num_vars_before_path_tile
  num_vars_before_reachability = num_vars_before_path_tile + N*N

  print(f'{num_vars_before_reachability = }')
  # whether there's a right-and-down path from (0, 0) to pos of only the specified tiles
  def reachable(pos):
    x, y = pos
    return x*N + y + 1 + num_vars_before_reachability

  ADD([reachable((0, 0))])
  ADD([reachable((N-1, N-1))])
  for x in range(N):
    for y in range(N):
      # is_path_tile(pos) => [assign_tile(pos, t_i) for t_i in allowed_tiles]
      ADD([-is_path_tile((x, y))] + [assign_tile((x, y), path_tile) for path_tile in allowed_tiles])
      if (x, y) == (1, 4):
        print([-is_path_tile((x, y))] + [assign_tile((x, y), path_tile) for path_tile in allowed_tiles])

      # assign_tile(pos, path_tile) => is_path_tile(pos)
      for path_tile in allowed_tiles:
        ADD([-assign_tile((x, y), path_tile), is_path_tile((x, y))])

      # reachable(x, y) <=> is_path_tile(x, y) and (reachable(x-1, y) or reachable(x, y-1))
      ADD([-reachable((x, y)), is_path_tile((x, y))])

      if x >= 1 and y >= 1:
        # reachable(x, y) => reachable(x-1, y) or reachable(x, y-1)
        ADD([-reachable((x, y)), reachable((x-1, y)), reachable((x, y-1))])
        # not reachable(x, y) => not is_path_tile(x, y) or not reachable((x-1, y))
        ADD([reachable((x, y)), -is_path_tile((x, y)), -reachable((x-1, y))])
        # not reachable(x, y) => not is_path_tile(x, y) or not reachable((x-1, y))
        ADD([reachable((x, y)), -is_path_tile((x, y)), -reachable((x, y-1))])
      elif x >= 1 and y == 0:
        ADD([-reachable((x, 0)), reachable((x-1, 0))])
        ADD([-is_path_tile((x, 0)), -reachable((x-1, 0)), reachable((x, 0))])
      elif y >= 1 and x == 0:
        ADD([-reachable((0, y)), reachable((0, y-1))])
        ADD([-is_path_tile((0, y)), -reachable((0, y-1)), reachable((0, y))])


def add_padding_nowrap_constraint(N, C, formula, tile_var_ordering):
  """
  If we want to generate outputs that don't wrap, we can pad the
  grid with a special tile and enforce that it appears on
  all the edges and nowhere else.
  """
  assign = tile_var_ordering
  PADDING_TILE = C-1
  for x in range(N):
    for y in range(N):
      if x == 0 or x == N-1 or y == 0 or y == N-1:
        formula.append([assign((x, y), PADDING_TILE)])
      else:
        formula.append([-assign((x, y), PADDING_TILE)])