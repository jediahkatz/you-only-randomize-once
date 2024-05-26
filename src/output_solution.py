import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def decode_solution_to_grid(N, C, solution, var_ordering, negate_literals=False):
  """ Decodes a satisfying assignment back to a grid of tiles. """
  soln_set = set(solution)
  grid = np.zeros((N,N))
  grid[:] = C
  for x in range(N):
    for y in range(N):
      for tile in range(C):
        var = var_ordering((x, y), tile)
        if negate_literals:
          var *= -1
        tile_assigned_to_cell = var in soln_set
        if tile_assigned_to_cell:
          grid[y, x] = tile
          break
  return grid

def visualize_grid_solution(C, solution, contrast=True, grid_to_img_func=None, output_file=None):
  """ Draws a grid of tiles to the screen. """
  if grid_to_img_func:
    img = grid_to_img_func(solution)
    if output_file:
      img.save(output_file)
    else:
      plt.imshow(img)
      plt.show()
    return

  # If no grid_to_img_func is provided, use a simple color map.
  black = 'dimgray' if contrast else 'black'
  white = 'lightgray' if contrast else 'white'
  cmap = 'terrain' if C > 5 else ListedColormap([black, white, 'red', 'blue', 'yellow'][:C])
  plt.axis('off')
  plt.imshow(solution,cmap=cmap, vmin=0, vmax=C-1);
  if output_file:
    plt.savefig(output_file)
  else:
    plt.show()