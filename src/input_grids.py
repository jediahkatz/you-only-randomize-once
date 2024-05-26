import numpy as np
import requests
import io
from enum import Enum
from PIL import Image
from os import path
from typing import Optional, Callable

class ExampleInput(Enum):
  STICK = 'STICK'
  DOT = 'DOT'
  L = 'L'
  ZELDA = 'ZELDA'
  MARIO = 'MARIO'

N = 7
C = 2
INPUT_2D_STICK = np.zeros((N,N),dtype=int)
INPUT_2D_STICK[1:N-1, N//2] = 1

INPUT_2D_DOT = np.zeros((N,N),dtype=int)
INPUT_2D_DOT[N//2, N//2] = 1

INPUT_2D_L = np.zeros((N,N),dtype=int)
INPUT_2D_L[N//2, N//2] = 1
INPUT_2D_L[N//2-1, N//2] = 1
INPUT_2D_L[N//2, N//2+1] = 1

def load_image_from_url(image_url) -> Image.Image:
  response = requests.get(image_url)
  if response.status_code == 200:
    # Open the image using Pillow (PIL)
    return Image.open(io.BytesIO(response.content))
  raise ValueError(f'Failed to fetch the image. Status code: {response.status_code}')

def load_image_from_file(file_path) -> Image.Image:
  return Image.open(file_path)

def get_input_image(image: Image.Image):
  """
  Download a map example from a URL and convert it to an array of tile indices.
  Returns a grid of tiles along with a function recreate_image(tile_grid)
  that creates an Image from the tile grid.
  """
  # Get the image dimensions
  width, height = image.size

  # Define the tile size (16x16)
  tile_size = 16

  # Create a dictionary to map unique tiles to IDs
  tile_id_map = {}
  id_to_tile_map = {}
  current_id = 0

  # Create a NumPy array to store the tile IDs
  tile_grid = np.zeros((height // tile_size, width // tile_size), dtype=np.int32)

  # Loop through the image and map unique tiles to IDs
  for y in range(0, height, tile_size):
    for x in range(0, width, tile_size):
      tile = image.crop((x, y, x + tile_size, y + tile_size))
      tile_data = tile.tobytes()

      if tile_data not in tile_id_map:
        tile_id_map[tile_data] = current_id
        id_to_tile_map[current_id] = tile
        current_id += 1

      tile_id = tile_id_map[tile_data]
      tile_grid[y // tile_size, x // tile_size] = tile_id

  print(f'Num unique tiles: {len(id_to_tile_map.keys())}')
  def image_from_grid(solution: np.ndarray):
    new_width, new_height = solution.shape[1] * tile_size, solution.shape[0] * tile_size
    reconstructed_image = Image.new('RGB', (new_width, new_height), 'white')

    for y in range(solution.shape[0]):
      for x in range(solution.shape[1]):
        tile_id = solution[y, x]
        if tile_id in id_to_tile_map:
          tile = id_to_tile_map[tile_id]
          reconstructed_image.paste(tile, (x * tile_size, y * tile_size))

    return reconstructed_image

  return tile_grid, image_from_grid

def get_zelda_input():
  zelda_image = load_image_from_file('inputs/zeldaMap.png')
  return get_input_image(zelda_image)

def get_mario_input():
  mario_image = load_image_from_file('inputs/marioMap.png')
  grid, image_from_grid = get_input_image(mario_image)
  PADDING_TILE = len(np.unique(grid))
  grid = np.pad(grid, 1, mode='constant', constant_values=PADDING_TILE)
  return grid, lambda soln: image_from_grid(soln[1:-1, 1:-1])

def get_input(input: str) -> tuple[np.ndarray, Optional[Callable[[np.ndarray], Image.Image]]]:
  if input == ExampleInput.STICK:
    return INPUT_2D_STICK, None
  elif input == ExampleInput.DOT:
    return INPUT_2D_DOT, None
  elif input == ExampleInput.L:
    return INPUT_2D_L, None
  elif input == ExampleInput.ZELDA:
    return get_zelda_input()
  elif input == ExampleInput.MARIO:
    return get_mario_input()
  # if input is a local file
  elif path.exists(input):
    raise ValueError("File input not implemented yet")
  # if input is a URL
  elif input.startswith('http'):
    return get_input_image(input)
  
  raise ValueError(f"Invalid input {input}")