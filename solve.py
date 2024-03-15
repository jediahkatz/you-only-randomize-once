from enum import Enum
from math import ceil
from collections import defaultdict
import pycosat
import subprocess

class Solver(Enum):
  PICOSAT = 1
  ORTOOLS = 2
  CLASP = 3
  PENNSAT = 4

def solve(formula, solver: Solver):
  if solver == Solver.PICOSAT:
    modified_formula = transform_formula_for_solver(formula, negate_literals=True, hack_jw_heuristic=True)
    soln = pycosat.solve(modified_formula)
    if soln == 'UNSAT':
      return None
    return negate_literals_in_solution(soln)
  elif solver == Solver.CLASP:
    modified_formula = transform_formula_for_solver(formula, negate_literals=True, hack_jw_heuristic=False)
    soln = solve_formula_with_clasp(modified_formula)
    if soln is None:
      return None
    return negate_literals_in_solution(soln)
  elif solver == Solver.ORTOOLS:
    raise ValueError("ORTOOLS solver is not implemented yet")
  elif solver == Solver.PENNSAT:
    raise ValueError("The PennSAT solver is not included with this repo, sorry!")
  else:
    raise ValueError("Unsupported solver")

def transform_formula_for_solver(formula, negate_literals=False, hack_jw_heuristic=False):
  """
  Processes the formula to account for internal details of the SAT solver.
  """
  new_formula = formula[:]

  # Replaces each literal v with its negation -v.
  # This is necessary for solvers that try assigning variables to False first.
  if negate_literals:
    new_formula = [[-v for v in clause] for clause in new_formula]

  # Pads the formula with extra clauses to fool the Jeroslaw-Wang heuristic.
  # These extra clauses modify the JW statistic so that the solver assigns
  # variables to True (or False if negate_literals=True) first.
  assign_false_first = negate_literals
  if hack_jw_heuristic:
    num_vars = max(abs(v) for clause in formula for v in clause)
    activity_scores = defaultdict(int)
    for clause in new_formula:
      for literal in clause:
        polarity = 1 if literal > 0 else -1
        activity_scores[abs(literal)] += polarity * 2**(-len(clause))

    clauses_added = 0
    vars_added = 0
    for literal, score in activity_scores.items():
      # each clause [-v, dummy] will add -2^(-2) = -0.25 to the score
      if assign_false_first:
        num_length_2_clauses_to_add = ceil(score / 0.25) if score >= 0 else 0
      # each clause [v, dummy] will add 2^(-2) = 0.25 to the score
      else:
        num_length_2_clauses_to_add = ceil(-score / 0.25) if score <= 0 else 0
      clauses_added += num_length_2_clauses_to_add
      vars_added = max(vars_added, num_length_2_clauses_to_add)
      for j in range(num_length_2_clauses_to_add):
        dummy_var = num_vars + 1 + j
        literal_to_add = -literal if assign_false_first else literal
        new_formula.append([literal_to_add, dummy_var])

  return new_formula

def negate_literals_in_solution(solution):
  return [-v for v in solution]

def solve_formula_with_clasp(formula, preprocess=False, default_sign='neg', heuristic='None'):
  write_formula_to_dimacs_file('/tmp/wfc-formula.cnf', formula)
  command = ['clasp', f'--heuristic={heuristic}', '--sat-prepro=no' if not preprocess else '', f'--sign-def={default_sign}', '/tmp/wfc-formula.cnf']
  output_lines = subprocess.run(command, encoding='utf-8', stdout=subprocess.PIPE).stdout.split('\n')
  if 'command not found' in output_lines[0]:
    raise ValueError("Clasp is not installed. Please install it with `brew install clasp` (see https://potassco.org/clingo/ if you don't have `brew`) or use a different solver.")
  solution = parse_clasp_output(output_lines)
  return solution

def write_formula_to_dimacs_file(filename, formula):
  """
  Writes a CNF formula to a local file in the DIMACS format.
  """
  num_vars = max(abs(v) for clause in formula for v in clause)
  num_clauses = len(formula)
  with open(filename, 'w') as file:
    file.write(f'p cnf {num_vars} {num_clauses}\n')
    for clause in formula:
      clause_strs = [str(literal) for literal in clause]
      file.write(f'{" ".join(clause_strs)} 0\n')

def parse_clasp_output(output_lines):
  """
  Takes in the stdout from the Clasp SAT solver as a list of lines and returns
  the solution as a list of true literals. Assumes solving was successful.
  """
  # solution line should be of the form 'v -1 2 3' or 'v -1 2 3 0'
  solution_literals = []
  if any('s UNSATISFIABLE' in line for line in output_lines):
    return None
  solution_lines = (line for line in output_lines if line and line[0] == 'v')
  for solution_line in solution_lines:
    line_strs = solution_line[2:].split(' ')
    solution_literals.extend(int(literal) for literal in line_strs if literal != '0')
  return solution_literals
