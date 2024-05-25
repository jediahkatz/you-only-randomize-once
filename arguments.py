from var_ordering import (
    VariableOrderingType,
    CellOrderingType
)
from encode_wfc import WFCEncodingType, GlobalConstraintType
from solve import Solver
from input_grids import ExampleInput, get_input
import argparse
from dataclasses import dataclass
from numpy import ndarray
from typing import Optional, Callable
from PIL.Image import Image

@dataclass
class Arguments:
    input_name: str
    input: ndarray
    display_solution: Optional[Callable[[ndarray], Image]]
    var_ordering_type: VariableOrderingType
    cell_ordering_type: CellOrderingType
    wfc_encoding_type: WFCEncodingType
    path_constraint: GlobalConstraintType
    solver: Solver
    random_seed: Optional[int]
    output_size: int = 20
    output_file: Optional[str] = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Implements WaveFunctionCollapse, and allows customizing the variable ordering in order to control the statistics of the output."
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Which example input to use. You can also use a custom image URL.",
        default=None
    )
    parser.add_argument(
        "--var_ordering",
        type=lambda input: VariableOrderingType(str.upper(input)),
        choices=[
            VariableOrderingType.CONTEXT_SENSITIVE,
            VariableOrderingType.TILE_FREQUENCY,
            VariableOrderingType.TRIVIAL,
            VariableOrderingType.UNIFORM,
        ],
        help="Which preset algorithm to use for variable ordering. The CONTEXT_SENSITIVE algorithm is only available for the NEIGHBORHOOD encoding type.",
        default=None,
    )
    parser.add_argument(
        "--cell_ordering",
        type=lambda input: CellOrderingType(str.upper(input)),
        choices=[
            CellOrderingType.COL_MAJOR,
            CellOrderingType.RANDOM,
            CellOrderingType.ROW_MAJOR,
            CellOrderingType.ROW_MAJOR_BOTTOM_UP,
        ],
        help="The order the solver will attempt to assign the cells in. Different biases can be observed in the output with each ordering.",
        default=None,
    )
    parser.add_argument(
        "--encoding",
        type=lambda input: WFCEncodingType(str.upper(input)),
        choices=[WFCEncodingType.NEIGHBORHOOD, WFCEncodingType.TILE],
        help="Whether to define variables based on individual tiles or (plus-shaped) neighborhoods in the input",
        default=None,
    )
    parser.add_argument(
        "--path_constraint",
        type=lambda input: GlobalConstraintType(str.upper(input)),
        choices=[GlobalConstraintType.PATH_RIGHT_DOWN, GlobalConstraintType.PATH_ALL_DIRECTIONS, GlobalConstraintType.NONE],
        help="Which global path constraint to require in the output. Only works with the Zelda input currently.",
        default=None,
    )
    parser.add_argument(
        "--solver",
        type=lambda input: Solver(str.upper(input)),
        choices=[
            Solver.PICOSAT,
            Solver.ORTOOLS,
            Solver.CLASP,
        ],
        help="Which solver backend to use",
        default=None,
    )
    parser.add_argument("--seed", type=int, help="The random seed", default=None)
    parser.add_argument("--n", type=int, help="The size of the output grid (N x N)", default=20)
    parser.add_argument(
        "--output", 
        type=str, 
        help="The file to save the output image to. If not specified, it will be displayed in a new window.", 
        default=None
    )

    PRESETS = {
        "default": {
            "input": ExampleInput.ZELDA,
            "var_ordering": VariableOrderingType.TILE_FREQUENCY,
            "cell_ordering": CellOrderingType.COL_MAJOR,
            "encoding": WFCEncodingType.TILE,
            "solver": Solver.PICOSAT,
            "path_constraint": GlobalConstraintType.PATH_RIGHT_DOWN,
        },
        "zelda-tile-freq": {
            "input": ExampleInput.ZELDA,
            "var_ordering": VariableOrderingType.CONTEXT_SENSITIVE,
            "cell_ordering": CellOrderingType.COL_MAJOR,
            "encoding": WFCEncodingType.NEIGHBORHOOD,
            "solver": Solver.PICOSAT,
            "path_constraint": GlobalConstraintType.PATH_RIGHT_DOWN,
        },
        "zelda-neighborhood-freq": {
            "input": ExampleInput.ZELDA,
            "var_ordering": VariableOrderingType.TILE_FREQUENCY,
            "cell_ordering": CellOrderingType.COL_MAJOR,
            "encoding": WFCEncodingType.TILE,
            "solver": Solver.PICOSAT,
            "path_constraint": GlobalConstraintType.PATH_RIGHT_DOWN,
        },
        "zelda-no-yoro": {
            "input": ExampleInput.ZELDA,
            "var_ordering": VariableOrderingType.TRIVIAL,
            "cell_ordering": CellOrderingType.COL_MAJOR,
            "encoding": WFCEncodingType.TILE,
            "solver": Solver.PICOSAT,
            "path_constraint": GlobalConstraintType.PATH_RIGHT_DOWN,
        },
        "mario": {
            "input": ExampleInput.MARIO,
            "var_ordering": VariableOrderingType.TILE_FREQUENCY,
            "cell_ordering": CellOrderingType.ROW_MAJOR,
            "encoding": WFCEncodingType.TILE,
            "solver": Solver.ORTOOLS,
        },
        "bw-uniform": {
            "input": ExampleInput.L,
            "var_ordering": VariableOrderingType.UNIFORM,
            "cell_ordering": CellOrderingType.ROW_MAJOR,
            "encoding": WFCEncodingType.TILE,
            "solver": Solver.PICOSAT,
        },
        "bw-tile-freq": {
            "input": ExampleInput.L,
            "var_ordering": VariableOrderingType.TILE_FREQUENCY,
            "cell_ordering": CellOrderingType.ROW_MAJOR,
            "encoding": WFCEncodingType.TILE,
            "solver": Solver.PICOSAT,
        },
    }

    parser.add_argument(
        "--preset",
        type=str,
        choices=PRESETS.keys(),
        help="Base off of a preset configuration, typically an attractive combination of the above options",
    )

    args = parser.parse_args()

    # Use the preset configuration if specified, but allow the user to override individual options
    preset = args.preset or "default"
    if preset in PRESETS:
        preset = PRESETS[preset]
        for key, value in preset.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    elif args.preset is not None:
        raise ValueError(f"Unknown preset {args.preset}")
    
    input, display_solution = get_input(args.input)

    return Arguments(
        input_name=args.input,
        input=input,
        display_solution=display_solution,
        var_ordering_type=args.var_ordering,
        cell_ordering_type=args.cell_ordering,
        wfc_encoding_type=args.encoding,
        path_constraint=args.path_constraint,
        solver=args.solver,
        random_seed=args.seed,
        output_size=args.n,
        output_file=args.output
    )