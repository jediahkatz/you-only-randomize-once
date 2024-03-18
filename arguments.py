from var_ordering import (
    VariableOrderingType,
    CellOrderingType
)
from encode_wfc import WFCEncodingType
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
    solver: Solver
    random_seed: int
    output_size: int = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Implements WaveFunctionCollapse, and allows customizing the variable ordering in order to control the statistics of the output."
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Which example input to use. You can also use a custom image URL.",
        default='zelda'
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
        default=VariableOrderingType.CONTEXT_SENSITIVE,
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
        help="The order the solver will attempt to assing the cells in. Different biases can be observed in the output with each ordering.",
        default=CellOrderingType.COL_MAJOR,
    )
    parser.add_argument(
        "--encoding",
        type=lambda input: WFCEncodingType(str.upper(input)),
        choices=[WFCEncodingType.NEIGHBORHOOD, WFCEncodingType.TILE],
        help="Whether to define variables based on individual tiles or (plus-shaped) neighborhoods in the input",
        default=WFCEncodingType.NEIGHBORHOOD,
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
        default=Solver.PICOSAT,
    )
    parser.add_argument("--seed", type=int, help="The random seed", default=1)
    parser.add_argument("--n", type=int, help="The size of the output grid (N x N)", default=20)

    PRESETS = {
        "zelda": {
            "input": ExampleInput.ZELDA,
            "var_ordering": VariableOrderingType.CONTEXT_SENSITIVE,
            "cell_ordering": CellOrderingType.COL_MAJOR,
            "encoding": WFCEncodingType.NEIGHBORHOOD,
            "solver": Solver.PICOSAT,
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
        "bw-tilefreq": {
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

    if args.preset in PRESETS:
        preset = PRESETS[args.preset]
        for key, value in preset.items():
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
        solver=args.solver,
        random_seed=args.seed,
        output_size=args.n
    )