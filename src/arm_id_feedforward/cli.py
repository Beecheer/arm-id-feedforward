"""Small entry points for modules with an argument parser but no main()."""

from . import arm_dynamics


def identify() -> int:
    args = arm_dynamics.build_parser().parse_args()
    return args.handler(args)
