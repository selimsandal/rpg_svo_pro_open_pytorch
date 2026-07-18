"""Allow ``python -m svo_torch`` to invoke the command-line interface."""

from .cli import main

raise SystemExit(main())
