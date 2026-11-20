def main() -> None:
    import sys

    from . import cli  # lazy: `import aida_data` stays cheap

    sys.exit(cli.main())
