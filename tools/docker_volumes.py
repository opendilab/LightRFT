import argparse
import os
import sys


def _build_volume_args(volume_str: str) -> str:
    segments = [seg.strip() for seg in volume_str.split(";") if seg.strip()]
    return " ".join(f"-v {seg}" for seg in segments)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="volume_str")
    args = parser.parse_args(argv)

    volume_str = args.volume_str or os.environ.get("DOCKER_VOLUME_STR", "")
    if not volume_str:
        return 0

    output = _build_volume_args(volume_str)
    if output:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
