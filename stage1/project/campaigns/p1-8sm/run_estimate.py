import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from codesign.challenge.hardware import Hardware
from codesign.challenge.runner import estimate_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    hw = Hardware.from_dict(json.loads(open(args.hardware).read()))
    text = open(args.program).read()
    t0 = time.time()
    result = estimate_case(args.case, hw, text)
    elapsed = time.time() - t0
    result["elapsed_seconds"] = elapsed
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"elapsed_seconds={elapsed:.1f}")
    print(f"cycles={result.get('cycles')}")


if __name__ == "__main__":
    main()
