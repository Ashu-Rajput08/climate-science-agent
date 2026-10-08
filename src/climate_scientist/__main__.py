import argparse

from .workflow import invoke


def main():
    parser = argparse.ArgumentParser(description="Run a reproducible climate investigation")
    parser.add_argument("question", help="Climate research question")
    result = invoke(parser.parse_args().question)
    if result.get("error"):
        raise SystemExit(result["error"])
    payload = result.get("result", result)
    print(payload.get("report", payload))
    if payload.get("report_path"):
        print(f"\nSaved report: {payload['report_path']}")


if __name__ == "__main__":
    main()
