from __future__ import print_function

import argparse
import io
import json


parser = argparse.ArgumentParser(description="Convert a JSON array to JSONLines")
parser.add_argument("input", help="Input JSON array file")
parser.add_argument("output", help="Output JSONLines file")
args = parser.parse_args()

print("Reading JSON array from: %s" % args.input)

with io.open(args.input, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total records: %d" % len(data))
print("Writing JSONLines to: %s" % args.output)

with io.open(args.output, "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Done! Converted %d records to JSONLines format." % len(data))
