"""Import an unpacked International RF2 Snapshot into a local SNOMED catalog."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from emr_analyzer.clinical.snomed_catalog import SnomedCatalog


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    catalog=SnomedCatalog(args.output)
    try:
        result=catalog.import_file(args.source,progress=lambda s:print(s,flush=True))
        print(json.dumps(result,ensure_ascii=False,indent=2))
    finally:
        catalog.db.close()


if __name__=='__main__':
    main()
