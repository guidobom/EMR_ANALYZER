"""Read-only Python inventory and exact-body duplicate/dead-code candidates.

Usage: python tools/audit_code.py > docs/CODE_INVENTORY.json
Static candidates require review: Qt callbacks and compatibility exports are
not reliably classifiable as unused by a lexical scan.
"""
from __future__ import annotations

import ast
from collections import defaultdict
import json
from pathlib import Path


def audit(root):
    inventory, unreachable = [], []
    bodies = defaultdict(list)
    for path in sorted(root.rglob('*.py')):
        source = path.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(path))
        functions = [node for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        inventory.append({'path': str(path), 'lines': len(source.splitlines()),
                          'functions': len(functions)})
        for node in functions:
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body = body[1:]
            if sum(1 for _ in ast.walk(node)) >= 60:
                key = ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)
                bodies[key].append({'path': str(path), 'line': node.lineno, 'name': node.name})
            for index, statement in enumerate(body[:-1]):
                if isinstance(statement, (ast.Return, ast.Raise)):
                    unreachable.append({'path': str(path), 'line': body[index + 1].lineno,
                                        'function': node.name})
                    break
    return {'scope': 'Python application sources; static scan, not proof of runtime coverage',
            'file_count': len(inventory), 'line_count': sum(i['lines'] for i in inventory),
            'inventory': inventory,
            'exact_body_duplicate_candidates': [items for items in bodies.values() if len(items) > 1],
            'unreachable_after_return_or_raise': unreachable}


if __name__ == '__main__':
    print(json.dumps(audit(Path('emr_analyzer')), ensure_ascii=False, indent=2))
