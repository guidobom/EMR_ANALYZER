"""Reproduce the read-only static inventory for the October 2026 review.

Only source files are read; generated JSON is written to the supplied output.
Lexical unused candidates are not proof that an API is dead.
"""
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tokenize
import io


def main():
    paths = sorted([*Path('emr_analyzer').rglob('*.py'), *Path('tools').rglob('*.py'), Path('run.py')])
    paths = [p for p in paths if not p.name.startswith('code_review_20261001')]
    trees, inventory, failures = {}, [], []
    bodies, names, definitions = defaultdict(list), Counter(), []
    modules = {'.'.join(p.with_suffix('').parts).removesuffix('.__init__') for p in paths}
    unresolved, unused_imports, unreachable, repeated = [], [], [], []
    for path in paths:
        source = path.read_text(encoding='utf-8')
        try:
            tree = ast.parse(source, str(path))
            compile(source, str(path), 'exec')
        except Exception as exc:
            failures.append({'path': str(path), 'error': str(exc)})
            continue
        trees[path] = tree
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.NAME:
                names[token.string] += 1
        funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        inventory.append({'path': str(path), 'lines': len(source.splitlines()),
                          'functions': len(funcs), 'classes': sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree)),
                          'sha256': hashlib.sha256(source.encode()).hexdigest(),
                          'symbols': [{'name': n.name, 'line': n.lineno, 'end': n.end_lineno} for n in funcs]})
        referenced = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        exports = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    bound = alias.asname or (alias.name.split('.')[0] if isinstance(node, ast.Import) else alias.name)
                    if bound not in referenced and bound not in exports and bound != '*' and getattr(node, 'module', None) != '__future__':
                        unused_imports.append({'path':str(path),'line':node.lineno,'name':bound})
                if isinstance(node, ast.ImportFrom):
                    if node.level:
                        base = list(path.parent.parts)
                        base = base[:len(base)-node.level+1]
                        target = '.'.join(base + ((node.module or '').split('.') if node.module else []))
                    else:
                        target = node.module or ''
                    if target.startswith(('emr_analyzer', 'tools')) and target not in modules:
                        unresolved.append({'path':str(path),'line':node.lineno,'module':target})
            if isinstance(node, (ast.Module, ast.ClassDef)):
                found = defaultdict(list)
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        found[child.name].append(child.lineno)
                repeated.extend({'path':str(path),'scope':getattr(node,'name','module'),'name':name,'lines':lines}
                                for name,lines in found.items() if len(lines)>1)
        for node in funcs:
            definitions.append({'path':str(path),'line':node.lineno,'name':node.name})
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                body = body[1:]
            if sum(1 for _ in ast.walk(node)) >= 60:
                bodies[ast.dump(ast.Module(body=body,type_ignores=[]),include_attributes=False)].append(definitions[-1])
            for parent in ast.walk(node):
                for _, sequence in ast.iter_fields(parent):
                    if isinstance(sequence,list):
                        for i, statement in enumerate(sequence[:-1]):
                            if isinstance(statement,(ast.Return,ast.Raise,ast.Break,ast.Continue)) and isinstance(sequence[i+1],ast.stmt):
                                unreachable.append({'path':str(path),'function':node.name,'line':sequence[i+1].lineno})
    result = {'scope':'Current working tree; source-only static review',
              'files':len(inventory),'lines':sum(i['lines'] for i in inventory),
              'functions':sum(i['functions'] for i in inventory),'syntax_errors':failures,
              'unresolved_local_modules':unresolved,'duplicate_definitions':repeated,
              'exact_body_duplicates':[v for v in bodies.values() if len(v)>1],
              'unreachable_candidates':unreachable,
              'lexically_unused_function_candidates':[d for d in definitions if names[d['name']]==1 and not d['name'].startswith('__')],
              'unused_import_candidates':unused_imports,'inventory':inventory}
    Path(sys.argv[1]).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('inventory','unused_import_candidates','lexically_unused_function_candidates')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
