import glob
import py_compile
import traceback

files = sorted(glob.glob('amd_tuned_torch/*.py') + glob.glob('tests/*.py') + ['setup.py', 'tools/bench.py'])
results = []
for f in files:
    try:
        py_compile.compile(f, doraise=True)
    except Exception:
        results.append((f, traceback.format_exc()))

print('CHECKED', len(files))
print('ERRORS', len(results))
for f, tb in results:
    print('---', f)
    print(tb)
