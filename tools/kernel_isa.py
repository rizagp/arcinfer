# Compiles one OpenCL kernel for Meteor Lake with Intel's offline compiler (ocloc, from the oneAPI DPC++/C++ compiler
# install) and summarizes its GPU assembly: instruction count, DP4a, memory messages, scratch (spill) traffic.
# usage: python -I tools/kernel_isa.py FILE.cl KERNEL "BUILD OPTIONS" [out_dir]
# The .asm is kept in out_dir (default: a folder next to this script's temp files) for reading.
# Note: ocloc carries its own compiler version, which can differ slightly from the one in the graphics driver.
import os, re, subprocess, sys, tempfile

OCLOC = os.environ.get('OCLOC', r'C:\Program Files (x86)\Intel\oneAPI\ocloc\2026.1\bin\ocloc.exe')

def main():
    if len(sys.argv) < 4:
        print(__doc__ or 'usage: kernel_isa.py FILE.cl KERNEL "OPTIONS" [out_dir]'); return 1
    src, kernel, opts = os.path.abspath(sys.argv[1]), sys.argv[2], sys.argv[3]
    out = os.path.abspath(sys.argv[4]) if len(sys.argv) > 4 else tempfile.mkdtemp(prefix='kisa_')
    os.makedirs(out, exist_ok=True)
    c = subprocess.run([OCLOC, 'compile', '-file', src, '-device', 'mtl', '-options', opts, '-output', 'k',
                        '-out_dir', out], capture_output=True, text=True)
    notes = [l for l in (c.stdout + c.stderr).splitlines() if kernel in l or 'error' in l.lower()]
    if c.returncode != 0:
        print('\n'.join(notes or (c.stdout + c.stderr).splitlines()[-20:])); return 1
    dis = os.path.join(out, 'dis')
    os.makedirs(dis, exist_ok=True)
    subprocess.run([OCLOC, 'disasm', '-file', os.path.join(out, 'k_mtl.bin'), '-dump', '.'], cwd=dis,
                   capture_output=True, text=True)
    asm = os.path.join(dis, f'.text.{kernel}.asm')
    if not os.path.exists(asm):
        print('kernel not found in the binary:', [f for f in os.listdir(dis) if f.startswith('.text.')]); return 1
    lines = [l for l in open(asm, encoding='utf-8', errors='replace') if l.strip() and not l.strip().endswith(':')]
    instr = [l for l in lines if not l.strip().startswith('//')]
    count = lambda pat: sum(1 for l in instr if re.search(pat, l))
    print(f'{kernel} [{opts}]')
    for n in notes: print('  ocloc:', n.strip())
    print(f'  instructions {len(instr)} | dp4a {count(r"\bdp4a\b")} | send (memory) {count(r"\bsend")}'
          f' | scratch (spill) {count(r"scratch")} | nop/sync {count(r"\bnop\b|sync\.")}')
    print(f'  asm: {asm}')
    return 0

sys.exit(main())
