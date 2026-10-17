# Downloads one file from a Hugging Face repo into the Hugging Face cache, gently: one HTTP connection, a speed
# limit (default 15 MB/s), resume after a dropped connection, and a SHA-256 check at the end. Plain HTTP only
# (no Xet), standard library only.
#
# usage: python -I scripts/hf-download.py REPO FILE [--limit MBPS] [--rev REVISION]
#   e.g. python -I scripts/hf-download.py unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
# The file lands where `hf download` puts it (HF_HUB_CACHE, else HF_HOME\hub, else ~\.cache\huggingface\hub,
# snapshots\<commit>\FILE), so arcinfer finds it as usual. Run it again after a failure: it continues from the
# partial file (FILE.incomplete next to it). HF_TOKEN is sent only for gated or private repos when it is set.
import argparse, hashlib, os, sys, time, urllib.error, urllib.parse, urllib.request

def cache_dir():
    if os.environ.get('HF_HUB_CACHE'): return os.environ['HF_HUB_CACHE']
    if os.environ.get('HF_HOME'): return os.path.join(os.environ['HF_HOME'], 'hub')
    return os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')

def request(url, method='GET', start=0):
    r = urllib.request.Request(url, method=method, headers={'User-Agent': 'arcinfer-hf-download/1'})
    if os.environ.get('HF_TOKEN') and urllib.parse.urlparse(url).hostname == 'huggingface.co':
        r.add_header('Authorization', 'Bearer ' + os.environ['HF_TOKEN'])
    if start: r.add_header('Range', f'bytes={start}-')
    return r

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k): return None

def metadata(repo, rev, name):
    # the resolve URL answers with the commit, the size and (LFS files) the SHA-256, then redirects to the CDN
    url = f'https://huggingface.co/{repo}/resolve/{urllib.parse.quote(rev)}/{urllib.parse.quote(name)}'
    opener = urllib.request.build_opener(NoRedirect)
    try:
        resp = opener.open(request(url, 'HEAD'), timeout=60)
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 307, 308): raise
        resp = e
    h = resp.headers
    commit = h.get('X-Repo-Commit')
    size = int(h.get('X-Linked-Size') or h.get('Content-Length') or 0)
    sha = (h.get('X-Linked-Etag') or '').strip('"')
    if not commit or not size: sys.exit(f'no file information for {repo}/{name} (HTTP {getattr(resp, "status", "?")})')
    return url, commit, size, sha if len(sha) == 64 else ''

def download(url, path, size, limit):
    rate = limit * 1e6
    tries = 0
    while True:
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have >= size: return
        try:
            with urllib.request.urlopen(request(url, start=have), timeout=60) as resp, open(path, 'ab') as f:
                if have and resp.status != 206: sys.exit('the server does not resume downloads (no HTTP 206)')
                t0, got, last = time.monotonic(), 0, 0.0
                while True:
                    buf = resp.read(1 << 20)
                    if not buf: break
                    f.write(buf)
                    got += len(buf)
                    tries = 0
                    ahead = got / rate - (time.monotonic() - t0)  # (keep the average at the limit)
                    if ahead > 0: time.sleep(ahead)
                    now = time.monotonic()
                    if now - last >= 1.0 or have + got >= size:
                        last = now
                        done = have + got
                        speed = got / max(now - t0, 1e-6) / 1e6
                        eta = (size - done) / max(speed * 1e6, 1)
                        print(f'\r  {done / 1e9:6.2f} / {size / 1e9:.2f} GB  {100 * done / size:5.1f} %  '
                              f'{speed:5.1f} MB/s  {int(eta // 60):3d} min left ', end='', flush=True)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            tries += 1
            wait = min(60, 5 * tries)
            print(f'\n  connection lost ({e}); continuing in {wait} s (attempt {tries})', flush=True)
            time.sleep(wait)

def sha256(path, size):
    h = hashlib.sha256()
    done = 0
    with open(path, 'rb') as f:
        while buf := f.read(16 << 20):
            h.update(buf)
            done += len(buf)
            print(f'\r  checking SHA-256 {100 * done / size:5.1f} %', end='', flush=True)
    print()
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description='Download one Hugging Face file with a speed limit.')
    ap.add_argument('repo')
    ap.add_argument('file')
    ap.add_argument('--limit', type=float, default=15.0, help='MB/s (default 15)')
    ap.add_argument('--rev', default='main')
    a = ap.parse_args()
    url, commit, size, sha = metadata(a.repo, a.rev, a.file)
    root = os.path.join(cache_dir(), 'models--' + a.repo.replace('/', '--'))
    dest = os.path.join(root, 'snapshots', commit, *a.file.split('/'))
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        print('already downloaded:', dest); return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    os.makedirs(os.path.join(root, 'refs'), exist_ok=True)
    part = dest + '.incomplete'
    print(f'{a.repo}/{a.file}: {size / 1e9:.2f} GB, limit {a.limit:g} MB/s\n  -> {dest}')
    download(url, part, size, a.limit)
    print()
    if os.path.getsize(part) != size: sys.exit(f'size mismatch: {os.path.getsize(part)} != {size}; delete {part}')
    if sha:
        got = sha256(part, size)
        if got != sha: sys.exit(f'SHA-256 mismatch ({got} != {sha}): delete {part} and download again')
        print('  SHA-256 ok')
    os.replace(part, dest)
    if a.rev == 'main':
        with open(os.path.join(root, 'refs', 'main'), 'w') as f: f.write(commit)
    print('done:', dest)
    return 0

if __name__ == '__main__':
    sys.exit(main())
