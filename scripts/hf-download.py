# Downloads files from a Hugging Face repo into the Hugging Face cache over one plain HTTP connection (no Xet),
# resuming until each file is complete, then checks its SHA-256. Standard library only.
#
# usage: python -I scripts/hf-download.py REPO FILE [FILE ...] [--limit MBPS] [--rev REVISION]
#   e.g. python -I scripts/hf-download.py unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
# The files land where `hf download` puts them (HF_HUB_CACHE, else HF_HOME\hub, else ~\.cache\huggingface\hub,
# snapshots\<commit>\FILE), so arcinfer finds them as usual. A dropped connection or a router that goes away is
# waited out (every 5..60 s, without limit) and the download continues from the partial file (FILE.incomplete next
# to it); run the script again after stopping it and it continues too. --limit caps the average speed (default:
# none). HF_TOKEN is sent only to huggingface.co, for gated or private repos, when it is set.
import argparse, hashlib, http.client, os, sys, time, urllib.error, urllib.parse, urllib.request

def cache_dir():
    if os.environ.get('HF_HUB_CACHE'): return os.environ['HF_HUB_CACHE']
    if os.environ.get('HF_HOME'): return os.path.join(os.environ['HF_HOME'], 'hub')
    return os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')

def request(url, method='GET', start=0):
    r = urllib.request.Request(url, method=method, headers={'User-Agent': 'arcinfer-hf-download/2'})
    if os.environ.get('HF_TOKEN') and urllib.parse.urlparse(url).hostname == 'huggingface.co':
        r.add_header('Authorization', 'Bearer ' + os.environ['HF_TOKEN'])
    if start: r.add_header('Range', f'bytes={start}-')
    return r

# errors of a network that is down or a connection that broke off: wait and try again (anything else is a real error)
NET_ERRORS = (urllib.error.URLError, OSError, TimeoutError, http.client.HTTPException)

def wait_out(e, tries, what):
    wait = min(60, 5 * tries)
    print(f'\n  {what}: {e}; trying again in {wait} s (attempt {tries})', flush=True)
    time.sleep(wait)

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k): return None

def metadata(repo, rev, name):
    # the resolve URL answers with the commit, the size and (LFS files) the SHA-256, then redirects to the CDN
    url = f'https://huggingface.co/{repo}/resolve/{urllib.parse.quote(rev)}/{urllib.parse.quote(name)}'
    opener = urllib.request.build_opener(NoRedirect)
    tries = 0
    while True:
        try:
            try:
                resp = opener.open(request(url, 'HEAD'), timeout=60)
            except urllib.error.HTTPError as e:
                if e.code in (401, 403, 404): sys.exit(f'{repo}/{name}: HTTP {e.code} (wrong name, or a gated repo without HF_TOKEN)')
                if e.code not in (301, 302, 307, 308): raise
                resp = e
            break
        except NET_ERRORS as e:
            tries += 1
            wait_out(e, tries, 'no connection')
    h = resp.headers
    commit = h.get('X-Repo-Commit')
    # (LFS files: X-Linked-Size / X-Linked-Etag; a small git file has neither, and the HEAD's Content-Length is not
    # its size: size 0 = fetch it whole)
    size = int(h.get('X-Linked-Size') or 0)
    sha = (h.get('X-Linked-Etag') or '').strip('"')
    if not commit: sys.exit(f'no file information for {repo}/{name} (HTTP {getattr(resp, "status", "?")})')
    return url, commit, size, sha if len(sha) == 64 else ''

def fetch_small(url, path):
    tries = 0
    while True:
        try:
            with urllib.request.urlopen(request(url), timeout=60) as resp:
                data = resp.read()
            with open(path, 'wb') as f: f.write(data)
            return
        except NET_ERRORS as e:
            tries += 1
            wait_out(e, tries, 'connection lost')

def download(url, path, size, limit):
    rate = limit * 1e6 if limit > 0 else 0
    tries = 0
    while True:
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have >= size: return
        try:
            with urllib.request.urlopen(request(url, start=have), timeout=60) as resp:
                if have and resp.status != 206:  # (a reply from the start: not this time, maybe the next)
                    raise http.client.HTTPException(f'HTTP {resp.status} instead of 206 for a resumed download')
                with open(path, 'ab') as f:
                    t0, got, last = time.monotonic(), 0, 0.0
                    while True:
                        buf = resp.read(1 << 20)
                        if not buf: break
                        f.write(buf)
                        got += len(buf)
                        tries = 0
                        if rate:
                            ahead = got / rate - (time.monotonic() - t0)  # (keep the average at the limit)
                            if ahead > 0: time.sleep(ahead)
                        now = time.monotonic()
                        if now - last >= 1.0 or have + got >= size:
                            last = now
                            done = have + got
                            speed = got / max(now - t0, 1e-6) / 1e6
                            eta = (size - done) / max(speed * 1e6, 1)
                            print(f'\r  {done / 1e9:6.2f} / {size / 1e9:.2f} GB  {100 * done / size:5.1f} %  '
                                  f'{speed:6.1f} MB/s  {int(eta // 60):4d} min left ', end='', flush=True)
        except NET_ERRORS as e:
            tries += 1
            wait_out(e, tries, 'connection lost')

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

def fetch(repo, name, rev, limit):
    url, commit, size, sha = metadata(repo, rev, name)
    root = os.path.join(cache_dir(), 'models--' + repo.replace('/', '--'))
    dest = os.path.join(root, 'snapshots', commit, *name.split('/'))
    if os.path.exists(dest) and (os.path.getsize(dest) == size or not size):
        print('already downloaded:', dest); return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    os.makedirs(os.path.join(root, 'refs'), exist_ok=True)
    part = dest + '.incomplete'
    if not size:  # (a small file kept in git, not LFS)
        fetch_small(url, part)
        os.replace(part, dest)
        if rev == 'main':
            with open(os.path.join(root, 'refs', 'main'), 'w') as f: f.write(commit)
        print(f'{repo}/{name}: {os.path.getsize(dest)} bytes -> {dest}')
        return True
    print(f'{repo}/{name}: {size / 1e9:.2f} GB{f", limit {limit:g} MB/s" if limit > 0 else ""}\n  -> {dest}')
    download(url, part, size, limit)
    print()
    if os.path.getsize(part) != size:
        print(f'  size mismatch: {os.path.getsize(part)} != {size}; delete {part} and run again'); return False
    if sha:
        got = sha256(part, size)
        if got != sha:
            bad = part + '.bad'  # (kept, not deleted: the next run starts the file again)
            os.replace(part, bad)
            print(f'  SHA-256 mismatch ({got} != {sha}): moved to {bad}; run again to download it anew'); return False
        print('  SHA-256 ok')
    os.replace(part, dest)
    if rev == 'main':
        with open(os.path.join(root, 'refs', 'main'), 'w') as f: f.write(commit)
    print('done:', dest)
    return True

def main():
    ap = argparse.ArgumentParser(description='Download Hugging Face files, resuming until they are complete.')
    ap.add_argument('repo')
    ap.add_argument('files', nargs='+')
    ap.add_argument('--limit', type=float, default=0.0, help='average MB/s (default: no limit)')
    ap.add_argument('--rev', default='main')
    a = ap.parse_args()
    ok = True
    for name in a.files:
        ok = fetch(a.repo, name, a.rev, a.limit) and ok
    return 0 if ok else 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nstopped; run the same command again to continue')
        sys.exit(130)
