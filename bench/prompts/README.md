# Benchmark and accuracy prompts (token ids)

Comma-separated token ids, fed with `bin\runz.exe <model> --tokens-file <file>`.

| File | Tokens | Use |
|---|---|---|
| `qwen-code-11k.ids.txt`, `qwen-code-34k.ids.txt` | 11,240 / 33,937 | time to first token and long-context generation, Qwen tokenizer (code scans of two local projects) |
| `gemma-code-11k.ids.txt`, `gemma-code-34k.ids.txt` | 13,921 / 35,957 | the same texts with the Gemma 4 tokenizer |
| `gemma-code160.ids.txt` | 160 | Gemma 4 accuracy: the first 160 tokens of `gemma-code-11k` |
| `gemma-raw120.ids.txt` | 120 | Gemma 4 accuracy: 120 tokens of raw text (llama.cpp documentation) |

Accuracy against the f32 CPU reference (the reference dumps are 120–170 MB, so they are not kept here; make them
once with `tools\ref_forward`):

```powershell
$g = (Get-ChildItem "$env:USERPROFILE\.cache\huggingface\hub" -Recurse -Filter "gemma-4-12b-it-qat-q4_0.gguf" | Select-Object -First 1).FullName
$ids = (Get-Content bench\prompts\gemma-code160.ids.txt -Raw).Trim()
$env:REF_ALL = "1"; .\bin\ref_forward.exe $g $ids ref_g12_code160.bin; $env:REF_ALL = $null
.\bin\runz.exe $g --tokens-file bench\prompts\gemma-code160.ids.txt --kl-ref ref_g12_code160.bin --n 1        # prompt path
.\bin\runz.exe $g --tokens-file bench\prompts\gemma-code160.ids.txt --kl-ref ref_g12_code160.bin --kl-decode  # generation path
```

Results on 10 October 2026: prompt path 0.018, generation path 0.013 (code); generation path 0.029 (raw text).
