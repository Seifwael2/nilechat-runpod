import json
import re
import shutil
from pathlib import Path

MODEL_DIR = Path('/workspace/models/Nile-Chat-12B')
config_path = MODEL_DIR / 'config.json'
if not config_path.exists():
    raise SystemExit('Missing /workspace/models/Nile-Chat-12B/config.json. Download NileChat to the attached volume first.')

cfg = json.loads(config_path.read_text())
backup = MODEL_DIR / 'config.json.original_backup'
if not backup.exists():
    backup.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

if isinstance(cfg.get('text_config'), dict) and 'num_attention_heads' not in cfg['text_config']:
    del cfg['text_config']

cfg['architectures'] = ['Gemma3ForCausalLM']
cfg['model_type'] = 'gemma3_text'
cfg['torch_dtype'] = 'bfloat16'
cfg['rope_local_base_freq'] = 10000.0
cfg['rope_theta'] = 1000000.0
cfg['sliding_window'] = 1024
cfg['sliding_window_pattern'] = 6
cfg['rope_scaling'] = {'rope_type': 'linear', 'type': 'linear', 'factor': 8.0}
config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
print('Model config patched.')

try:
    import vllm.transformers_utils.config as cfg_mod
    import vllm.transformers_utils.tokenizer as tok_mod
    import vllm.model_executor.models.gemma3 as gemma3_mod
except Exception as e:
    raise SystemExit('Could not import vLLM. Use this custom image or install the pinned stack first.\nImport error: '+repr(e))

# Patch rope parser
vllm_cfg_path = Path(cfg_mod.__file__)
text = vllm_cfg_path.read_text()
marker = 'NILECHAT_GEMMA3_ROPE_FINAL_FIX'
if marker not in text:
    backup_path = vllm_cfg_path.with_suffix(vllm_cfg_path.suffix + '.backup_nilechat')
    if not backup_path.exists(): shutil.copy2(vllm_cfg_path, backup_path)
    idx = text.find('def patch_rope_scaling_dict')
    if idx != -1:
        header_end = text.find('\n', idx); body_start = header_end + 1
        m = re.match(r'([ \t]+)', text[body_start:]); indent = m.group(1) if m else '    '
        hotfix = (
            f'{indent}# {marker}\n'
            f'{indent}if isinstance(rope_scaling, dict) and "rope_type" not in rope_scaling:\n'
            f'{indent}    full_attention = rope_scaling.get("full_attention")\n'
            f'{indent}    if isinstance(full_attention, dict):\n'
            f'{indent}        rope_scaling.clear()\n'
            f'{indent}        rope_scaling.update(full_attention)\n'
            f'{indent}if isinstance(rope_scaling, dict) and "rope_type" not in rope_scaling:\n'
            f'{indent}    rope_scaling["rope_type"] = rope_scaling.get("type", "linear")\n'
            f'{indent}if isinstance(rope_scaling, dict) and "type" not in rope_scaling and "rope_type" in rope_scaling:\n'
            f'{indent}    rope_scaling["type"] = rope_scaling["rope_type"]\n'
        )
        text = text[:body_start] + hotfix + text[body_start:]
        vllm_cfg_path.write_text(text)
        print('vLLM rope parser patched.')

# Patch tokenizer compatibility
tok_path = Path(tok_mod.__file__)
text = tok_path.read_text()
if 'NILECHAT_GEMMA_TOKENIZER_FIX' not in text and 'tokenizer.all_special_tokens_extended)' in text:
    backup_path = tok_path.with_suffix(tok_path.suffix + '.backup_nilechat')
    if not backup_path.exists(): shutil.copy2(tok_path, backup_path)
    text = text.replace('tokenizer.all_special_tokens_extended)', "getattr(tokenizer, 'all_special_tokens_extended', tokenizer.all_special_tokens))  # NILECHAT_GEMMA_TOKENIZER_FIX")
    tok_path.write_text(text)
    print('vLLM tokenizer patched.')

# Patch Gemma3 missing rope fields
gemma_path = Path(gemma3_mod.__file__)
text = gemma_path.read_text()
backup_path = gemma_path.with_suffix(gemma_path.suffix + '.backup_nilechat')
if not backup_path.exists(): shutil.copy2(gemma_path, backup_path)
new_text = text.replace('self.rope_theta = config.rope_local_base_freq', 'self.rope_theta = getattr(config, "rope_local_base_freq", getattr(config, "rope_theta", 10000.0))  # NILECHAT_ROPE_LOCAL_BASE_FREQ_FIX')
new_text = new_text.replace('self.rope_theta = config.rope_theta', 'self.rope_theta = getattr(config, "rope_theta", 1000000.0)  # NILECHAT_ROPE_THETA_FIX')
if new_text != text:
    gemma_path.write_text(new_text)
    print('vLLM gemma3 patched.')
print('DONE PATCHING.')
