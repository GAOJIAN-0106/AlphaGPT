"""
Formula migration utility for vocabulary changes.

Migrates RPN formulas from old vocabulary (6 features + 12 ops = 18 tokens)
to new vocabulary (12 features + 12 ops = 24 tokens).

Old mapping: tokens 0-5 = features, tokens 6-17 = ops
New mapping: tokens 0-11 = features, tokens 12-23 = ops

Migration rule: operator tokens shift by +6 (the number of new features added).
Feature tokens 0-5 are unchanged.
"""

import json

OLD_N_FEATURES = 6
NEW_N_FEATURES = 12
FEATURE_SHIFT = NEW_N_FEATURES - OLD_N_FEATURES  # 6


def migrate_token(old_token):
    """Migrate a single token from old to new vocabulary."""
    if old_token < OLD_N_FEATURES:
        return old_token  # feature token, unchanged
    return old_token + FEATURE_SHIFT  # operator token, shift forward


def migrate_formula(old_formula):
    """Migrate a full RPN formula from old to new vocabulary."""
    return [migrate_token(t) for t in old_formula]


def migrate_ensemble_file(input_path, output_path=None):
    """
    Migrate a best_ensemble.json file from old to new vocabulary.

    Args:
        input_path: path to old ensemble JSON
        output_path: path for migrated JSON (default: overwrite input)

    Returns:
        dict with migrated ensemble data
    """
    if output_path is None:
        output_path = input_path

    with open(input_path) as f:
        data = json.load(f)

    if 'ensemble' in data and 'formulas' in data['ensemble']:
        old_formulas = data['ensemble']['formulas']
        data['ensemble']['formulas'] = [
            migrate_formula(f) for f in old_formulas
        ]
        data['ensemble']['_migrated_from'] = f'{OLD_N_FEATURES}feat_to_{NEW_N_FEATURES}feat'

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    return data


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print(f"Usage: python -m model_core.migrate <input.json> [output.json]")
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    result = migrate_ensemble_file(inp, out)
    n = len(result.get('ensemble', {}).get('formulas', []))
    print(f"Migrated {n} formulas from {OLD_N_FEATURES}-feature to {NEW_N_FEATURES}-feature vocabulary")
