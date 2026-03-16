# B4. Templates: coding/conversational/research/custom

Templates provide repeatable task shapes and default graders.

## Dispatch model

`EvalTemplate.from_yaml(...)` in `ash_hawk/templates/__init__.py` reads `template_type` and dispatches to:
- `coding`
- `conversational`
- `research`
- `custom`

## Why use templates

- Enforce consistent fields per task family
- Inherit sensible default graders/policies
- Reduce boilerplate when scaling task count

## Notable implementation references

- Base contracts: `ash_hawk/templates/__init__.py`
- Coding template: `ash_hawk/templates/coding.py`
- Custom template/builder: `ash_hawk/templates/custom.py`

## Skill links

- If you need strict custom shape control, start with `template_type: custom` and move to [B3](05_graders_and_weights.md) for grader tuning.
