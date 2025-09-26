import yaml
import pandas as pd
import re
import sys

def load_schema(schema_path):
    with open(schema_path, 'r') as f:
        return yaml.safe_load(f)

def validate_column(col_series, field_def):
    errors = []
    for idx, val in enumerate(col_series):
        # type check
        if field_def['type'] == 'integer':
            if not isinstance(val, (int, float)):
                errors.append((idx, val, 'type'))
            else:
                if 'min' in field_def and val < field_def['min']:
                    errors.append((idx, val, 'min'))
                if 'max' in field_def and val > field_def['max']:
                    errors.append((idx, val, 'max'))
        elif field_def['type'] == 'string':
            if not isinstance(val, str):
                errors.append((idx, val, 'type'))
            if 'allowed' in field_def and val not in field_def['allowed']:
                errors.append((idx, val, 'allowed'))
            if 'regex' in field_def and not re.match(field_def['regex'], str(val)):
                errors.append((idx, val, 'regex'))
    return errors

def validate_data(csv_path, schema_path='data/schema.yml'):
    schema = load_schema(schema_path)
    df = pd.read_csv(csv_path)
    all_errors = {}
    for field in schema['fields']:
        name = field['name']
        if name not in df.columns:
            all_errors[name] = 'missing column'
            continue
        errors = validate_column(df[name], field)
        if errors:
            all_errors[name] = errors
    return all_errors

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_schema.py path/to/data.csv")
        sys.exit(1)
    csv_path = sys.argv[1]
    errors = validate_data(csv_path)
    if errors:
        print("Validation errors found:")
        for col, errs in errors.items():
            print(col, errs)
    else:
        print("All good!")
