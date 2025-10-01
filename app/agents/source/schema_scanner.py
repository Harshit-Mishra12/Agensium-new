import pandas as pd
import io
import sqlparse
import warnings
from fastapi import HTTPException

def _profile_dataframe(df: pd.DataFrame):
    """
    Helper function to generate a schema summary for a single DataFrame.
    """
    # Handle empty dataframes
    if df.empty:
        return {
            "summary_table": [],
            "total_rows": 0,
            "message": "Sheet is empty."
        }
        
    schema_summary = []

    for col in df.columns:
        # 1. Calculate Null Percentage
        null_count = df[col].isnull().sum()
        total_count = len(df)
        null_percentage = (null_count / total_count) * 100 if total_count > 0 else 0

        # 2. Get Distinct Count
        distinct_count = df[col].nunique()

        # 3. Get Top Values (we'll take the top 3)
        top_values = df[col].value_counts().nlargest(3).index.tolist()
        top_values_str = ', '.join(map(str, top_values))

        # 4. Get Data Type (with date detection)
        col_type = str(df[col].dtype)
        if 'object' in col_type:
            # Try to convert to datetime to detect date columns stored as text
            try:
                # Suppress the UserWarning pandas throws when it cannot infer a consistent date format.
                # This is expected behavior for this agent, as it's trying to detect dates.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    pd.to_datetime(df[col].dropna().iloc[:100], errors='raise')
                data_type = 'Date'
            except (ValueError, TypeError, IndexError):
                data_type = 'Text'
        elif 'int' in col_type:
            data_type = 'Integer'
        elif 'float' in col_type:
            data_type = 'Float'
        elif 'datetime' in col_type:
            data_type = 'Date'
        else:
            data_type = col_type

        schema_summary.append({
            "field": col,
            "data_type": data_type,
            "null": f"{null_percentage:.1f}%",
            "distinct_count": distinct_count,
            "top_values": top_values_str + ('…' if distinct_count > 3 else '')
        })

    return {
        "summary_table": schema_summary,
        "total_rows": len(df)
    }

def _profile_sql_schema(sql_script: str):
    """
    Helper function to parse a SQL script and extract schema from CREATE TABLE statements.
    """
    parsed = sqlparse.parse(sql_script)
    all_tables_summary = {}

    for stmt in parsed:
        if stmt.get_type() == 'CREATE':
            # Find the table name
            table_name = None
            for token in stmt.tokens:
                if isinstance(token, sqlparse.sql.Identifier):
                    table_name = token.get_name()
                    break
            
            if not table_name:
                continue

            schema_summary = []
            # Find columns within the parenthesis
            paren = next((t for t in stmt.tokens if isinstance(t, sqlparse.sql.Parenthesis)), None)
            if paren:
                # Simple parsing of columns, assumes "column_name data_type," format
                columns_str = paren.value.strip('()')
                for col_def in columns_str.split(','):
                    col_def = col_def.strip()
                    if not col_def or col_def.lower().startswith(('primary key', 'constraint', 'foreign key')):
                        continue
                    
                    parts = col_def.split()
                    field_name = parts[0].strip('`"')
                    data_type = parts[1] if len(parts) > 1 else 'UNKNOWN'

                    schema_summary.append({
                        "field": field_name,
                        "data_type": data_type,
                        "null": "N/A",
                        "distinct_count": "N/A",
                        "top_values": "N/A"
                    })
            
            all_tables_summary[table_name] = {
                "summary_table": schema_summary,
                "total_rows": "N/A",
                "message": "Schema extracted from SQL DDL. No data to profile."
            }
            
    return all_tables_summary


def scan_schema(file_contents: bytes, filename: str):
    """
    Scans a CSV, Excel, JSON, or SQL file and returns a schema profile.
    Handles multiple sheets in Excel files and multiple tables in SQL files.
    """
    file_extension = filename.split('.')[-1].lower()
    all_sheets_summary = {}

    try:
        if file_extension == 'csv':
            df = pd.read_csv(io.BytesIO(file_contents))
            sheet_name = filename.rsplit('.', 1)[0]
            all_sheets_summary[sheet_name] = _profile_dataframe(df)

        elif file_extension in ['xlsx', 'xls']:
            xls_sheets = pd.read_excel(io.BytesIO(file_contents), sheet_name=None)
            if not xls_sheets:
                raise ValueError("The Excel file appears to be empty or has no sheets.")
            for sheet_name, df in xls_sheets.items():
                all_sheets_summary[sheet_name] = _profile_dataframe(df)
        
        elif file_extension == 'json':
            df = pd.read_json(io.BytesIO(file_contents))
            sheet_name = filename.rsplit('.', 1)[0]
            all_sheets_summary[sheet_name] = _profile_dataframe(df)

        elif file_extension == 'sql':
            sql_script = file_contents.decode('utf-8')
            sql_summary = _profile_sql_schema(sql_script)
            if not sql_summary:
                raise ValueError("No valid 'CREATE TABLE' statements found in the SQL file.")
            all_sheets_summary.update(sql_summary)
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}")

        return all_sheets_summary

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process the file '{filename}'. Error: {str(e)}")

