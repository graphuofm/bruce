Reference copies (read-only, for provenance), all DuckDB v1.5.6:
- duckdb.h, duckdb_extension.h  from https://github.com/duckdb/duckdb/releases/download/v1.5.6/libduckdb-linux-amd64.zip
- aggregate_function-c.cpp       https://raw.githubusercontent.com/duckdb/duckdb/v1.5.6/src/main/capi/aggregate_function-c.cpp
                                 (shows the C-API wrapper flattens inputs/states before calling update/combine/finalize)
- extension_load.cpp             https://raw.githubusercontent.com/duckdb/duckdb/v1.5.6/src/main/extension/extension_load.cpp
                                 (metadata footer parsing, C_STRUCT version check, <name>_init_c_api lookup)
- ../append_extension_metadata.py  https://raw.githubusercontent.com/duckdb/extension-ci-tools/main/scripts/append_extension_metadata.py
