"""
Catalog utilities for the LSST pipeline project.

This module provides the Catalog class for managing Parquet files, including creating,
appending, and combining data. It supports efficient handling of large datasets with
schema validation and is optimized for memory efficiency through batch processing and
temporary file management. The module is designed for use in microlensing simulations,
such as those involving LSST DP0 or DP1 data, to store event and photometry data.

Classes
-------
Catalog : Manages Parquet file operations with schema enforcement and batch processing.
"""

# Standard library imports
import logging
import os
import tempfile
from glob import glob
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class Catalog:
    """
    Class for managing Parquet files with schema enforcement and batch processing.

    Provides methods to create, append, combine, and query Parquet files, optimized for
    memory efficiency through chunked processing and temporary file management. Supports
    schema validation and is designed for use in parallel processing pipelines.

    Attributes
    ----------
    file_path : str
        Path to the Parquet file.
    schema : pyarrow.Schema, optional
        Schema for the Parquet file.
    logger : logging.Logger
        Logger for debugging and logging.
    """

    def __init__(self, file_path: str,  schema: Optional[Dict[str, pa.DataType]] = None) -> None:
        """
        Initialize a Catalog instance.

        Parameters
        ----------
        file_path : str
            Path to the Parquet file (absolute or relative to main_path).
        schema : Dict[str, pyarrow.DataType], optional
            Schema for the Parquet file as a dictionary (default: None).
        """
        self.file_path = file_path
        if schema is None:
            self.schema = None
        elif isinstance(schema, pa.Schema):
            self.schema = schema 
        elif isinstance(schema, dict):
            self.schema = pa.schema(list(schema.items()))
        else:
            raise TypeError(f"Unsupported schema type: {type(schema)}")
        self.logger = logging.getLogger(__name__)
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def create(self, schema: Optional[Dict[str, pa.DataType]] = None, overwrite: bool = False) -> None:
        """
        Create an empty Parquet file with the specified schema.

        Parameters
        ----------
        schema : Dict[str, pyarrow.DataType], optional
            Schema for the Parquet file as a dictionary; uses self.schema if None.
        overwrite : bool, optional
            Overwrite the file if it exists (default: False).

        Raises
        ------
        ValueError
            If schema is not provided and self.schema is None.
        OSError
            If file creation fails.
        """
        if schema:
            self.schema = pa.schema(list(schema.items()))
        if not self.schema:
            raise ValueError("Schema must be provided to create a Parquet file.")

        if os.path.exists(self.file_path) and not overwrite:
            self.logger.info(f"File {self.file_path} already exists and overwrite is False.")
            return

        try:
            empty_df = pd.DataFrame({field.name: [] for field in self.schema})
            table = pa.Table.from_pandas(empty_df, schema=self.schema, preserve_index=False)
            pq.write_table(table, self.file_path, compression="snappy")
            self.logger.info(f"Created empty Parquet file at {self.file_path}")
        except Exception as e:
            self.logger.error(f"Error creating Parquet file {self.file_path}: {str(e)}")
            raise OSError(f"Error creating Parquet file {self.file_path}: {str(e)}")

    def add_rows(self, rows: List[Dict[str, Any]], mode: str = "append", schema: Optional[Union[Dict[str, pa.DataType], pa.Schema]] = None) -> None:
        """
        Add rows to the Parquet file with thread-safe temporary file handling.

        Uses a lock file to prevent concurrent writes in parallel processing, ensuring
        safe appending to the Parquet file.

        Parameters
        ----------
        rows : List[Dict[str, Any]]
            List of dictionaries containing row data.
        mode : str, optional
            Mode for adding rows: 'append' or 'overwrite' (default: 'append').
        schema : Union[Dict[str, pyarrow.DataType], pyarrow.Schema], optional
            Schema for the Parquet file; uses self.schema if None.

        Raises
        ------
        ValueError
            If mode is invalid, rows are empty, or schema is missing.
        OSError
            If file writing or lock acquisition fails.
        """
        if not rows:
            self.logger.info("No rows to add.")
            return

        if mode not in ["append", "overwrite"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'append' or 'overwrite'.")

        # schema = schema if isinstance(schema, pa.Schema) else pa.schema(list(schema.items())) if schema else self.schema
        # if not schema:
        #     raise ValueError("Schema must be provided to add rows.")

        df = pd.DataFrame(rows)
        table = pa.Table.from_pandas(df, schema=self.schema, preserve_index=False)

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        # Use a lock file to prevent concurrent writes
        lock_file = f"{self.file_path}.lock"
        temp_file = f"{self.file_path}.tmp"
        try:
            # Attempt to acquire lock
            for _ in range(10):  # Retry up to 10 times
                try:
                    with open(lock_file, "x") as f:
                        f.write(str(os.getpid()))
                        break
                except FileExistsError:
                    import time
                    time.sleep(0.1)  # Wait briefly before retrying
            else:
                raise OSError(f"Could not acquire lock for {self.file_path} after multiple attempts.")

            if mode == "append" and os.path.exists(self.file_path):
                existing_table = pq.read_table(self.file_path)
                table = pa.concat_tables([existing_table, table])
            pq.write_table(table, temp_file, compression="snappy")
            os.replace(temp_file, self.file_path)
            self.logger.info(f"Added {len(rows)} rows to {self.file_path} (mode={mode})")
        except Exception as e:
            self.logger.error(f"Error adding rows to {self.file_path}: {str(e)}")
            raise OSError(f"Error adding rows to {self.file_path}: {str(e)}")
        finally:
            try:
                os.remove(lock_file)
            except OSError:
                pass

    def combine_parquet_files(
        self,
        temp_dir: str,
        schema: Optional[Union[List[Tuple[str, pa.DataType]], Dict[str, pa.DataType]]],
        batch_size: int = 1000,
        columns: Optional[List[str]] = None,
        cleanup: bool = True
    ) -> None:
        """
        Combine temporary Parquet files into a single final Parquet file, handling column order differences.

        Args:
            temp_dir (str): Directory containing temporary Parquet files.
            schema (Optional[Union[List[Tuple[str, pa.DataType]], Dict[str, pa.DataType]]]): Schema for the output file.
                Required if final_path doesn't exist or is invalid.
            batch_size (int): Number of rows to process at a time (default: 1000).
            columns (Optional[List[str]]): Columns to read from temporary files (default: None, read all).
            cleanup (bool): Delete temporary files after combining (default: True).

        Raises
        ------
        ValueError
            If no valid temporary files are found or schema is required but not provided.
        """
        import glob
        import pyarrow as pa
        import pyarrow.parquet as pq
        final_path = self.file_path
        to_combine_name = final_path.split("/")[-1].split(".")[0].split("_")[0]
        temp_pattern = f'temp_{to_combine_name}_*.parquet'
        temp_files = glob.glob(os.path.join(temp_dir, temp_pattern))
        if not temp_files:
            return

        # Validate schema
        if schema is not None:
            if isinstance(schema, dict):
                schema = [(name, dtype) for name, dtype in schema.items()]
            schema = pa.schema(schema)
        elif self.schema is None and not os.path.exists(final_path):
            raise ValueError(f"No schema provided and {final_path} does not exist.")

        # Filter valid temporary files
        valid_temp_files = []
        for temp_file in temp_files:
            try:
                file_size = os.path.getsize(temp_file)
                if file_size < 8:
                    continue
                parquet_file = pq.ParquetFile(temp_file)
                valid_temp_files.append(temp_file)
            except Exception as e:
                continue

        if not valid_temp_files:
            return

        # Check if final_path is valid
        final_exists = os.path.exists(final_path)
        final_valid = final_exists and os.path.getsize(final_path) >= 8
        if final_exists and not final_valid:
            os.remove(final_path)
            final_valid = False

        # Combine files
        target_schema = schema or self.schema
        if not target_schema:
            try:
                target_schema = pq.ParquetFile(valid_temp_files[0]).schema_arrow
            except Exception as e:
                raise ValueError(f"Cannot infer schema from {valid_temp_files[0]}: {str(e)}")

        with pq.ParquetWriter(final_path, target_schema, compression='snappy', write_statistics=True) as writer:
            # Write existing data if appending and final file is valid
            if final_valid:
                try:
                    parquet_file = pq.ParquetFile(final_path)
                    for i in range(parquet_file.num_row_groups):
                        table = parquet_file.read_row_group(i, columns=columns)
                        if table.schema != target_schema:
                            # Check if schema difference is only column order
                            table_schema_dict = {field.name: field.type for field in table.schema}
                            target_schema_dict = {field.name: field.type for field in target_schema}
                            if set(table_schema_dict.keys()) == set(target_schema_dict.keys()) and all(
                                table_schema_dict[k] == target_schema_dict[k] for k in table_schema_dict
                            ):
                                table = table.select(target_schema.names)
                            else:
                                table = table.cast(target_schema, safe=False)
                        writer.write_table(table)
                except Exception as e:
                    final_valid = False

            # Combine temporary files
            for temp_file in valid_temp_files:
                try:
                    parquet_file = pq.ParquetFile(temp_file)
                    for i in range(parquet_file.num_row_groups):
                        table = parquet_file.read_row_group(i, columns=columns)
                        # Select only columns present in target_schema
                        available_columns = [col for col in table.column_names if col in target_schema.names]
                        if not available_columns:
                            continue
                        table = table.select(available_columns)
                        # Check if schema difference is only column order
                        table_schema_dict = {field.name: field.type for field in table.schema}
                        target_schema_dict = {field.name: field.type for field in target_schema}
                        if set(table_schema_dict.keys()) == set(target_schema_dict.keys()) and all(
                            table_schema_dict[k] == target_schema_dict[k] for k in table_schema_dict
                        ):
                            table = table.select(target_schema.names)
                        else:
                            table = table.cast(target_schema, safe=False)
                        writer.write_table(table)
                except Exception as e:
                    continue

            if cleanup:
                for temp_file in valid_temp_files:
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        print(e)

    def get_max_value(self, column: str) -> Union[int, float, None]:
        """
        Retrieve the maximum value of a specified column.

        Parameters
        ----------
        column : str
            Column name to query.

        Returns
        -------
        Union[int, float, None]
            Maximum value in the column; None if file is empty or column is missing.
        """
        if not os.path.exists(self.file_path):
            self.logger.warning(f"File {self.file_path} does not exist.")
            return None

        try:
            table = pq.read_table(self.file_path, columns=[column])
            if table.num_rows == 0:
                return None
            return table[column].to_pandas().max()
        except Exception as e:
            self.logger.error(f"Error getting max value from {self.file_path} for column {column}: {str(e)}")
            return None

    def get_schema(self) -> Optional[pa.Schema]:
        """
        Retrieve the schema of the Parquet file.

        Returns
        -------
        pyarrow.Schema, optional
            Schema of the Parquet file; None if file does not exist.
        """
        if not os.path.exists(self.file_path):
            self.logger.warning(f"File {self.file_path} does not exist.")
            return None

        try:
            return pq.read_schema(self.file_path)
        except Exception as e:
            self.logger.error(f"Error reading schema from {self.file_path}: {str(e)}")
            return None