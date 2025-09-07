"""
Parallel processing utilities for the LSST pipeline project.

This module provides the ParallelProcessor class for parallelizing data processing
tasks on large Parquet datasets with progress tracking, checkpointing, and temporary
file management. It integrates with the catalogs_utils.Catalog class for efficient
management of Parquet files, supporting appending results and combining temporary
files to optimize disk space. The design prioritizes memory efficiency by processing
data in chunks and time efficiency through multiprocessing.

The module is intended for use in scientific experiments, such as injecting sources
into LSST images (DP0 simulations or DP1 real data samples) and performing photometric
measurements, or simulating light curves with rubin_sim.

Classes
-------
ParallelProcessor : Manages parallel processing of Parquet datasets with checkpointing and file management.
"""

# Standard library imports
import logging
import os
from glob import glob
from typing import Callable, Dict, Any, Optional, List, Tuple

# Third-party imports
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool
from tqdm import tqdm

# Local imports
from catalogs_utils import Catalog


class ParallelProcessor:
    """
    Class for parallel processing of Parquet datasets with progress tracking and checkpointing.

    This class reads a Parquet file in chunks to minimize memory usage, processes rows in parallel
    using a provided function, and saves results to Parquet files. It supports checkpointing by
    saving partial results at intervals and combines temporary files when a maximum number is
    reached to optimize disk space. Integrates with catalogs_utils.Catalog for efficient Parquet
    handling.

    Attributes
    ----------
    input_path : str
        Path to the input Parquet file.
    output_dir : str
        Directory for output and temporary files.
    process_fn : Callable[[pd.Series, Dict[str, Any]], Dict[str, Any]]
        Function to process each row.
    config : Dict[str, Any]
        Configuration dictionary.
    chunk_size : int
        Number of rows per chunk.
    checkpoint_interval : int
        Interval for saving partial results.
    debug : bool
        Enable debug logging and progress bars.
    max_temp_files : int, optional
        Maximum number of temporary files before combining.
    photometry_catalog : catalogs_utils.Catalog
        Catalog for photometry data.
    events_catalog : catalogs_utils.Catalog
        Catalog for events data.
    results : List[Dict[str, Any]]
        Accumulated processing results.
    logger : logging.Logger
        Logger for debugging and logging.
    """

    def __init__(
        self,
        input_path: str,
        output_dir: str,
        process_fn: Callable[[pd.Series, Dict[str, Any]], Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1000,
        checkpoint_interval: int = 1000,
        debug: bool = False,
        max_temp_files: Optional[int] = None,
        photometry_catalog: Optional[Catalog] = None,
        events_catalog: Optional[Catalog] = None,
    ) -> None:
        """
        Initialize the parallel processor.

        Parameters
        ----------
        input_path : str
            Path to the input Parquet file containing data to process.
        output_dir : str
            Directory to store output results and temporary files.
        process_fn : Callable[[pd.Series, Dict[str, Any]], Dict[str, Any]]
            Function to process a single row, taking a pandas.Series and config dict, returning a result dict.
        config : dict, optional
            Additional configuration (e.g., from a YAML file).
        chunk_size : int, optional
            Number of rows per chunk for processing (default: 1000).
        checkpoint_interval : int, optional
            Interval (in rows) to save partial results (default: 1000).
        debug : bool, optional
            Enable debug logging and progress bars (default: False).
        max_temp_files : int, optional
            Maximum number of temporary files before combining (default: None).
        photometry_catalog : catalogs_utils.Catalog, optional
            Catalog for photometry Parquet file; creates new if None.
        events_catalog : catalogs_utils.Catalog, optional
            Catalog for events Parquet file; creates new if None.
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.process_fn = process_fn
        self.config = config or {}
        self.chunk_size = chunk_size
        self.checkpoint_interval = checkpoint_interval
        self.debug = debug
        self.max_temp_files = max_temp_files
        self.photometry_catalog = photometry_catalog or Catalog(
            os.path.join(self.output_dir, "photometry.parquet")
        )
        self.events_catalog = events_catalog or Catalog(
            os.path.join(self.output_dir, "data-events.parquet")
        )
        self.results = []
        self.logger = logging.getLogger(__name__)
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        # os.makedirs(self.output_dir, exist_ok=True)

    def save_results(self, chunk_idx: str, mode: str = "append") -> None:
        """
        Save accumulated results to results.parquet and combine temporary files if needed.

        Appends to the results file if it exists; otherwise, creates it with a schema. Combines
        temporary photometry and events files when max_temp_files is reached to free disk space.

        Parameters
        ----------
        chunk_idx : str
            Identifier for the current chunk (for logging).
        mode : str, optional
            Mode for saving results: 'append' or 'overwrite' (default: 'append').

        Raises
        ------
        ValueError
            If results are empty, mode is invalid, or saving/combining fails.
        """
        if not self.results:
            self.logger.info(f"No results to save for chunk {chunk_idx}")
            return

        if mode not in ["append", "overwrite"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'append' or 'overwrite'.")

        output_path = os.path.join(self.output_dir, f"results_{str(self.process_fn).split(" ")[1]}.parquet")
        columns = list(self.results[0].keys())
        results_schema = {
                columns[0]: pa.int32(),
                'status': pa.string(),
                'error': pa.string()
            }
        output_catalog = Catalog(output_path, schema=results_schema)
        

        try:
            if os.path.exists(output_path) and mode == "append":
                output_catalog.add_rows(self.results, mode="append")
            else:
                output_catalog.add_rows(
                    self.results, mode="overwrite", schema=results_schema
                )
            self.logger.info(
                f"Saved {len(self.results)} results to {output_path} (mode={mode}, chunk={chunk_idx})"
            )
            self.results = []
        except Exception as e:
            self.logger.error(f"Error saving results to {output_path}: {str(e)}")
            raise ValueError(f"Error saving results to {output_path}: {str(e)}")

        # Combine temporary files if max_temp_files is reached
        if self.max_temp_files is not None:
            temp_dir = self.config.get("temp_dir", self.output_dir)
            photometry_files = glob(os.path.join(temp_dir, "temp_photometry_*.parquet"))
            events_files = glob(os.path.join(temp_dir, "temp_events_*.parquet"))
            if (
                len(photometry_files) >= self.max_temp_files
                or len(events_files) >= self.max_temp_files
            ):
                self.logger.info(f"Reached {self.max_temp_files} temporary files. Combining...")
                try:
                    self.photometry_catalog.combine_parquet_files(
                        temp_dir=temp_dir,
                        schema=self.photometry_catalog.get_schema()
                        if os.path.exists(self.photometry_catalog.file_path)
                        else None,
                        columns=None,
                        batch_size=500,
                        cleanup=True,
                    )
                    self.logger.info(
                        f"Combined {len(photometry_files)} temporary photometry files to {self.photometry_catalog.file_path}"
                    )
                except Exception as e:
                    self.logger.error(f"Error combining photometry files: {str(e)}")
                    raise ValueError(f"Error combining photometry files: {str(e)}")

                try:
                    self.events_catalog.combine_parquet_files(
                        temp_dir=temp_dir,
                        schema=self.events_catalog.get_schema()
                        if os.path.exists(self.events_catalog.file_path)
                        else None,
                        columns=None,
                        batch_size=500,
                        cleanup=True,
                    )
                    self.logger.info(
                        f"Combined {len(events_files)} temporary events files to {self.events_catalog.file_path}"
                    )
                except Exception as e:
                    self.logger.error(f"Error combining events files: {str(e)}")
                    raise ValueError(f"Error combining events files: {str(e)}")

    def process_chunk(
        self, chunk: pd.DataFrame, chunk_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Process a chunk of data using the provided function with progress bar and checkpointing.

        Saves partial results at checkpoint intervals to avoid memory buildup during long chunks.

        Parameters
        ----------
        chunk : pd.DataFrame
            DataFrame chunk to process.
        chunk_idx : int
            Index of the chunk for identification.

        Returns
        -------
        List[Dict[str, Any]]
            List of results for each row in the chunk.
        """
        results = []
        for idx, row in tqdm(
            chunk.iterrows(),
            total=len(chunk),
            desc=f"Chunk {chunk_idx}",
            disable=not self.debug,
        ):
            result = self.process_fn(row, self.config)
            results.append(result)
            if (idx + 1) % self.checkpoint_interval == 0:
                self.results.extend(results)
                self.save_results(f"{chunk_idx}_{idx}", mode="append")
                results = []
        if results:
            self.results.extend(results)
            self.save_results(f"{chunk_idx}_end", mode="append")
        return self.results

    def _chunk_wrapper(self, args: Tuple[pd.DataFrame, int]) -> List[Dict[str, Any]]:
        """
        Wrapper for process_chunk to make it picklable for multiprocessing.

        Parameters
        ----------
        args : Tuple[pd.DataFrame, int]
            Tuple containing the chunk DataFrame and chunk index.

        Returns
        -------
        List[Dict[str, Any]]
            Results from process_chunk.
        """
        chunk, chunk_idx = args
        return self.process_chunk(chunk, chunk_idx)

    def process(
        self, num_pools: int = 4, start_idx: int = 0, end_idx: Optional[int] = None
    ) -> None:
        """
        Process the dataset in parallel using multiple processes.

        Reads the input Parquet file, applies index limits, splits into chunks, and processes
        them in parallel. To optimize memory, avoids loading unnecessary columns if possible,
        but currently loads full DataFrame (future improvement: use Parquet batches).

        Parameters
        ----------
        num_pools : int, optional
            Number of parallel processes to use (default: 4).
        start_idx : int, optional
            Starting row index for processing (default: 0).
        end_idx : int, optional
            Ending row index for processing (processes all if None).

        Examples
        --------
        >>> processor = ParallelProcessor(
        ...     input_path="input.parquet",
        ...     output_dir="output",
        ...     process_fn=my_function,
        ...     chunk_size=1000,
        ...     debug=True
        ... )
        >>> processor.process(num_pools=4)
        """
        try:
            if self.input_path.endswith(".parquet"):
                table = pq.read_table(self.input_path)
                df = table.to_pandas()
            elif self.input_path.endswith(".csv"):
                df = pd.read_csv(self.input_path)
        except FileNotFoundError:
            self.logger.error("Input Parquet file not found. Creating empty results.")
            return


        df = df.iloc[start_idx:end_idx] if end_idx is not None else df.iloc[start_idx:]
        chunks = [
            df.iloc[i : i + self.chunk_size] for i in range(0, len(df), self.chunk_size)
        ]

        with Pool(processes=num_pools) as pool:
            chunk_results = list(
                tqdm(
                    pool.imap(
                        self._chunk_wrapper,
                        [(chunk, idx) for idx, chunk in enumerate(chunks)],
                    ),
                    total=len(chunks),
                    desc="Running parallel processing",
                    disable=not self.debug,
                )
            )

        self.results = [item for sublist in chunk_results for item in sublist]
        self.save_results("final", mode="append")

    def get_results(self) -> pd.DataFrame:
        """
        Retrieve accumulated results from results.parquet.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all processing results. Returns empty DataFrame if file does not exist.
        """
        output_path = os.path.join(self.output_dir, f"results_{str(self.process_fn).split(" ")[1]}.parquet")
        if os.path.exists(output_path):
            return pq.read_table(output_path).to_pandas()
        return pd.DataFrame(columns=["event_id", "status", "error"])