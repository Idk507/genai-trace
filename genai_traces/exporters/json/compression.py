"""
Compression utilities for GenAI-Traces exporters.

Provides gzip and zstd compression support.
"""

import gzip
import io
from pathlib import Path
from typing import Union, Optional
from enum import Enum


class CompressionType(Enum):
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    ZSTD = "zstd"


def compress_data(
    data: Union[str, bytes],
    compression: CompressionType = CompressionType.GZIP,
    level: int = 6,
) -> bytes:
    """
    Compress data using the specified algorithm.
    
    Args:
        data: Data to compress (string or bytes)
        compression: Compression algorithm
        level: Compression level (1-9 for gzip, 1-22 for zstd)
        
    Returns:
        Compressed bytes
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    
    if compression == CompressionType.NONE:
        return data
    
    if compression == CompressionType.GZIP:
        return gzip.compress(data, compresslevel=level)
    
    if compression == CompressionType.ZSTD:
        try:
            import zstandard as zstd
            cctx = zstd.ZstdCompressor(level=level)
            return cctx.compress(data)
        except ImportError:
            return gzip.compress(data, compresslevel=level)
    
    return data


def decompress_data(
    data: bytes,
    compression: CompressionType = CompressionType.GZIP,
) -> bytes:
    """
    Decompress data using the specified algorithm.
    
    Args:
        data: Compressed data
        compression: Compression algorithm used
        
    Returns:
        Decompressed bytes
    """
    if compression == CompressionType.NONE:
        return data
    
    if compression == CompressionType.GZIP:
        return gzip.decompress(data)
    
    if compression == CompressionType.ZSTD:
        try:
            import zstandard as zstd
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data)
        except ImportError:
            return gzip.decompress(data)
    
    return data


def compress_file(
    input_path: str,
    output_path: Optional[str] = None,
    compression: CompressionType = CompressionType.GZIP,
    level: int = 6,
    delete_original: bool = False,
) -> str:
    """
    Compress a file.
    
    Args:
        input_path: Path to input file
        output_path: Path for output (default: input + extension)
        compression: Compression algorithm
        level: Compression level
        delete_original: Whether to delete the original file
        
    Returns:
        Path to compressed file
    """
    input_path = Path(input_path)
    
    if output_path is None:
        ext = ".gz" if compression == CompressionType.GZIP else ".zst"
        output_path = str(input_path) + ext
    
    with open(input_path, "rb") as f_in:
        data = f_in.read()
    
    compressed = compress_data(data, compression, level)
    
    with open(output_path, "wb") as f_out:
        f_out.write(compressed)
    
    if delete_original:
        input_path.unlink()
    
    return output_path


def decompress_file(
    input_path: str,
    output_path: Optional[str] = None,
    compression: Optional[CompressionType] = None,
    delete_original: bool = False,
) -> str:
    """
    Decompress a file.
    
    Args:
        input_path: Path to compressed file
        output_path: Path for output (default: input without extension)
        compression: Compression algorithm (auto-detected if None)
        delete_original: Whether to delete the compressed file
        
    Returns:
        Path to decompressed file
    """
    input_path = Path(input_path)
    
    if compression is None:
        if str(input_path).endswith(".gz"):
            compression = CompressionType.GZIP
        elif str(input_path).endswith(".zst"):
            compression = CompressionType.ZSTD
        else:
            compression = CompressionType.NONE
    
    if output_path is None:
        if compression == CompressionType.GZIP:
            output_path = str(input_path)[:-3]
        elif compression == CompressionType.ZSTD:
            output_path = str(input_path)[:-4]
        else:
            output_path = str(input_path) + ".decompressed"
    
    with open(input_path, "rb") as f_in:
        data = f_in.read()
    
    decompressed = decompress_data(data, compression)
    
    with open(output_path, "wb") as f_out:
        f_out.write(decompressed)
    
    if delete_original:
        input_path.unlink()
    
    return output_path


class CompressedWriter:
    """
    Context manager for writing compressed data.
    
    Usage:
        with CompressedWriter("traces.jsonl.gz") as writer:
            writer.write('{"trace": "data"}\\n')
    """
    
    def __init__(
        self,
        path: str,
        compression: CompressionType = CompressionType.GZIP,
        level: int = 6,
    ):
        self._path = path
        self._compression = compression
        self._level = level
        self._file = None
        self._writer = None
    
    def __enter__(self):
        if self._compression == CompressionType.GZIP:
            self._file = gzip.open(self._path, "wt", compresslevel=self._level)
        elif self._compression == CompressionType.ZSTD:
            try:
                import zstandard as zstd
                self._file = open(self._path, "wb")
                cctx = zstd.ZstdCompressor(level=self._level)
                self._writer = cctx.stream_writer(self._file)
            except ImportError:
                self._file = gzip.open(self._path, "wt", compresslevel=self._level)
        else:
            self._file = open(self._path, "w")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._writer:
            self._writer.close()
        if self._file:
            self._file.close()
        return False
    
    def write(self, data: str) -> None:
        """Write data to the compressed file."""
        if self._writer:
            self._writer.write(data.encode("utf-8"))
        else:
            self._file.write(data)


class CompressedReader:
    """
    Context manager for reading compressed data.
    
    Usage:
        with CompressedReader("traces.jsonl.gz") as reader:
            for line in reader:
                print(line)
    """
    
    def __init__(
        self,
        path: str,
        compression: Optional[CompressionType] = None,
    ):
        self._path = path
        
        if compression is None:
            if path.endswith(".gz"):
                compression = CompressionType.GZIP
            elif path.endswith(".zst"):
                compression = CompressionType.ZSTD
            else:
                compression = CompressionType.NONE
        
        self._compression = compression
        self._file = None
        self._reader = None
    
    def __enter__(self):
        if self._compression == CompressionType.GZIP:
            self._file = gzip.open(self._path, "rt")
        elif self._compression == CompressionType.ZSTD:
            try:
                import zstandard as zstd
                self._file = open(self._path, "rb")
                dctx = zstd.ZstdDecompressor()
                self._reader = io.TextIOWrapper(
                    dctx.stream_reader(self._file),
                    encoding="utf-8",
                )
            except ImportError:
                self._file = gzip.open(self._path, "rt")
        else:
            self._file = open(self._path, "r")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._reader:
            self._reader.close()
        if self._file:
            self._file.close()
        return False
    
    def __iter__(self):
        return iter(self._reader or self._file)
    
    def read(self) -> str:
        """Read all data from the compressed file."""
        return (self._reader or self._file).read()
    
    def readline(self) -> str:
        """Read a single line."""
        return (self._reader or self._file).readline()
