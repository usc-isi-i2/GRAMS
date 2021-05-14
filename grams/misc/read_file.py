from pathlib import Path
from typing import List, Optional, Union

import bz2
import chardet
import glob
import gzip
from tqdm.auto import tqdm


class ChainableIterator:

    def __init__(self, files: List[str]):
        self.files = files
        self.counter = 0
        self.fns = []
        self.reader = None

    def map(self, fn):
        self.fns.append(fn)
        return self

    def next(self, n: int):
        """Get one element at index n
        
        Parameters
        ----------
        n : int
            the index of the element
        
        Returns
        -------
        Any
            (n+1)th item on the chain
        """
        it = iter(self)
        for i in range(n+1):
            x = next(it)
        return x

    def __del__(self):
        if self.reader is not None:
            self.reader.close()

    def __iter__(self):
        self.counter = 0
        self.reader = get_open_fn(self.files[self.counter])(self.files[self.counter], "rb")
        return self

    def __next__(self):
        try:
            result = next(self.reader)
            for fn in self.fns:
                result = fn(result)
            return result
        except StopIteration:
            self.counter += 1
            self.reader.close()
            if self.counter >= len(self.files):
                raise
            else:
                self.reader = get_open_fn(self.files[self.counter])(self.files[self.counter], "rb")
                result = next(self.reader)
                for fn in self.fns:
                    result = fn(result)
                return result


def read_text(fpath: Union[str, Path]) -> str:
    """
    Read content of a file as text. Use this function to load data in files that may not be in UTF-8
    """
    with open(str(fpath), 'rb') as f:
        content = f.read()

    try:
        content = content.decode("utf-8")
    except UnicodeDecodeError:
        encoding = chardet.detect(content)['encoding']
        content = content.decode(encoding)
    return content


def read_lines(fpath: str, max_lines: Optional[int]=None, report: bool=False) -> str:
    """
    Read lines from file with maximum number of lines
    """
    with get_open_fn(fpath)(fpath, "r") as f:
        if report:
            if max_lines is None:
                lines = []
                for line in tqdm(f):
                    lines.append(line)
            else:
                lines = []
                for i in tqdm(range(max_lines)):
                    lines.append(next(f))
        else:
            if max_lines is None:
                lines = list(f)
            else:
                lines = [next(f) for i in range(max_lines)]

    return lines


def get_lines_iter(inglob: str) -> ChainableIterator:
    infiles = sorted(glob.glob(inglob))
    assert len(infiles) > 0, "No matched files"
    return ChainableIterator(infiles)


def get_open_fn(infile: Union[str, Path]):
    """Get the correct open function for the input file based on its extension. Supported bzip2, gz
    
    Parameters
    ----------
    infile : Union[str, Path]
        the file we wish to open
    
    Returns
    -------
    Callable
        the open function that can use to open the file
    
    Raises
    ------
    ValueError
        when encounter unknown extension
    """
    infile = str(infile)

    if infile.endswith(".bz2"):
        return bz2.open
    elif infile.endswith(".gz"):
        return gzip.open
    else:
        return open
