Added in version 3.11.

**Source code:** [Lib/tomllib](https://github.com/python/cpython/tree/3.13/Lib/tomllib)

This module provides an interface for parsing TOML 1.0.0 (Tom’s Obvious Minimal Language, [https://toml.io](https://toml.io/en/)). This module does not support writing TOML.

See also: The [Tomli-W package](https://pypi.org/project/tomli-w/) is a TOML writer that can be used in conjunction with this module, providing a write API familiar to users of the standard library [`marshal`](https://docs.python.org/3/library/marshal.html#module-marshal "marshal: Convert Python objects to streams of bytes and back (with different constraints).") and [`pickle`](https://docs.python.org/3/library/pickle.html#module-pickle "pickle: Convert Python objects to streams of bytes and back.") modules.

See also: The [TOML Kit package](https://pypi.org/project/tomlkit/) is a style-preserving TOML library with both read and write capability. It is a recommended replacement for this module for editing already existing TOML files.

This module defines the following functions:

```python
tomlib.load(fp, /, *, parse_float=float) -> dict
```

Read a TOML file. The first argument should be a readable and binary file object. Return a [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "dict"). Convert TOML types to Python using this [conversion table](https://docs.python.org/3/library/tomllib.html#toml-to-py-table).

`parse_float` will be called with the string of every TOML float to be decoded. By default, this is equivalent to ` float(num_str) `. This can be used to use another datatype or parser for TOML floats (e.g. [` decimal.Decimal `](https://docs.python.org/3/library/decimal.html#decimal.Decimal "decimal.Decimal")). The callable must not return a [` dict `](https://docs.python.org/3/library/stdtypes.html#dict "dict") or a [` list `](https://docs.python.org/3/library/stdtypes.html#list "list"), else a [` ValueError `](https://docs.python.org/3/library/exceptions.html#ValueError "ValueError") is raised.

A [`TOMLDecodeError`](https://docs.python.org/3/library/tomllib.html#tomllib.TOMLDecodeError "tomllib.TOMLDecodeError") will be raised on an invalid TOML document.

```python
tomlib.loads(s, /, *, parse_float=float) -> dict
```

Load TOML from a [`str`](https://docs.python.org/3/library/stdtypes.html#str "str") object. Return a [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "dict"). Convert TOML types to Python using this [conversion table](https://docs.python.org/3/library/tomllib.html#toml-to-py-table). The `parse_float` argument has the same meaning as in [` load() `](https://docs.python.org/3/library/tomllib.html#tomllib.load "tomllib.load").

A [`TOMLDecodeError`](https://docs.python.org/3/library/tomllib.html#tomllib.TOMLDecodeError "tomllib.TOMLDecodeError") will be raised on an invalid TOML document.

The following exceptions are available:

```python
exception tomlib.TOMLDecodeError
```

Subclass of [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError "ValueError").

## Examples
Parsing a TOML file:

```python
import tomllib

with open("pyproject.toml", "rb") as f:
    data = tomllib.load(f)
```

Parsing a TOML string:

```python
import tomllib

toml_str = """
python-version = "3.11.0"
python-implementation = "CPython"
"""

data = tomllib.loads(toml_str)
```

## Conversion Table

| TOML             | Python                                                                           |
| ---------------- | -------------------------------------------------------------------------------- |
| TOML document    | dict                                                                             |
| string           | str                                                                              |
| integer          | int                                                                              |
| float            | float (configurable with _parse_float_)                                          |
| boolean          | bool                                                                             |
| offset date-time | datetime.datetime (`tzinfo` attribute set to an instance of `datetime.timezone`) |
| local date-time  | datetime.datetime (`tzinfo` attribute set to `None`)                             |
| local date       | datetime.date                                                                    |
| local time       | datetime.time                                                                    |
| array            | list                                                                             |
| table            | dict                                                                             |
| inline table     | dict                                                                             |
| array of tables  | list of dicts                                                                    |