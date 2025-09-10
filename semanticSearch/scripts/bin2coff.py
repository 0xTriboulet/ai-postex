#!/usr/bin/env python3
"""bin2coff.py
usage: bin2coff.py [-h] [-s SYMBOL] [-m {amd64,i386,arm,arm64}] input [output]

Converts an arbitrary file into a linkable COFF.

positional arguments:
  input                 Input file for generating the COFF
  output                Output for the generated COFF (defaults to the input file name with a '.o' extension)

options:
  -h, --help            show this help message and exit
  -s, --symbol SYMBOL   Name of the output symbol (defaults to the name of the input file with '.' replaced with '_')
  -m, --machine {amd64,i386,arm,arm64}
                        Machine value for the COFF
"""

from __future__ import annotations
import argparse
import enum
import io
import mmap
import os
import pathlib
import struct
from dataclasses import dataclass, astuple, field

__author__  = "Matt Ehrnschwender" # Modifications made by S. Salinas
__license__ = "MIT"


class DataclassStruct(struct.Struct):
    def __init__(self, form: str):
        super().__init__(form)

    def pack(self) -> bytes:
        return super().pack(*astuple(self))

    def pack_into(self, buffer, offset):
        return super().pack_into(buffer, offset, *astuple(self))

class CoffCharacteristics(enum.IntFlag):
    RelocsStripped = 0x0001
    LineNumsStripped = 0x0004

@dataclass
class CoffFileHeader(DataclassStruct):
    machine: CoffMachine
    number_of_sections: int = 0
    timedate_stamp: int = 0
    pointer_to_symbol_table: int = 0
    number_of_symbols: int = 0
    size_of_optional_header: int = 0
    characteristics: CoffCharacteristics = CoffCharacteristics.LineNumsStripped

    def __post_init__(self):
        super().__init__("<2H3I2H")

class CoffMachine(enum.IntEnum):
    Amd64 = 0x8664
    I386 = 0x14c
    Arm = 0x1c0
    Arm64 = 0xaa64

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def from_str(s: str) -> CoffMachine:
        match s:
            case "amd64":
                return CoffMachine.Amd64
            case "i386":
                return CoffMachine.I386
            case "arm":
                return CoffMachine.Arm
            case "arm64":
                return CoffMachine.Arm64
            case _:
                raise ValueError()

class CoffSectionCharacteristics(enum.IntFlag):
    Reserved0 = 0
    CntInitializedData = 0x40
    Align16Bytes = 0x00500000
    MemRead = 0x40000000
    MemWrite = 0x80000000

@dataclass
class CoffSectionHeader(DataclassStruct):
    name: bytes = bytes(8)
    virtual_size: int = 0
    virtual_address: int = 0
    size_of_raw_data: int = 0
    pointer_to_raw_data: int = 0
    pointer_to_relocations: int = 0
    pointer_to_line_numbers: int = 0
    number_of_relocations: int = 0
    number_of_line_numbers: int = 0
    characteristics: CoffSectionCharacteristics = CoffSectionCharacteristics.Reserved0

    def __post_init__(self):
        super().__init__("<8s6I2HI")


class CoffSymbolComplexType(enum.IntEnum):
    Null = 0

class CoffSymbolBaseType(enum.IntEnum):
    Null = 0

@dataclass
class CoffSymbolType(DataclassStruct):
    complex: CoffSymbolComplexType = CoffSymbolComplexType.Null
    base: CoffSymbolBaseType = CoffSymbolBaseType.Null

    def __int__(self):
        return self.complex << 8 | self.base

class CoffSectionNumberValue(enum.IntEnum):
    Undefined = 0
    Absolute = -1
    Debug = -2

class CoffSymbolStorageClass(enum.IntEnum):
    Null = 0
    External = 2

@dataclass
class CoffSymbol(DataclassStruct):
    name: bytes = bytes(8)
    value: int = 0
    section_number: CoffSectionNumberValue = CoffSectionNumberValue.Undefined
    symbol_type: int = field(default_factory=lambda: int(CoffSymbolType()))
    storage_class: CoffSymbolStorageClass = CoffSymbolStorageClass.Null
    number_of_aux_symbols: int = 0

    def __post_init__(self):
        super().__init__("<8sIhH2B")


class StringTable:
    _data: bytearray = bytearray()

    def add_string(self, s: str) -> int:
        offset = len(self._data)
        self._data.extend(s.encode() + b'\0')
        return offset + 4

    def pack(self) -> bytes:
        return struct.pack("<I", len(self._data) + 4) + bytes(self._data)

    @property
    def size(self) -> int:
        return 4 + len(self._data)

class CoffSymbolTable:
    _tbl: list[CoffSymbol]

    def __init__(self):
        self._tbl = []

    def add_symbol(self, string_table: SringTable, **kwargs) -> StringTable:
        if symbol_name := kwargs.get("name"):
            if len(symbol_name) > 8:
                offset = string_table.add_string(symbol_name)
                symbol_name = struct.pack("<II", 0, offset)
            else:
                symbol_name = symbol_name.encode().ljust(8, b'\0')

            kwargs["name"] = symbol_name

        symbol = CoffSymbol(**kwargs)
        self._tbl.append(symbol)
        return string_table

    @property
    def size(self) -> int:
        return len(self._tbl) * CoffSymbol.size

    def pack(self) -> bytes:
        packed = bytearray()
        for symbol in self._tbl:
            packed += symbol.pack()

        return packed

class CoffBuilder:
    _inputfp: io.FileIO
    _inputsize: int
    symbol: str
    machine: CoffMachine

    def __init__(self, inputfp: io.FileIO, symbol: str, machine: CoffMachine):
        self._inputfp = inputfp
        inputfp.seek(0, os.SEEK_END)
        self._inputsize = inputfp.tell()
        inputfp.seek(0)

        self.symbol = symbol
        self.machine = machine

    def write_output(self, outfp: io.BytesIO):
        string_table = StringTable()
        symbol_table = CoffSymbolTable()
        prefix = ""
        size_val = struct.pack("<Q", self._inputsize)
        if self.machine == CoffMachine.I386 or self.machine == CoffMachine.Arm:
            size_val = struct.pack("<I", self._inputsize)
            
        if self.machine == CoffMachine.I386:
            prefix = "_"

        string_table = symbol_table.add_symbol(
            string_table,
            name=f"{prefix}{self.symbol}_start",
            value=len(size_val),
            section_number=1,
            storage_class=CoffSymbolStorageClass.External,
        )

        string_table = symbol_table.add_symbol(
            string_table,
            name=f"{prefix}{self.symbol}_end",
            value=len(size_val) + self._inputsize,
            section_number=1,
            storage_class=CoffSymbolStorageClass.External
        )

        string_table = symbol_table.add_symbol(
            string_table,
            name=f"{prefix}{self.symbol}_size",
            value=0,
            section_number=1,
            storage_class=CoffSymbolStorageClass.External,
        )

        section_len = (len(size_val) + self._inputsize + 0xf) & -0x10

        header_size = CoffFileHeader(self.machine).size + CoffSectionHeader().size
        pointer_to_raw_data = header_size
        pointer_to_symbol_table = pointer_to_raw_data + section_len

        rdata_section_header = CoffSectionHeader(
            name=b".rdata",
            size_of_raw_data=section_len,
            pointer_to_raw_data=pointer_to_raw_data,
            characteristics=CoffSectionCharacteristics.Align16Bytes | CoffSectionCharacteristics.MemRead | CoffSectionCharacteristics.CntInitializedData
        )

        file_header = CoffFileHeader(
            machine=self.machine,
            number_of_sections=1,
            pointer_to_symbol_table=pointer_to_symbol_table,
            number_of_symbols=3,
            characteristics=CoffCharacteristics.LineNumsStripped | CoffCharacteristics.RelocsStripped,
        )

        outfp.write(file_header.pack())
        outfp.write(rdata_section_header.pack())

        outfp.write(size_val)
        if self._inputsize > 8 * 2**20:
            with mmap.mmap(self._inputfp.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                outfp.write(mm)
        else:
            outfp.write(self._inputfp.read())

        outfp.write(bytearray(section_len - (len(size_val) + self._inputsize)))

        outfp.write(symbol_table.pack())
        outfp.write(string_table.pack())


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="bin2coff.py",
        description="Converts an arbitrary file into a linkable COFF."
    )

    parser.add_argument(
        "input",
        help="Input file for generating the COFF",
        type=pathlib.Path,
    )

    parser.add_argument(
        "-s",
        "--symbol",
        help="Name of the output symbol (defaults to the name of the input file with '.' replaced with '_')",
        type=str,
        default=None
    )

    parser.add_argument(
        "-m",
        "--machine",
        help="Machine value for the COFF",
        type=CoffMachine.from_str,
        choices=list(CoffMachine),
        default=CoffMachine.Amd64,
    )

    parser.add_argument(
        "output",
        help="Output for the generated COFF (defaults to the input file name with a '.o' extension)",
        type=pathlib.Path,
        nargs="?",
        default=None
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.symbol is None:
        args.symbol = args.input.name.replace(".", "_")

    if args.output:
        output = args.output
    else:
        parent = args.input.parent
        output = parent.joinpath(args.input.stem).with_suffix(".o")

    with args.input.open("rb") as f:
        builder = CoffBuilder(f, args.symbol, args.machine)

        with output.open("wb") as f:
            builder.write_output(f)