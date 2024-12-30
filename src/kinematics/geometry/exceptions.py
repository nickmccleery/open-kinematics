from enum import Enum, auto
from pathlib import Path
from typing import Optional


class GeometryErrorCode(Enum):
    FILE_NOT_FOUND = auto()
    INVALID_FILE_CONTENTS = auto()
    FILE_ERROR = auto()


class GeometryError(Exception):
    def __init__(
        self,
        message: str,
        code: GeometryErrorCode,
        detail: Optional[str] = None,
    ):
        self.message = message
        self.code = code
        self.detail = detail
        super().__init__(self.message)


class GeometryFileError(GeometryError):
    def __init__(self, message: Optional[str] = None, detail: Optional[str] = None):
        super().__init__(
            message=message or "Encountered an error while processing geometry.",
            code=GeometryErrorCode.FILE_ERROR,
            detail=detail,
        )


class GeometryFileNotFound(GeometryError):
    def __init__(self, path: str | Path, message: Optional[str] = None):
        self.path = path
        super().__init__(
            message=message or f"Geometry file not found: {path}",
            code=GeometryErrorCode.FILE_NOT_FOUND,
        )


class InvalidGeometryFileContents(GeometryError):
    def __init__(self, message: Optional[str] = None, detail: Optional[str] = None):
        super().__init__(
            message=message or "Invalid geometry file contents.",
            code=GeometryErrorCode.INVALID_FILE_CONTENTS,
            detail=detail,
        )


class UnsupportedGeometryType(GeometryError):
    def __init__(self, message: Optional[str] = None, detail: Optional[str] = None):
        super().__init__(
            message=message or "Unsupported geometry type.",
            code=GeometryErrorCode.INVALID_FILE_CONTENTS,
            detail=detail,
        )
