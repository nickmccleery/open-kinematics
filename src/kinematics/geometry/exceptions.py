from enum import Enum, auto
from pathlib import Path
from typing import Optional


class GeometryErrorCode(Enum):
    FILE_NOT_FOUND = auto()
    INVALID_CONTENTS = auto()
    FILE_ERROR = auto()


class GeometryError(Exception):
    def __init__(
        self, message: str, code: GeometryErrorCode, detail: Optional[str] = None
    ):
        self.message = message
        self.code = code
        self.detail = detail
        super().__init__(self.message)

    @classmethod
    def get_default_message(cls) -> str:
        return "An error occurred while processing geometry"


class GeometryFileNotFound(GeometryError):
    """Raised when a geometry file cannot be found."""

    def __init__(self, path: str | Path, message: Optional[str] = None):
        self.path = path
        super().__init__(
            message=message or f"Geometry file not found: {path}",
            code=GeometryErrorCode.FILE_NOT_FOUND,
        )


class InvalidGeometryFileContents(GeometryError):
    def __init__(self, message: Optional[str] = None, detail: Optional[str] = None):
        super().__init__(
            message=message or "Invalid geometry file contents",
            code=GeometryErrorCode.INVALID_CONTENTS,
            detail=detail,
        )


class GeometryFileError(GeometryError):
    def __init__(self, message: Optional[str] = None, detail: Optional[str] = None):
        super().__init__(
            message=message or "Error processing geometry file",
            code=GeometryErrorCode.FILE_ERROR,
            detail=detail,
        )
