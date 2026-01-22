"""
WGSL (WebGPU Shading Language) frontend for parsing and converting to NAGA IR.

Translated from wgpu-trunk/naga/src/front/wgsl/
"""

from .parser import Frontend, WgslParser, Options, parse_str, ParseError
from .error import ParseError as WgslParseError, Error
from .index import Index
from .ast import TranslationUnit, GlobalDecl, Expression, Statement
from .lexer import Lexer, Token, TokenKind
from .conv import (
    get_scalar_type,
    map_built_in,
    map_interpolation,
    map_sampling,
    map_address_space,
    map_storage_access,
    map_storage_format,
    map_conservative_depth,
)
from .number import Number, NumberType, parse_number
from .directive import (
    EnableExtension,
    LanguageExtension,
    EnableExtensions,
    LanguageExtensions,
    EnableDirective,
    RequiresDirective,
    DiagnosticDirective,
)

__all__ = [
    # Main API
    'Frontend',
    'WgslParser',
    'Options',
    'parse_str',
    'ParseError',
    
    # Error handling
    'WgslParseError',
    'Error',
    
    # AST and indexing
    'Index',
    'TranslationUnit',
    'GlobalDecl',
    'Expression',
    'Statement',
    
    # Lexer
    'Lexer',
    'Token',
    'TokenKind',
    
    # Conversion utilities
    'get_scalar_type',
    'map_built_in',
    'map_interpolation',
    'map_sampling',
    'map_address_space',
    'map_storage_access',
    'map_storage_format',
    'map_conservative_depth',
    
    # Number parsing
    'Number',
    'NumberType',
    'parse_number',
    
    # Directives and extensions
    'EnableExtension',
    'LanguageExtension',
    'EnableExtensions',
    'LanguageExtensions',
    'EnableDirective',
    'RequiresDirective',
    'DiagnosticDirective',
]
