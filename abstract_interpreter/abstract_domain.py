"""
Abstract domains for type and shape analysis.
Defines the lattice structures used in abstract interpretation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum


class TypeDomain(Enum):
    """Type lattice for abstract interpretation."""
    BOTTOM = "⊥"  # No information
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "string"
    TENSOR = "tensor"
    LIST = "list"
    DICT = "dict"
    TOP = ""  # Any type (maximum uncertainty)

    @staticmethod
    def join(t1: 'TypeDomain', t2: 'TypeDomain') -> 'TypeDomain':
        """Compute least upper bound (join) in type lattice."""
        if t1 == t2:
            return t1
        if t1 == TypeDomain.BOTTOM:
            return t2
        if t2 == TypeDomain.BOTTOM:
            return t1
        if t1 == TypeDomain.TOP or t2 == TypeDomain.TOP:
            return TypeDomain.TOP

        # Numeric types can join to a common supertype
        if {t1, t2} <= {TypeDomain.INT, TypeDomain.FLOAT}:
            return TypeDomain.FLOAT

        return TypeDomain.TOP


class ShapeDimension:
    """Represents a single dimension in a tensor shape."""

    def __init__(self, value: Optional[int] = None, symbolic: Optional[str] = None):
        self.value = value
        self.symbolic = symbolic

    def is_concrete(self) -> bool:
        """Check if dimension has a concrete value."""
        return self.value is not None

    def is_symbolic(self) -> bool:
        """Check if dimension is symbolic."""
        return self.symbolic is not None

    def __eq__(self, other):
        if not isinstance(other, ShapeDimension):
            return False
        return self.value == other.value and self.symbolic == other.symbolic

    def __repr__(self):
        if self.is_concrete():
            return str(self.value)
        elif self.is_symbolic():
            return self.symbolic
        return "?"

    def __hash__(self):
        return hash((self.value, self.symbolic))

    def matches(self, other: 'ShapeDimension') -> bool:
        """Check if dimensions are compatible."""
        if self.is_concrete() and other.is_concrete():
            return self.value == other.value
        # Symbolic dimensions can match any concrete or same symbolic
        return True


@dataclass
class TensorShape:
    """Symbolic tensor shape representation."""
    dimensions: List[ShapeDimension] = field(default_factory=list)
    rank: Optional[int] = None

    def __post_init__(self):
        if self.rank is None and self.dimensions:
            self.rank = len(self.dimensions)

    @staticmethod
    def from_list(shape_list: List[Union[int, str]]) -> 'TensorShape':
        """Create TensorShape from list of ints or symbolic strings."""
        dims = []
        for dim in shape_list:
            if isinstance(dim, int):
                dims.append(ShapeDimension(value=dim))
            elif isinstance(dim, str):
                dims.append(ShapeDimension(symbolic=dim))
            else:
                dims.append(ShapeDimension())
        return TensorShape(dimensions=dims)

    @staticmethod
    def unknown(rank: int = None) -> 'TensorShape':
        """Create unknown tensor shape."""
        if rank is not None:
            dims = [ShapeDimension() for _ in range(rank)]
            return TensorShape(dimensions=dims, rank=rank)
        return TensorShape(rank=rank)

    def is_fully_known(self) -> bool:
        """Check if all dimensions are concrete."""
        return all(dim.is_concrete() for dim in self.dimensions)

    def has_symbolic_dims(self) -> bool:
        """Check if any dimensions are symbolic."""
        return any(dim.is_symbolic() for dim in self.dimensions)

    def get_symbolic_dims(self) -> Dict[str, int]:
        """Get mapping of symbolic names to their positions."""
        return {dim.symbolic: i for i, dim in enumerate(self.dimensions) if dim.is_symbolic()}

    def compatible_with(self, other: 'TensorShape') -> bool:
        """Check if shapes are compatible (for operations like matmul)."""
        if self.rank is None or other.rank is None:
            return True  # Unknown rank, assume compatible

        if len(self.dimensions) != len(other.dimensions):
            return False

        for d1, d2 in zip(self.dimensions, other.dimensions):
            if not d1.matches(d2):
                return False

        return True

    def __repr__(self):
        if not self.dimensions:
            return f"Tensor[?]"
        dims_str = ", ".join(str(d) for d in self.dimensions)
        return f"Tensor[{dims_str}]"


@dataclass
class AbstractValue:
    """
    Abstract value in the analysis domain.
    Combines type information with shape information for tensors.
    """
    type_domain: TypeDomain = TypeDomain.BOTTOM
    tensor_shape: Optional[TensorShape] = None
    concrete_value: Any = None  # For constants
    symbolic_constraints: List[str] = field(default_factory=list)
    source_location: Optional[str] = None  # Where this value came from

    def is_tensor(self) -> bool:
        """Check if value is a tensor."""
        return self.type_domain == TypeDomain.TENSOR

    def has_shape(self) -> bool:
        """Check if tensor shape is known."""
        return self.is_tensor() and self.tensor_shape is not None

    def get_shape(self) -> Optional[TensorShape]:
        """Get tensor shape if available."""
        return self.tensor_shape if self.is_tensor() else None

    @staticmethod
    def from_type(type_domain: TypeDomain) -> 'AbstractValue':
        """Create abstract value from type."""
        return AbstractValue(type_domain=type_domain)

    @staticmethod
    def from_tensor(shape: TensorShape) -> 'AbstractValue':
        """Create abstract tensor value with shape."""
        return AbstractValue(
            type_domain=TypeDomain.TENSOR,
            tensor_shape=shape
        )

    @staticmethod
    def constant(value: Any, type_domain: TypeDomain) -> 'AbstractValue':
        """Create abstract value for a constant."""
        return AbstractValue(
            type_domain=type_domain,
            concrete_value=value
        )

    def join(self, other: 'AbstractValue') -> 'AbstractValue':
        """Compute join (least upper bound) of two abstract values."""
        joined_type = TypeDomain.join(self.type_domain, other.type_domain)

        # For tensors, try to merge shapes
        joined_shape = None
        if self.is_tensor() and other.is_tensor():
            if self.tensor_shape and other.tensor_shape:
                # Merge shapes (take most general)
                if self.tensor_shape.rank == other.tensor_shape.rank:
                    dims = []
                    for d1, d2 in zip(self.tensor_shape.dimensions, other.tensor_shape.dimensions):
                        if d1 == d2:
                            dims.append(d1)
                        else:
                            dims.append(ShapeDimension())  # Unknown
                    joined_shape = TensorShape(dimensions=dims)
                else:
                    joined_shape = TensorShape.unknown()
            elif self.tensor_shape:
                joined_shape = self.tensor_shape
            else:
                joined_shape = other.tensor_shape

        return AbstractValue(
            type_domain=joined_type,
            tensor_shape=joined_shape
        )

    def __repr__(self):
        parts = [self.type_domain.value]
        if self.has_shape():
            parts.append(str(self.tensor_shape))
        if self.concrete_value is not None:
            parts.append(f"={self.concrete_value}")
        return " ".join(parts)


@dataclass
class AbstractState:
    """
    Complete abstract state after analysis.
    Contains inferred types, shapes, and data flow facts for all variables.
    """
    # Variable environments: var_name -> AbstractValue
    global_env: Dict[str, AbstractValue] = field(default_factory=dict)
    function_envs: Dict[str, Dict[str, AbstractValue]
                        ] = field(default_factory=dict)

    # Shape constraints and facts
    shape_facts: List[str] = field(
        default_factory=list)  # e.g., "B > 0", "D = 768"
    type_constraints: List[str] = field(default_factory=list)

    # Data flow information
    data_flow_graph: Dict[str, List[str]] = field(
        default_factory=dict)  # var -> [deps]

    # Function signatures with inferred types
    function_signatures: Dict[str, Dict[str, Any]
                              ] = field(default_factory=dict)

    # Tensor operations metadata
    tensor_ops_metadata: Dict[str, Dict[str, Any]
                              ] = field(default_factory=dict)

    # Analysis metadata
    analysis_complete: bool = False
    warnings: List[str] = field(default_factory=list)

    def add_shape_fact(self, fact: str):
        """Add a shape constraint fact."""
        if fact not in self.shape_facts:
            self.shape_facts.append(fact)

    def add_type_constraint(self, constraint: str):
        """Add a type constraint."""
        if constraint not in self.type_constraints:
            self.type_constraints.append(constraint)

    def get_variable_type(self, var_name: str, function: str = None) -> Optional[AbstractValue]:
        """Get inferred type for a variable."""
        if function and function in self.function_envs:
            return self.function_envs[function].get(var_name)
        return self.global_env.get(var_name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'global_env': {k: repr(v) for k, v in self.global_env.items()},
            'function_envs': {
                func: {k: repr(v) for k, v in env.items()}
                for func, env in self.function_envs.items()
            },
            'shape_facts': self.shape_facts,
            'type_constraints': self.type_constraints,
            'function_signatures': self.function_signatures,
            'tensor_ops_metadata': self.tensor_ops_metadata,
            'analysis_complete': self.analysis_complete,
            'warnings': self.warnings
        }
