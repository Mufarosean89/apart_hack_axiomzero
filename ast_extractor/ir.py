"""
Intermediate Representation (IR) data structures for Axiom Zero.
Defines normalized, semantically meaningful constructs stripped of Python-isms.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum


class TypeKind(Enum):
    """Fundamental types in the normalized IR."""
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "string"
    TENSOR = "tensor"
    LIST = "list"
    DICT = "dict"
    TUPLE = "tuple"
    VOID = "void"
    UNKNOWN = "unknown"
    CUSTOM = "custom"


class TensorOpKind(Enum):
    """Tensor operations for PyTorch code."""
    MATMUL = "matmul"
    ADD = "add"
    MUL = "mul"
    DIV = "div"
    SUB = "sub"
    RELU = "relu"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    CONV2D = "conv2d"
    LINEAR = "linear"
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    CONCAT = "concat"
    SPLIT = "split"
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    UNKNOWN = "unknown"


@dataclass
class TypeAnnotationIR:
    """Type annotation in normalized form."""
    type_kind: TypeKind
    type_name: Optional[str] = None  # For custom types
    shape: Optional[List[int]] = None  # For tensors
    element_type: Optional['TypeAnnotationIR'] = None  # For containers
    nullable: bool = False

    def to_string(self) -> str:
        """Convert to string representation."""
        if self.type_kind == TypeKind.TENSOR:
            shape_str = f"[{','.join(map(str, self.shape))}]" if self.shape else "[?]"
            return f"Tensor{shape_str}"
        elif self.type_kind == TypeKind.LIST:
            elem = self.element_type.to_string() if self.element_type else "any"
            return f"List[{elem}]"
        elif self.type_kind == TypeKind.CUSTOM and self.type_name:
            return self.type_name
        else:
            return self.type_kind.value


@dataclass
class ExpressionIR:
    """Normalized expression representation."""
    expr_type: str  # 'literal', 'variable', 'binary_op', 'call', 'tensor_op', 'attribute'
    value: Any = None
    operator: Optional[str] = None
    left: Optional['ExpressionIR'] = None
    right: Optional['ExpressionIR'] = None
    function_name: Optional[str] = None
    arguments: List['ExpressionIR'] = field(default_factory=list)
    tensor_op: Optional[TensorOpKind] = None
    attributes: List[str] = field(
        default_factory=list)  # For chained attributes
    line_number: Optional[int] = None

    def is_tensor_operation(self) -> bool:
        """Check if this is a tensor operation."""
        return self.tensor_op is not None or (self.function_name and
                                              any(op in self.function_name.lower() for op in
                                                  ['matmul', 'addmm', 'conv2d', 'linear', 'relu', 'sigmoid']))


@dataclass
class StatementIR:
    """Normalized statement representation."""
    stmt_type: str  # 'assignment', 'return', 'expression', 'pass'
    expression: Optional[ExpressionIR] = None
    target: Optional[str] = None  # Variable name for assignments
    line_number: Optional[int] = None


@dataclass
class LoopIR:
    """Normalized loop construct."""
    loop_type: str  # 'for', 'while'
    variable: Optional[str] = None  # Loop variable
    iterable: Optional[ExpressionIR] = None  # For 'for' loops
    condition: Optional[ExpressionIR] = None  # For 'while' loops
    range_start: Optional[ExpressionIR] = None  # For range() loops
    range_end: Optional[ExpressionIR] = None
    range_step: Optional[ExpressionIR] = None
    body: List[Union['StatementIR', 'LoopIR', 'ConditionalIR']
               ] = field(default_factory=list)
    line_number: Optional[int] = None

    def is_bounded(self) -> bool:
        """Check if loop has statically determinable bounds."""
        if self.loop_type == 'for' and self.range_start is not None and self.range_end is not None:
            return True
        return False


@dataclass
class ConditionalIR:
    """Normalized conditional construct."""
    condition: ExpressionIR
    then_branch: List[Union['StatementIR', 'LoopIR',
                            'ConditionalIR']] = field(default_factory=list)
    else_branch: List[Union['StatementIR', 'LoopIR',
                            'ConditionalIR']] = field(default_factory=list)
    line_number: Optional[int] = None


@dataclass
class FunctionSignatureIR:
    """Normalized function signature."""
    name: str
    parameters: List[Dict[str, Any]] = field(
        default_factory=list)  # [{name, type, default}]
    return_type: Optional[TypeAnnotationIR] = None
    is_method: bool = False
    class_name: Optional[str] = None
    line_number: Optional[int] = None


@dataclass
class FunctionIR:
    """Normalized function representation."""
    signature: FunctionSignatureIR
    body: List[Union[StatementIR, LoopIR, ConditionalIR,
                     'FunctionIR']] = field(default_factory=list)
    preconditions: List[ExpressionIR] = field(default_factory=list)
    postconditions: List[ExpressionIR] = field(default_factory=list)
    invariants: List[ExpressionIR] = field(default_factory=list)
    tensor_operations: List[TensorOpKind] = field(default_factory=list)
    line_number: Optional[int] = None

    def extract_tensor_ops(self):
        """Extract all tensor operations from function body."""
        def traverse(node):
            if isinstance(node, StatementIR) and node.expression:
                if node.expression.is_tensor_operation():
                    if node.expression.tensor_op:
                        self.tensor_operations.append(
                            node.expression.tensor_op)
            elif isinstance(node, LoopIR):
                for stmt in node.body:
                    traverse(stmt)
            elif isinstance(node, ConditionalIR):
                for stmt in node.then_branch + node.else_branch:
                    traverse(stmt)

        for stmt in self.body:
            traverse(stmt)


@dataclass
class ClassIR:
    """Normalized class representation."""
    name: str
    methods: List[FunctionIR] = field(default_factory=list)
    attributes: List[Dict[str, Any]] = field(default_factory=list)
    base_classes: List[str] = field(default_factory=list)
    line_number: Optional[int] = None


@dataclass
class NormalizedIR:
    """
    Complete normalized intermediate representation.
    Strips Python-isms and keeps only semantically meaningful constructs.
    """
    source_file: Optional[str] = None
    functions: List[FunctionIR] = field(default_factory=list)
    classes: List[ClassIR] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    global_statements: List[StatementIR] = field(default_factory=list)

    # Metadata for verification
    total_functions: int = 0
    total_loops: int = 0
    total_conditionals: int = 0
    total_tensor_ops: int = 0

    def compute_statistics(self):
        """Compute statistics about the IR."""
        self.total_functions = len(self.functions)

        def count_constructs(nodes):
            loops = 0
            conditionals = 0
            tensor_ops = 0
            for node in nodes:
                if isinstance(node, LoopIR):
                    loops += 1
                    sub_loops, sub_cond, sub_ops = count_constructs(node.body)
                    loops += sub_loops
                    conditionals += sub_cond
                    tensor_ops += sub_ops
                elif isinstance(node, ConditionalIR):
                    conditionals += 1
                    sub_loops, sub_cond, sub_ops = count_constructs(
                        node.then_branch)
                    loops += sub_loops
                    conditionals += sub_cond
                    tensor_ops += sub_ops
                    sub_loops, sub_cond, sub_ops = count_constructs(
                        node.else_branch)
                    loops += sub_loops
                    conditionals += sub_cond
                    tensor_ops += sub_ops
                elif isinstance(node, FunctionIR):
                    sub_loops, sub_cond, sub_ops = count_constructs(node.body)
                    loops += sub_loops
                    conditionals += sub_cond
                    tensor_ops += sub_ops
                    tensor_ops += len(node.tensor_operations)
            return loops, conditionals, tensor_ops

        for func in self.functions:
            loops, conds, ops = count_constructs(func.body)
            self.total_loops += loops
            self.total_conditionals += conds
            self.total_tensor_ops += ops + len(func.tensor_operations)

    def to_dict(self) -> Dict[str, Any]:
        """Convert IR to dictionary for serialization."""
        return {
            'source_file': self.source_file,
            'functions': [self._function_to_dict(f) for f in self.functions],
            'classes': [self._class_to_dict(c) for c in self.classes],
            'imports': self.imports,
            'statistics': {
                'total_functions': self.total_functions,
                'total_loops': self.total_loops,
                'total_conditionals': self.total_conditionals,
                'total_tensor_ops': self.total_tensor_ops
            }
        }

    def _function_to_dict(self, func: FunctionIR) -> Dict[str, Any]:
        """Convert FunctionIR to dict."""
        return {
            'name': func.signature.name,
            'parameters': func.signature.parameters,
            'return_type': func.signature.return_type.to_string() if func.signature.return_type else None,
            'has_tensor_ops': len(func.tensor_operations) > 0,
            'tensor_ops': [op.value for op in func.tensor_operations],
            'line_number': func.line_number
        }

    def _class_to_dict(self, cls: ClassIR) -> Dict[str, Any]:
        """Convert ClassIR to dict."""
        return {
            'name': cls.name,
            'methods': [self._function_to_dict(m) for m in cls.methods],
            'base_classes': cls.base_classes
        }
