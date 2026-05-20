"""
Type inference engine for abstract interpretation.
Infers types and tracks data flow across variables and expressions.
"""

from typing import Dict, List, Optional, Any, Union
from .abstract_domain import (
    AbstractValue,
    TypeDomain,
    TensorShape,
    ShapeDimension,
)
from .shape_analysis import TensorShapeAnalyzer
from ast_extractor.ir import (
    NormalizedIR,
    FunctionIR,
    LoopIR,
    ConditionalIR,
    StatementIR,
    ExpressionIR,
    TypeAnnotationIR,
    TypeKind,
    TensorOpKind,
)


class TypeInferenceEngine:
    """
    Infers types for variables and expressions in the IR.
    Uses type annotations where available, infers from operations otherwise.
    """

    def __init__(self):
        """Initialize type inference engine."""
        self.type_env = {}  # Current type environment
        self.inference_log = []

    def infer_function_types(self, func: FunctionIR,
                             global_env: Dict[str, AbstractValue] = None) -> Dict[str, AbstractValue]:
        """
        Infer types for all variables in a function.

        Args:
            func: FunctionIR to analyze
            global_env: Global type environment

        Returns:
            Local type environment with inferred types
        """
        local_env = {}

        # Initialize with parameter types
        if global_env:
            local_env.update(global_env)

        for param in func.signature.parameters:
            param_name = param['name']
            if param['type']:
                local_env[param_name] = self._type_annotation_to_abstract(
                    param['type'])
            else:
                local_env[param_name] = AbstractValue(
                    type_domain=TypeDomain.BOTTOM)

        # Analyze function body
        self._infer_block_types(func.body, local_env)

        return local_env

    def _infer_block_types(self, block: List, env: Dict[str, AbstractValue]):
        """
        Infer types for a block of statements.

        Args:
            block: List of IR nodes (statements, loops, conditionals)
            env: Type environment (modified in-place)
        """
        for node in block:
            if isinstance(node, StatementIR):
                self._infer_statement_types(node, env)
            elif isinstance(node, LoopIR):
                self._infer_loop_types(node, env)
            elif isinstance(node, ConditionalIR):
                self._infer_conditional_types(node, env)
            elif isinstance(node, FunctionIR):
                # Nested function
                self.infer_function_types(node, env.copy())

    def _infer_statement_types(self, stmt: StatementIR, env: Dict[str, AbstractValue]):
        """Infer types for a statement."""
        if stmt.stmt_type == 'assignment' and stmt.target:
            if stmt.expression:
                expr_type = self._infer_expression_type(stmt.expression, env)
                env[stmt.target] = expr_type
                self.inference_log.append(f"{stmt.target}: {expr_type}")

        elif stmt.stmt_type == 'return':
            if stmt.expression:
                return_type = self._infer_expression_type(stmt.expression, env)
                # Could validate against function signature here

    def _infer_loop_types(self, loop: LoopIR, env: Dict[str, AbstractValue]):
        """Infer types for a loop."""
        if loop.loop_type == 'for':
            # Infer loop variable type from iterable
            if loop.variable:
                if loop.range_start is not None and loop.range_end is not None:
                    env[loop.variable] = AbstractValue.constant(
                        0, TypeDomain.INT)
                else:
                    # General for loop — element type unknown without deeper analysis
                    env[loop.variable] = AbstractValue(
                        type_domain=TypeDomain.BOTTOM)

            # Infer body types
            self._infer_block_types(loop.body, env)

        elif loop.loop_type == 'while':
            if loop.condition:
                self._infer_expression_type(loop.condition, env)
            self._infer_block_types(loop.body, env)

    def _infer_conditional_types(self, cond: ConditionalIR, env: Dict[str, AbstractValue]):
        """Infer types for a conditional."""
        if cond.condition:
            self._infer_expression_type(cond.condition, env)

        # Both branches should have consistent types
        env_then = env.copy()
        self._infer_block_types(cond.then_branch, env_then)

        env_else = env.copy()
        self._infer_block_types(cond.else_branch, env_else)

        # Merge environments (join)
        for var in env_then.keys() | env_else.keys():
            if var in env_then and var in env_else:
                env[var] = env_then[var].join(env_else[var])
            elif var in env_then:
                env[var] = env_then[var]
            elif var in env_else:
                env[var] = env_else[var]

    def _infer_expression_type(self, expr: ExpressionIR,
                               env: Dict[str, AbstractValue]) -> AbstractValue:
        """
        Infer type of an expression.

        Args:
            expr: ExpressionIR
            env: Type environment

        Returns:
            Inferred abstract value
        """
        if expr.expr_type == 'literal':
            # Infer type from literal value
            if isinstance(expr.value, bool):
                return AbstractValue.constant(expr.value, TypeDomain.BOOL)
            elif isinstance(expr.value, int):
                return AbstractValue.constant(expr.value, TypeDomain.INT)
            elif isinstance(expr.value, float):
                return AbstractValue.constant(expr.value, TypeDomain.FLOAT)
            elif isinstance(expr.value, str):
                return AbstractValue.constant(expr.value, TypeDomain.STRING)
            elif isinstance(expr.value, list):
                return AbstractValue(type_domain=TypeDomain.LIST)
            else:
                return AbstractValue(type_domain=TypeDomain.BOTTOM)

        elif expr.expr_type == 'variable':
            # Look up variable type
            return env.get(expr.value, AbstractValue(type_domain=TypeDomain.BOTTOM))

        elif expr.expr_type == 'binary_op':
            # Infer from operands
            left_type = self._infer_expression_type(
                expr.left, env) if expr.left else AbstractValue()
            right_type = self._infer_expression_type(
                expr.right, env) if expr.right else AbstractValue()

            # Arithmetic operations preserve numeric types
            if expr.operator in ['+', '-', '*', '/', '//', '%', '**']:
                return AbstractValue(type_domain=TypeDomain.FLOAT)

            # Comparison operations return bool
            elif expr.operator in ['<', '<=', '>', '>=', '==', '!=']:
                return AbstractValue(type_domain=TypeDomain.BOOL)

            # Boolean operations
            elif expr.operator in ['and', 'or']:
                return AbstractValue(type_domain=TypeDomain.BOOL)

            return AbstractValue(type_domain=TypeDomain.BOTTOM)

        elif expr.expr_type == 'call':
            # Check if it's a tensor operation
            if expr.tensor_op:
                return self._infer_tensor_op_type(expr, env)

            # For other function calls, try to infer from function name
            return AbstractValue(type_domain=TypeDomain.BOTTOM)

        elif expr.expr_type == 'subscript':
            # Array/tensor indexing
            if expr.left:
                base_type = self._infer_expression_type(expr.left, env)
                if base_type.is_tensor():
                    # Indexing a tensor reduces rank by 1
                    if base_type.tensor_shape and len(base_type.tensor_shape.dimensions) > 0:
                        new_dims = base_type.tensor_shape.dimensions[1:]
                        return AbstractValue.from_tensor(TensorShape(dimensions=new_dims))
                elif base_type.type_domain == TypeDomain.LIST:
                    # Element type
                    return AbstractValue(type_domain=TypeDomain.BOTTOM)

            return AbstractValue(type_domain=TypeDomain.BOTTOM)

        return AbstractValue(type_domain=TypeDomain.BOTTOM)

    def _infer_tensor_op_type(self, expr: ExpressionIR,
                              env: Dict[str, AbstractValue]) -> AbstractValue:
        """Infer type for tensor operations."""
        analyzer = TensorShapeAnalyzer()

        if expr.tensor_op == TensorOpKind.MATMUL:
            if len(expr.arguments) >= 2:
                shape_a = self._get_tensor_shape(expr.arguments[0], env)
                shape_b = self._get_tensor_shape(expr.arguments[1], env)

                if shape_a and shape_b:
                    output_shape = analyzer.infer_matmul_shape(
                        shape_a, shape_b)
                    if output_shape:
                        return AbstractValue.from_tensor(output_shape)

        elif expr.tensor_op == TensorOpKind.ADD:
            if len(expr.arguments) >= 2:
                shape_a = self._get_tensor_shape(expr.arguments[0], env)
                shape_b = self._get_tensor_shape(expr.arguments[1], env)

                if shape_a and shape_b:
                    output_shape = analyzer.infer_elementwise_shape(
                        shape_a, shape_b)
                    if output_shape:
                        return AbstractValue.from_tensor(output_shape)

        elif expr.tensor_op == TensorOpKind.RELU:
            if len(expr.arguments) >= 1:
                shape = self._get_tensor_shape(expr.arguments[0], env)
                if shape:
                    return AbstractValue.from_tensor(shape)

        # Default: return tensor with unknown shape
        return AbstractValue(type_domain=TypeDomain.TENSOR)

    def _get_tensor_shape(self, expr: ExpressionIR,
                          env: Dict[str, AbstractValue]) -> Optional[TensorShape]:
        """Get tensor shape from expression."""
        abs_value = self._infer_expression_type(expr, env)
        return abs_value.tensor_shape if abs_value.is_tensor() else None

    def _type_annotation_to_abstract(self, type_ann: TypeAnnotationIR) -> AbstractValue:
        """Convert TypeAnnotationIR to AbstractValue."""
        type_map = {
            TypeKind.INT: TypeDomain.INT,
            TypeKind.FLOAT: TypeDomain.FLOAT,
            TypeKind.BOOL: TypeDomain.BOOL,
            TypeKind.STRING: TypeDomain.STRING,
            TypeKind.TENSOR: TypeDomain.TENSOR,
            TypeKind.LIST: TypeDomain.LIST,
            TypeKind.DICT: TypeDomain.DICT,
        }

        domain = type_map.get(type_ann.type_kind, TypeDomain.BOTTOM)

        if type_ann.type_kind == TypeKind.TENSOR and type_ann.shape:
            shape = TensorShape.from_list(type_ann.shape)
            return AbstractValue.from_tensor(shape)

        return AbstractValue(type_domain=domain)
