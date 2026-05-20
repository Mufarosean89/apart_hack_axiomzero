"""
AST Normalizer for Axiom Zero.
Converts Python AST into normalized intermediate representation (IR),
stripping Python-isms and keeping only semantically meaningful constructs.
"""

import ast
from typing import List, Dict, Any, Optional, Union

from .ir import (
    NormalizedIR,
    FunctionIR,
    FunctionSignatureIR,
    ClassIR,
    LoopIR,
    ConditionalIR,
    StatementIR,
    ExpressionIR,
    TypeAnnotationIR,
    TypeKind,
    TensorOpKind
)


class ASTNormalizer:
    """
    Normalizes Python AST into a clean intermediate representation.
    Strips Python-specific constructs and keeps semantic meaning.
    """

    def __init__(self):
        """Initialize normalizer."""
        self.imports = []

    def normalize(self, ast_tree: ast.AST, source_file: str = None) -> NormalizedIR:
        """
        Normalize Python AST into IR.

        Args:
            ast_tree: Python AST tree
            source_file: Source filename

        Returns:
            NormalizedIR object
        """
        ir = NormalizedIR(source_file=source_file)
        self.imports = []

        # Extract imports
        for node in ast_tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.imports.append(self._normalize_import(node))

        ir.imports = self.imports

        # Extract top-level functions and classes
        for node in ast_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ir.functions.append(self._normalize_function(node))
            elif isinstance(node, ast.ClassDef):
                ir.classes.append(self._normalize_class(node))
            elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                ir.global_statements.append(self._normalize_statement(node))

        # Compute statistics
        ir.compute_statistics()

        return ir

    def _normalize_import(self, node: Union[ast.Import, ast.ImportFrom]) -> str:
        """Normalize import statement."""
        if isinstance(node, ast.Import):
            return ", ".join([alias.name for alias in node.names])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = ", ".join([alias.name for alias in node.names])
            return f"from {module} import {names}"
        return ""

    def _normalize_type_annotation(self, annotation: Optional[ast.expr]) -> Optional[TypeAnnotationIR]:
        """Normalize type annotation."""
        if annotation is None:
            return None

        if isinstance(annotation, ast.Name):
            type_name = annotation.id
            type_kind = self._map_type_name(type_name)
            return TypeAnnotationIR(type_kind=type_kind, type_name=type_name)

        elif isinstance(annotation, ast.Attribute):
            # Handle torch.Tensor, nn.Module, etc.
            attr_chain = []
            current = annotation
            while isinstance(current, ast.Attribute):
                attr_chain.append(current.attr)
                current = current.value

            if isinstance(current, ast.Name):
                attr_chain.append(current.id)

            attr_chain.reverse()
            full_name = '.'.join(attr_chain)

            # Check if it's a tensor type
            if 'Tensor' in full_name:
                return TypeAnnotationIR(type_kind=TypeKind.TENSOR, type_name=full_name)

            return TypeAnnotationIR(type_kind=TypeKind.CUSTOM, type_name=full_name)

        elif isinstance(annotation, ast.Subscript):
            # Handle generic types like List[int], Tensor[3, 4]
            if isinstance(annotation.value, ast.Name):
                container_type = annotation.value.id

                if container_type in ['List', 'list']:
                    elem_type = self._normalize_type_annotation(
                        annotation.slice)
                    return TypeAnnotationIR(
                        type_kind=TypeKind.LIST,
                        element_type=elem_type
                    )
                elif container_type == 'Tensor':
                    # Extract shape
                    shape = self._extract_shape(annotation.slice)
                    return TypeAnnotationIR(
                        type_kind=TypeKind.TENSOR,
                        shape=shape
                    )
            elif isinstance(annotation.value, ast.Attribute):
                # Handle torch.Tensor[...] if needed
                return self._normalize_type_annotation(annotation.value)

        elif isinstance(annotation, ast.Constant):
            # Handle string annotations
            return TypeAnnotationIR(type_kind=TypeKind.UNKNOWN, type_name=str(annotation.value))

        return TypeAnnotationIR(type_kind=TypeKind.UNKNOWN)

    def _map_type_name(self, name: str) -> TypeKind:
        """Map type name to TypeKind enum."""
        type_map = {
            'int': TypeKind.INT,
            'float': TypeKind.FLOAT,
            'bool': TypeKind.BOOL,
            'str': TypeKind.STRING,
            'string': TypeKind.STRING,
            'Tensor': TypeKind.TENSOR,
            'list': TypeKind.LIST,
            'dict': TypeKind.DICT,
            'tuple': TypeKind.TUPLE,
            'None': TypeKind.VOID,
        }
        return type_map.get(name, TypeKind.CUSTOM)

    def _extract_shape(self, node: ast.expr) -> Optional[List[int]]:
        """Extract tensor shape from AST node."""
        if isinstance(node, ast.Tuple):
            shape = []
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                    shape.append(elt.value)
                else:
                    shape.append(-1)  # Unknown dimension
            return shape
        elif isinstance(node, ast.Constant) and isinstance(node.value, int):
            return [node.value]
        return None

    def _normalize_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> FunctionIR:
        """Normalize function definition."""
        # Extract signature
        signature = self._normalize_function_signature(node)

        # Normalize body
        body = []
        for stmt in node.body:
            normalized = self._normalize_statement_or_block(stmt)
            if normalized:
                body.append(normalized)

        func_ir = FunctionIR(
            signature=signature,
            body=body,
            line_number=node.lineno
        )

        # Extract tensor operations
        func_ir.extract_tensor_ops()

        return func_ir

    def _normalize_function_signature(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> FunctionSignatureIR:
        """Normalize function signature."""
        parameters = []

        # Regular arguments
        for arg in node.args.args:
            param = {
                'name': arg.arg,
                'type': self._normalize_type_annotation(arg.annotation)
            }
            parameters.append(param)

        # Return type
        return_type = self._normalize_type_annotation(node.returns)

        return FunctionSignatureIR(
            name=node.name,
            parameters=parameters,
            return_type=return_type,
            line_number=node.lineno
        )

    def _normalize_class(self, node: ast.ClassDef) -> ClassIR:
        """Normalize class definition."""
        methods = []
        attributes = []
        base_classes = []

        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                # Handle nn.Module, torch.nn.Module, etc.
                attr_chain = []
                current = base
                while isinstance(current, ast.Attribute):
                    attr_chain.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    attr_chain.append(current.id)
                attr_chain.reverse()
                base_classes.append('.'.join(attr_chain))
            else:
                base_classes.append(str(base))

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._normalize_function(item))
            elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                # Class attribute
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    attributes.append({
                        'name': item.target.id,
                        'type': self._normalize_type_annotation(item.annotation)
                    })

        return ClassIR(
            name=node.name,
            methods=methods,
            attributes=attributes,
            base_classes=base_classes,
            line_number=node.lineno
        )

    def _normalize_statement_or_block(self, node: ast.AST) -> Optional[Union[StatementIR, LoopIR, ConditionalIR, FunctionIR]]:
        """Normalize a statement or block (can be statement, loop, conditional, or nested function)."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self._normalize_function(node)
        elif isinstance(node, (ast.For, ast.While)):
            return self._normalize_loop(node)
        elif isinstance(node, ast.If):
            return self._normalize_conditional(node)
        else:
            return self._normalize_statement(node)

    def _normalize_statement(self, node: ast.AST) -> Optional[StatementIR]:
        """Normalize statement."""
        if isinstance(node, ast.Assign):
            # Assignment: x = value
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                return StatementIR(
                    stmt_type='assignment',
                    target=node.targets[0].id,
                    expression=self._normalize_expression(node.value),
                    line_number=node.lineno
                )

        elif isinstance(node, ast.AnnAssign):
            # Annotated assignment: x: int = value
            if isinstance(node.target, ast.Name):
                return StatementIR(
                    stmt_type='assignment',
                    target=node.target.id,
                    expression=self._normalize_expression(
                        node.value) if node.value else None,
                    line_number=node.lineno
                )

        elif isinstance(node, ast.Return):
            return StatementIR(
                stmt_type='return',
                expression=self._normalize_expression(node.value),
                line_number=node.lineno
            )

        elif isinstance(node, ast.Expr):
            # Expression statement
            return StatementIR(
                stmt_type='expression',
                expression=self._normalize_expression(node.value),
                line_number=node.lineno
            )

        elif isinstance(node, ast.Pass):
            return StatementIR(
                stmt_type='pass',
                line_number=node.lineno
            )

        return None

    def _normalize_loop(self, node: Union[ast.For, ast.While]) -> LoopIR:
        """Normalize loop construct."""
        if isinstance(node, ast.For):
            loop_ir = LoopIR(
                loop_type='for',
                line_number=node.lineno
            )

            # Check if it's a range() loop
            if isinstance(node.iter, ast.Call):
                call = node.iter
                if isinstance(call.func, ast.Name) and call.func.id == 'range':
                    args = call.args
                    if len(args) == 1:
                        # range(stop)
                        loop_ir.range_start = self._normalize_expression(
                            ast.Constant(value=0))
                        loop_ir.range_end = self._normalize_expression(args[0])
                    elif len(args) == 2:
                        # range(start, stop)
                        loop_ir.range_start = self._normalize_expression(
                            args[0])
                        loop_ir.range_end = self._normalize_expression(args[1])
                    elif len(args) == 3:
                        # range(start, stop, step)
                        loop_ir.range_start = self._normalize_expression(
                            args[0])
                        loop_ir.range_end = self._normalize_expression(args[1])
                        loop_ir.range_step = self._normalize_expression(
                            args[2])
                    loop_ir.iterable = self._normalize_expression(node.iter)

            # Loop variable
            if isinstance(node.target, ast.Name):
                loop_ir.variable = node.target.id

            # Body
            for stmt in node.body:
                normalized = self._normalize_statement_or_block(stmt)
                if normalized:
                    loop_ir.body.append(normalized)

            return loop_ir

        elif isinstance(node, ast.While):
            body = []
            for stmt in node.body:
                normalized = self._normalize_statement_or_block(stmt)
                if normalized:
                    body.append(normalized)
            return LoopIR(
                loop_type='while',
                condition=self._normalize_expression(node.test),
                body=body,
                line_number=node.lineno
            )

        return None

    def _normalize_conditional(self, node: ast.If) -> ConditionalIR:
        """Normalize conditional construct."""
        cond_ir = ConditionalIR(
            condition=self._normalize_expression(node.test),
            line_number=node.lineno
        )

        # Then branch
        for stmt in node.body:
            normalized = self._normalize_statement_or_block(stmt)
            if normalized:
                cond_ir.then_branch.append(normalized)

        # Else branch
        for stmt in node.orelse:
            normalized = self._normalize_statement_or_block(stmt)
            if normalized:
                cond_ir.else_branch.append(normalized)

        return cond_ir

    def _normalize_expression(self, node: Optional[ast.expr]) -> Optional[ExpressionIR]:
        """Normalize expression."""
        if node is None:
            return None

        if isinstance(node, ast.Constant):
            return ExpressionIR(
                expr_type='literal',
                value=node.value,
                line_number=getattr(node, 'lineno', None)
            )

        elif isinstance(node, ast.Name):
            return ExpressionIR(
                expr_type='variable',
                value=node.id,
                line_number=getattr(node, 'lineno', None)
            )

        elif isinstance(node, ast.BinOp):
            return ExpressionIR(
                expr_type='binary_op',
                operator=self._normalize_operator(node.op),
                left=self._normalize_expression(node.left),
                right=self._normalize_expression(node.right),
                line_number=getattr(node, 'lineno', None)
            )

        elif isinstance(node, ast.Call):
            # Check if it's a tensor operation
            tensor_op = self._detect_tensor_op(node)

            return ExpressionIR(
                expr_type='call',
                function_name=self._extract_function_name(node.func),
                arguments=[self._normalize_expression(
                    arg) for arg in node.args],
                tensor_op=tensor_op,
                line_number=getattr(node, 'lineno', None)
            )

        elif isinstance(node, ast.Attribute):
            # Attribute access: obj.attr or tensor.matmul()
            attrs = []
            current = node
            while isinstance(current, ast.Attribute):
                attrs.append(current.attr)
                current = current.value

            if isinstance(current, ast.Name):
                attrs.append(current.id)

            attrs.reverse()

            return ExpressionIR(
                expr_type='attribute',
                attributes=attrs,
                line_number=getattr(node, 'lineno', None)
            )

        elif isinstance(node, ast.Compare):
            # Comparison: a < b
            if len(node.ops) == 1 and len(node.comparators) == 1:
                return ExpressionIR(
                    expr_type='binary_op',
                    operator=self._normalize_operator(node.ops[0]),
                    left=self._normalize_expression(node.left),
                    right=self._normalize_expression(node.comparators[0]),
                    line_number=getattr(node, 'lineno', None)
                )

        elif isinstance(node, ast.BoolOp):
            # Boolean operation: a and b, a or b
            if len(node.values) == 2:
                return ExpressionIR(
                    expr_type='binary_op',
                    operator=self._normalize_operator(node.op),
                    left=self._normalize_expression(node.values[0]),
                    right=self._normalize_expression(node.values[1]),
                    line_number=getattr(node, 'lineno', None)
                )

        elif isinstance(node, ast.UnaryOp):
            # Unary operation: -x, not x
            return ExpressionIR(
                expr_type='unary_op',
                operator=self._normalize_operator(node.op),
                right=self._normalize_expression(node.operand),
                line_number=getattr(node, 'lineno', None)
            )

        elif isinstance(node, ast.Subscript):
            # Subscript: arr[i]
            return ExpressionIR(
                expr_type='subscript',
                left=self._normalize_expression(node.value),
                right=self._normalize_expression(node.slice),
                line_number=getattr(node, 'lineno', None)
            )

        elif isinstance(node, ast.List):
            # List literal
            return ExpressionIR(
                expr_type='list',
                value=[self._normalize_expression(elt) for elt in node.elts],
                line_number=getattr(node, 'lineno', None)
            )

        return ExpressionIR(
            expr_type='unknown',
            line_number=getattr(node, 'lineno', None)
        )

    def _detect_tensor_op(self, node: ast.Call) -> Optional[TensorOpKind]:
        """Detect if a function call is a tensor operation."""
        func_name = self._extract_function_name(node.func)

        if func_name is None:
            return None

        # Map common PyTorch/tensor operations
        op_map = {
            'matmul': TensorOpKind.MATMUL,
            'addmm': TensorOpKind.MATMUL,
            'bmm': TensorOpKind.MATMUL,
            'add': TensorOpKind.ADD,
            'sub': TensorOpKind.SUB,
            'mul': TensorOpKind.MUL,
            'div': TensorOpKind.DIV,
            'relu': TensorOpKind.RELU,
            'sigmoid': TensorOpKind.SIGMOID,
            'softmax': TensorOpKind.SOFTMAX,
            'conv2d': TensorOpKind.CONV2D,
            'linear': TensorOpKind.LINEAR,
            'reshape': TensorOpKind.RESHAPE,
            'view': TensorOpKind.RESHAPE,
            'transpose': TensorOpKind.TRANSPOSE,
            'permute': TensorOpKind.TRANSPOSE,
            'cat': TensorOpKind.CONCAT,
            'stack': TensorOpKind.CONCAT,
            'sum': TensorOpKind.SUM,
            'mean': TensorOpKind.MEAN,
            'max': TensorOpKind.MAX,
            'min': TensorOpKind.MIN,
        }

        # Check function name
        for key, op_kind in op_map.items():
            if key in func_name.lower():
                return op_kind

        return None

    def _extract_function_name(self, node: ast.expr) -> Optional[str]:
        """Extract function name from call node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _normalize_operator(self, op: ast.operator) -> str:
        """Normalize operator to string."""
        op_map = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.FloorDiv: '//',
            ast.Mod: '%',
            ast.Pow: '**',
            ast.Lt: '<',
            ast.LtE: '<=',
            ast.Gt: '>',
            ast.GtE: '>=',
            ast.Eq: '==',
            ast.NotEq: '!=',
            ast.And: 'and',
            ast.Or: 'or',
            ast.Not: 'not',
            ast.UAdd: '+',
            ast.USub: '-',
        }
        return op_map.get(type(op), 'unknown')
