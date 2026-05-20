"""
Main abstract interpreter that orchestrates type inference and shape analysis.
Analyzes normalized IR and produces abstract state with inferred types, shapes, and facts.
"""

from typing import Dict, List, Optional, Any
from .abstract_domain import AbstractState, AbstractValue, TypeDomain, TensorShape
from .type_inference import TypeInferenceEngine
from .shape_analysis import TensorShapeAnalyzer
from ast_extractor.ir import (
    NormalizedIR,
    FunctionIR,
    ClassIR,
    LoopIR,
    ConditionalIR,
    StatementIR,
    ExpressionIR,
    TensorOpKind,
)


class AbstractInterpreter:
    """
    Main abstract interpreter for Axiom Zero.
    Orchestrates type inference and shape analysis over normalized IR.
    """

    def __init__(self):
        """Initialize abstract interpreter."""
        self.type_engine = TypeInferenceEngine()
        self.shape_analyzer = TensorShapeAnalyzer()

    def analyze(self, normalized_ir: NormalizedIR) -> AbstractState:
        """
        Run complete abstract interpretation over normalized IR.

        Args:
            normalized_ir: Normalized IR from AST extraction

        Returns:
            AbstractState with inferred types, shapes, and facts
        """
        state = AbstractState()

        # Phase 1: Global analysis
        self._analyze_globals(normalized_ir, state)

        # Phase 2: Function-level analysis
        self._analyze_functions(normalized_ir, state)

        # Phase 3: Class-level analysis
        self._analyze_classes(normalized_ir, state)

        # Phase 4: Extract and consolidate facts
        self._extract_facts(normalized_ir, state)

        state.analysis_complete = True

        return state

    def _analyze_globals(self, ir: NormalizedIR, state: AbstractState):
        """Analyze global statements and imports."""
        # Process imports for type information
        for imp in ir.imports:
            if 'torch' in imp:
                state.add_type_constraint("torch.tensors_available == True")
            if 'typing' in imp:
                state.add_type_constraint("typing.annotations_enabled == True")

        # Analyze global statements
        for stmt in ir.global_statements:
            if stmt.stmt_type == 'assignment' and stmt.target and stmt.expression:
                # TODO: Infer type from expression for global variables
                pass

    def _analyze_functions(self, ir: NormalizedIR, state: AbstractState):
        """Analyze all functions in the IR."""
        for func in ir.functions:
            self._analyze_single_function(func, state)

    def _analyze_single_function(self, func: FunctionIR, state: AbstractState):
        """
        Analyze a single function for types and shapes.

        Args:
            func: FunctionIR to analyze
            state: Abstract state to update
        """
        func_name = func.signature.name

        # Run type inference
        local_env = self.type_engine.infer_function_types(
            func, state.global_env.copy())
        state.function_envs[func_name] = local_env

        # Store function signature with inferred types
        sig_info = {
            'name': func_name,
            'parameters': [],
            'return_type': None,
            'has_tensor_ops': len(func.tensor_operations) > 0
        }

        for param in func.signature.parameters:
            param_info = {'name': param['name']}
            if param['type']:
                param_info['type'] = param['type'].to_string()
                param_info['abstract_type'] = self.type_engine._type_annotation_to_abstract(
                    param['type'])
            else:
                # Use inferred type
                if param['name'] in local_env:
                    param_info['inferred_type'] = str(local_env[param['name']])
            sig_info['parameters'].append(param_info)

        if func.signature.return_type:
            sig_info['return_type'] = func.signature.return_type.to_string()

        state.function_signatures[func_name] = sig_info

        # Analyze tensor operations in function
        for tensor_op in func.tensor_operations:
            self._analyze_tensor_op(func, tensor_op, local_env, state)

        # Extract shape facts from local variables
        for var_name, abs_value in local_env.items():
            if abs_value.is_tensor() and abs_value.has_shape():
                shape_facts = self.shape_analyzer.extract_shape_facts(
                    abs_value.tensor_shape, var_name
                )
                for fact in shape_facts:
                    state.add_shape_fact(fact)

    def _analyze_tensor_op(self, func: FunctionIR, tensor_op,
                           local_env: Dict[str, AbstractValue], state: AbstractState):
        """
        Analyze a specific tensor operation for shape constraints.

        Args:
            func: Containing function
            tensor_op: Tensor operation kind
            local_env: Local type environment
            state: Abstract state
        """
        # Find the statement containing this tensor op
        op_stmt = self._find_tensor_op_statement(func.body, tensor_op)
        if not op_stmt or not op_stmt.expression:
            return

        expr = op_stmt.expression

        # Analyze based on operation type
        if tensor_op == TensorOpKind.MATMUL:
            self._analyze_matmul_op(expr, local_env, state)
        elif tensor_op == TensorOpKind.CONV2D:
            self._analyze_conv2d_op(expr, local_env, state)
        elif tensor_op == TensorOpKind.LINEAR:
            self._analyze_linear_op(expr, local_env, state)
        elif tensor_op in [TensorOpKind.ADD, TensorOpKind.MUL, TensorOpKind.SUB, TensorOpKind.DIV]:
            self._analyze_elementwise_op(expr, local_env, state)
        elif tensor_op == TensorOpKind.RESHAPE:
            self._analyze_reshape_op(expr, local_env, state)
        elif tensor_op == TensorOpKind.CONCAT:
            self._analyze_concat_op(expr, local_env, state)

    def _find_tensor_op_statement(self, body, tensor_op) -> Optional[StatementIR]:
        """Find statement containing a specific tensor operation."""
        def search(nodes):
            for node in nodes:
                if isinstance(node, StatementIR) and node.expression:
                    if node.expression.tensor_op == tensor_op:
                        return node
                elif isinstance(node, LoopIR):
                    result = search(node.body)
                    if result:
                        return result
                elif isinstance(node, ConditionalIR):
                    result = search(node.then_branch + node.else_branch)
                    if result:
                        return result
                elif isinstance(node, FunctionIR):
                    result = search(node.body)
                    if result:
                        return result
            return None

        return search(body)

    def _analyze_matmul_op(self, expr: ExpressionIR, env: Dict[str, AbstractValue],
                           state: AbstractState):
        """Analyze matrix multiplication operation."""
        if len(expr.arguments) >= 2:
            arg_a = self._get_value_from_expr(expr.arguments[0], env)
            arg_b = self._get_value_from_expr(expr.arguments[1], env)

            if arg_a and arg_a.is_tensor() and arg_b and arg_b.is_tensor():
                if arg_a.has_shape() and arg_b.has_shape():
                    shape_a = arg_a.tensor_shape
                    shape_b = arg_b.tensor_shape

                    # Add constraint: inner dimensions must match
                    if len(shape_a.dimensions) >= 2 and len(shape_b.dimensions) >= 2:
                        inner_a = shape_a.dimensions[-1]
                        inner_b = shape_b.dimensions[-2]

                        if inner_a.is_symbolic() and inner_b.is_symbolic():
                            state.add_shape_fact(
                                f"{inner_a.symbolic} == {inner_b.symbolic}")
                        elif inner_a.is_concrete() and inner_b.is_concrete():
                            if inner_a.value != inner_b.value:
                                state.warnings.append(
                                    f"Shape mismatch in matmul: {shape_a} @ {shape_b}"
                                )

    def _analyze_conv2d_op(self, expr: ExpressionIR, env: Dict[str, AbstractValue],
                           state: AbstractState):
        """Analyze convolution operation."""
        if len(expr.arguments) >= 2:
            input_val = self._get_value_from_expr(expr.arguments[0], env)
            weight_val = self._get_value_from_expr(expr.arguments[1], env)

            if input_val and input_val.is_tensor() and weight_val and weight_val.is_tensor():
                if input_val.has_shape() and weight_val.has_shape():
                    # Conv2D: input [N, C_in, H, W], weight [C_out, C_in, kH, kW]
                    state.add_shape_fact("conv2d_input_rank == 4")
                    state.add_shape_fact("conv2d_weight_rank == 4")

    def _analyze_linear_op(self, expr: ExpressionIR, env: Dict[str, AbstractValue],
                           state: AbstractState):
        """Analyze linear layer operation."""
        if len(expr.arguments) >= 2:
            input_val = self._get_value_from_expr(expr.arguments[0], env)
            weight_val = self._get_value_from_expr(expr.arguments[1], env)

            if input_val and input_val.is_tensor() and weight_val and weight_val.is_tensor():
                state.add_shape_fact("linear_weight_rank == 2")

    def _analyze_elementwise_op(self, expr: ExpressionIR, env: Dict[str, AbstractValue],
                                state: AbstractState):
        """Analyze elementwise operation (broadcasting constraints)."""
        if len(expr.arguments) >= 2:
            arg_a = self._get_value_from_expr(expr.arguments[0], env)
            arg_b = self._get_value_from_expr(expr.arguments[1], env)

            if arg_a and arg_b:
                if arg_a.is_tensor() and arg_b.is_tensor():
                    if arg_a.has_shape() and arg_b.has_shape():
                        # TODO: Implement proper broadcasting constraint check
                        state.add_shape_fact(
                            "elementwise_broadcast_compatible == True")

    def _analyze_reshape_op(self, expr: ExpressionIR, env: Dict[str, AbstractValue],
                            state: AbstractState):
        """Analyze reshape operation."""
        if len(expr.arguments) >= 2:
            input_val = self._get_value_from_expr(expr.arguments[0], env)

            if input_val and input_val.is_tensor() and input_val.has_shape():
                # TODO: Validate element count preservation with concrete shapes
                state.add_shape_fact("reshape_element_count_preserved == True")

    def _analyze_concat_op(self, expr: ExpressionIR, env: Dict[str, AbstractValue],
                           state: AbstractState):
        """Analyze concatenation operation."""
        state.add_shape_fact("concat_same_rank_required == True")
        state.add_shape_fact("concat_dims_match_except_axis == True")

    def _get_value_from_expr(self, expr: ExpressionIR,
                             env: Dict[str, AbstractValue]) -> Optional[AbstractValue]:
        """Get abstract value from expression."""
        if expr.expr_type == 'variable':
            return env.get(expr.value)
        elif expr.expr_type == 'literal':
            return AbstractValue.constant(expr.value, TypeDomain.BOTTOM)
        return None

    def _analyze_classes(self, ir: NormalizedIR, state: AbstractState):
        """Analyze classes in the IR."""
        for cls in ir.classes:
            # Analyze class methods
            for method in cls.methods:
                self._analyze_single_function(method, state)

            # Extract class attributes
            for attr in cls.attributes:
                if attr.get('type'):
                    state.add_type_constraint(
                        f"{cls.name}.{attr['name']}: {attr['type'].to_string()}"
                    )

    def _extract_facts(self, ir: NormalizedIR, state: AbstractState):
        """Extract additional facts from IR structure."""
        # Loop bounds
        for func in ir.functions:
            loop_facts = self._extract_loop_facts(func.body)
            state.shape_facts.extend(loop_facts)

    def _extract_loop_facts(self, body) -> List[str]:
        """Extract facts from loop constructs."""
        facts = []

        for node in body:
            if isinstance(node, LoopIR):
                if node.is_bounded() and node.range_start and node.range_end:
                    # Bounded loop
                    facts.append(f"loop_{node.variable}_is_bounded == True")

                    # Could extract concrete bounds if known
                    if node.range_start.expr_type == 'literal':
                        start = node.range_start.value
                        facts.append(f"loop_{node.variable}_start == {start}")
                    if node.range_end.expr_type == 'literal':
                        end = node.range_end.value
                        facts.append(f"loop_{node.variable}_end == {end}")

            elif isinstance(node, (FunctionIR,)):
                facts.extend(self._extract_loop_facts(node.body))
            elif isinstance(node, ConditionalIR):
                facts.extend(self._extract_loop_facts(node.then_branch))
                facts.extend(self._extract_loop_facts(node.else_branch))

        return facts
