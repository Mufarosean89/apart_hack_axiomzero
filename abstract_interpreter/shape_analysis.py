"""
Tensor shape analyzer for symbolic shape inference.
Tracks tensor shapes through operations and propagates constraints.
"""

from typing import Dict, List, Optional, Tuple
from .abstract_domain import TensorShape, ShapeDimension, AbstractValue, TypeDomain


class TensorShapeAnalyzer:
    """
    Analyzes tensor operations and infers output shapes symbolically.
    Handles common PyTorch operations with shape propagation rules.
    """

    def __init__(self):
        """Initialize shape analyzer."""
        self.shape_constraints = []

    def infer_matmul_shape(self, shape_a: TensorShape, shape_b: TensorShape) -> Optional[TensorShape]:
        """
        Infer output shape of matrix multiplication.

        For 2D tensors: [M, K] @ [K, N] -> [M, N]
        For batched: [B, M, K] @ [B, K, N] -> [B, M, N]

        Args:
            shape_a: Shape of first tensor
            shape_b: Shape of second tensor

        Returns:
            Inferred output shape
        """
        if not shape_a.dimensions or not shape_b.dimensions:
            return TensorShape.unknown()

        rank_a = len(shape_a.dimensions)
        rank_b = len(shape_b.dimensions)

        if rank_a == 2 and rank_b == 2:
            inner_a = shape_a.dimensions[1]
            inner_b = shape_b.dimensions[0]

            if not self._dimensions_compatible(inner_a, inner_b):
                self.shape_constraints.append(
                    f"Shape mismatch: {shape_a} @ {shape_b} - inner dimensions must match"
                )
                return None

            return TensorShape(dimensions=[
                shape_a.dimensions[0],
                shape_b.dimensions[1]
            ])

        elif rank_a >= 2 and rank_b >= 2:
            batch_dims_a = shape_a.dimensions[:-2]
            batch_dims_b = shape_b.dimensions[:-2]

            batch_dims = self._infer_broadcast_dims(batch_dims_a, batch_dims_b)
            if batch_dims is None:
                return None

            output_dims = batch_dims + [
                shape_a.dimensions[-2],
                shape_b.dimensions[-1]
            ]

            return TensorShape(dimensions=output_dims)

        return TensorShape.unknown()

    def infer_linear_shape(self, input_shape: TensorShape, weight_shape: TensorShape) -> Optional[TensorShape]:
        """
        Infer output shape of linear layer.

        Linear: [*, in_features] @ [out_features, in_features].T -> [*, out_features]

        Args:
            input_shape: Input tensor shape
            weight_shape: Weight matrix shape

        Returns:
            Output tensor shape
        """
        if len(input_shape.dimensions) < 1 or len(weight_shape.dimensions) != 2:
            return TensorShape.unknown()

        # Last dimension of input should match second dimension of weight
        in_features_input = input_shape.dimensions[-1]
        in_features_weight = weight_shape.dimensions[1]

        if not self._dimensions_compatible(in_features_input, in_features_weight):
            self.shape_constraints.append(
                f"Linear shape mismatch: input {input_shape}, weight {weight_shape}"
            )
            return None

        # Output: [*batch_dims, out_features]
        batch_dims = input_shape.dimensions[:-1]
        out_features = weight_shape.dimensions[0]

        return TensorShape(dimensions=batch_dims + [out_features])

    def infer_conv2d_shape(self, input_shape: TensorShape, weight_shape: TensorShape,
                           stride: int = 1, padding: int = 0) -> Optional[TensorShape]:
        """
        Infer output shape of 2D convolution.

        Output: [N, C_out, H_out, W_out]
        where H_out = floor((H_in + 2*padding - kernel_h) / stride) + 1

        Args:
            input_shape: Input shape [N, C_in, H, W]
            weight_shape: Weight shape [C_out, C_in, kH, kW]
            stride: Convolution stride
            padding: Padding size

        Returns:
            Output tensor shape
        """
        if len(input_shape.dimensions) != 4 or len(weight_shape.dimensions) != 4:
            return TensorShape.unknown()

        N = input_shape.dimensions[0]
        H_in = input_shape.dimensions[2]
        W_in = input_shape.dimensions[3]

        C_out = weight_shape.dimensions[0]
        kH = weight_shape.dimensions[2]
        kW = weight_shape.dimensions[3]

        # Calculate output spatial dimensions
        if H_in.is_concrete() and kH.is_concrete():
            H_out = ShapeDimension(
                value=(H_in.value + 2 * padding - kH.value) // stride + 1)
        else:
            H_out = ShapeDimension(symbolic="H_out")

        if W_in.is_concrete() and kW.is_concrete():
            W_out = ShapeDimension(
                value=(W_in.value + 2 * padding - kW.value) // stride + 1)
        else:
            W_out = ShapeDimension(symbolic="W_out")

        return TensorShape(dimensions=[N, C_out, H_out, W_out])

    def infer_elementwise_shape(self, shape_a: TensorShape, shape_b: TensorShape) -> Optional[TensorShape]:
        """
        Infer output shape of elementwise operations (add, mul, sub, etc.).

        Uses broadcasting rules.

        Args:
            shape_a: First operand shape
            shape_b: Second operand shape

        Returns:
            Output shape after broadcasting
        """
        broadcast_dims = self._infer_broadcast_dims(shape_a.dimensions, shape_b.dimensions)
        if broadcast_dims is None:
            return None
        return TensorShape(dimensions=broadcast_dims)

    def infer_concat_shape(self, shapes: List[TensorShape], dim: int = 0) -> Optional[TensorShape]:
        """
        Infer output shape of concatenation.

        Args:
            shapes: List of tensor shapes to concatenate
            dim: Concatenation dimension

        Returns:
            Output tensor shape
        """
        if not shapes:
            return TensorShape.unknown()

        # All shapes must have same rank
        rank = len(shapes[0].dimensions)
        if not all(len(s.dimensions) == rank for s in shapes):
            self.shape_constraints.append(
                "Concat: all tensors must have same rank")
            return None

        # All dimensions except concat dim must match
        output_dims = []
        for d in range(rank):
            if d == dim:
                # Sum dimensions along concat axis
                if all(s.dimensions[d].is_concrete() for s in shapes):
                    total = sum(s.dimensions[d].value for s in shapes)
                    output_dims.append(ShapeDimension(value=total))
                else:
                    output_dims.append(ShapeDimension(symbolic="sum_dims"))
            else:
                # Check other dimensions match
                first_dim = shapes[0].dimensions[d]
                if all(s.dimensions[d] == first_dim for s in shapes[1:]):
                    output_dims.append(first_dim)
                else:
                    self.shape_constraints.append(
                        f"Concat: dimension {d} mismatch across tensors"
                    )
                    return None

        return TensorShape(dimensions=output_dims)

    def infer_reshape_shape(self, input_shape: TensorShape,
                            new_shape: List[int]) -> Optional[TensorShape]:
        """
        Infer output shape of reshape operation.

        Args:
            input_shape: Input tensor shape
            new_shape: Target shape (can include -1 for inference)

        Returns:
            Output tensor shape
        """
        if not input_shape.is_fully_known():
            # Can't fully infer if input shape unknown
            return TensorShape.from_list([
                dim if dim != -1 else "?" for dim in new_shape
            ])

        # Calculate total elements
        total_elements = 1
        for dim in input_shape.dimensions:
            if dim.is_concrete():
                total_elements *= dim.value
            else:
                return TensorShape.from_list(new_shape)  # Can't infer

        # Handle -1 dimension
        output_dims = []
        inferred_dim = -1
        known_product = 1

        for i, dim in enumerate(new_shape):
            if dim == -1:
                inferred_dim = i
            else:
                output_dims.append(ShapeDimension(value=dim))
                known_product *= dim

        if inferred_dim >= 0:
            inferred_value = total_elements // known_product
            output_dims.insert(
                inferred_dim, ShapeDimension(value=inferred_value))

        return TensorShape(dimensions=output_dims)

    def infer_transpose_shape(self, input_shape: TensorShape,
                              dim0: int, dim1: int) -> Optional[TensorShape]:
        """
        Infer output shape of transpose operation.

        Args:
            input_shape: Input tensor shape
            dim0: First dimension to swap
            dim1: Second dimension to swap

        Returns:
            Transposed tensor shape
        """
        dims = input_shape.dimensions.copy()
        if 0 <= dim0 < len(dims) and 0 <= dim1 < len(dims):
            dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
            return TensorShape(dimensions=dims)
        return TensorShape.unknown()

    def _infer_broadcast_shape(self, dims_a: List[ShapeDimension],
                               dims_b: List[ShapeDimension]) -> Optional[TensorShape]:
        """
        Infer broadcasted shape from two dimension lists.

        Args:
            dims_a: Dimensions of first tensor
            dims_b: Dimensions of second tensor

        Returns:
            Broadcasted shape
        """
        # Align shapes from right
        max_rank = max(len(dims_a), len(dims_b))
        dims_a_padded = [ShapeDimension(
            value=1)] * (max_rank - len(dims_a)) + dims_a
        dims_b_padded = [ShapeDimension(
            value=1)] * (max_rank - len(dims_b)) + dims_b

        output_dims = []
        for d1, d2 in zip(dims_a_padded, dims_b_padded):
            if d1 == d2:
                output_dims.append(d1)
            elif d1.value == 1:
                output_dims.append(d2)
            elif d2.value == 1:
                output_dims.append(d1)
            elif d1.is_symbolic() or d2.is_symbolic():
                # Symbolic dimensions can potentially broadcast
                output_dims.append(d1 if d1.is_symbolic() else d2)
            else:
                self.shape_constraints.append(
                    f"Broadcast error: {d1} vs {d2}"
                )
                return None

        return TensorShape(dimensions=output_dims)

    def _infer_broadcast_dims(self, dims_a: List[ShapeDimension],
                              dims_b: List[ShapeDimension]) -> Optional[List[ShapeDimension]]:
        """
        Infer broadcasted dimensions from two dimension lists.

        Returns:
            List of broadcasted dimensions, or None if incompatible
        """
        max_rank = max(len(dims_a), len(dims_b))
        dims_a_padded = [ShapeDimension(value=1)] * (max_rank - len(dims_a)) + dims_a
        dims_b_padded = [ShapeDimension(value=1)] * (max_rank - len(dims_b)) + dims_b

        output_dims = []
        for d1, d2 in zip(dims_a_padded, dims_b_padded):
            if d1 == d2:
                output_dims.append(d1)
            elif d1.value == 1:
                output_dims.append(d2)
            elif d2.value == 1:
                output_dims.append(d1)
            elif d1.is_symbolic() or d2.is_symbolic():
                output_dims.append(d1 if d1.is_symbolic() else d2)
            else:
                self.shape_constraints.append(
                    f"Broadcast error: {d1} vs {d2}"
                )
                return None

        return output_dims

    def _dimensions_compatible(self, dim1: ShapeDimension, dim2: ShapeDimension) -> bool:
        """Check if two dimensions are compatible."""
        if dim1.is_concrete() and dim2.is_concrete():
            return dim1.value == dim2.value
        return True  # If either is symbolic, assume compatible

    def extract_shape_facts(self, shape: TensorShape, var_name: str) -> List[str]:
        """
        Extract shape constraints as logical facts.

        Args:
            shape: Tensor shape
            var_name: Variable name

        Returns:
            List of shape constraint strings
        """
        facts = []

        for i, dim in enumerate(shape.dimensions):
            if dim.is_concrete():
                facts.append(f"{var_name}.shape[{i}] == {dim.value}")
            elif dim.is_symbolic():
                facts.append(f"{var_name}.shape[{i}] == {dim.symbolic}")
                facts.append(f"{dim.symbolic} > 0")  # Dimensions are positive

        # Rank constraint
        if shape.rank is not None:
            facts.append(f"rank({var_name}) == {shape.rank}")

        return facts
