#!/usr/bin/env python3
"""
PyTorch Model Parser
Extracts neural network architectures from PyTorch code for verification.
"""

import ast
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LayerInfo:
    """Information about a neural network layer."""
    layer_type: str  # Linear, Conv2d, ReLU, etc.
    parameters: Dict[str, Any]
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    name: str = ""


@dataclass
class PyTorchModelIR:
    """Intermediate representation for PyTorch models."""
    model_name: str
    layers: List[LayerInfo] = field(default_factory=list)
    forward_operations: List[Dict[str, Any]] = field(default_factory=list)
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    parameters_count: int = 0
    has_skip_connections: bool = False
    has_attention: bool = False


class PyTorchModelParser:
    """
    Parses PyTorch model definitions into verifiable IR.
    
    Extracts:
    - Layer architectures
    - Forward pass operations
    - Shape transformations
    - Skip connections
    - Attention mechanisms
    """
    
    def __init__(self):
        self.model_ir = None
    
    def parse_model(self, code: str) -> PyTorchModelIR:
        """
        Parse PyTorch model code.
        
        Args:
            code: Python code containing PyTorch model
            
        Returns:
            PyTorchModelIR with extracted information
        """
        ast_tree = ast.parse(code)
        
        # Find model class
        model_class = None
        for node in ast_tree.body:
            if isinstance(node, ast.ClassDef):
                # Check if it inherits from nn.Module
                for base in node.bases:
                    if isinstance(base, ast.Attribute) and base.attr == 'Module':
                        model_class = node
                        break
                    elif isinstance(base, ast.Name) and base.id == 'Module':
                        model_class = node
                        break
        
        if not model_class:
            raise ValueError("No PyTorch model class found")
        
        # Extract model info
        model_name = model_class.name
        self.model_ir = PyTorchModelIR(model_name=model_name)
        
        # Parse __init__ for layers
        for item in model_class.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                self._parse_init(item)
            elif isinstance(item, ast.FunctionDef) and item.name == 'forward':
                self._parse_forward(item)
        
        return self.model_ir
    
    def _parse_init(self, init_node: ast.FunctionDef):
        """Parse __init__ method to extract layers."""
        for stmt in init_node.body:
            # Look for self.layer = nn.Linear(...) patterns
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute) and target.value.id == 'self':
                        layer_name = target.attr
                        layer_info = self._extract_layer(stmt.value, layer_name)
                        if layer_info:
                            self.model_ir.layers.append(layer_info)
    
    def _extract_layer(self, node: ast.Call, name: str) -> Optional[LayerInfo]:
        """Extract layer information from AST node."""
        if not isinstance(node, ast.Call):
            return None
        
        # Get layer type
        if isinstance(node.func, ast.Attribute):
            layer_type = node.func.attr
        elif isinstance(node.func, ast.Name):
            layer_type = node.func.id
        else:
            return None
        
        # Extract parameters
        params = {}
        for i, arg in enumerate(node.args):
            if isinstance(arg, ast.Constant):
                params[f'arg_{i}'] = arg.value
            elif isinstance(arg, ast.Name):
                params[f'arg_{i}'] = arg.id
        
        for keyword in node.keywords:
            if isinstance(keyword.value, ast.Constant):
                params[keyword.arg] = keyword.value.value
            elif isinstance(keyword.value, ast.Name):
                params[keyword.arg] = keyword.value.id
        
        return LayerInfo(
            layer_type=layer_type,
            parameters=params,
            name=name
        )
    
    def _parse_forward(self, forward_node: ast.FunctionDef):
        """Parse forward method to extract operations."""
        for stmt in forward_node.body:
            op = self._extract_operation(stmt)
            if op:
                self.model_ir.forward_operations.append(op)
    
    def _extract_operation(self, stmt: ast.stmt) -> Optional[Dict[str, Any]]:
        """Extract forward pass operation."""
        if isinstance(stmt, ast.Return):
            return {'type': 'return', 'value': self._ast_to_string(stmt.value)}
        elif isinstance(stmt, ast.Assign):
            return {
                'type': 'assignment',
                'target': self._ast_to_string(stmt.targets[0]),
                'value': self._ast_to_string(stmt.value)
            }
        return None
    
    def _ast_to_string(self, node: ast.AST) -> str:
        """Convert AST node to string representation."""
        if isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._ast_to_string(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            func = self._ast_to_string(node.func)
            args = ', '.join(self._ast_to_string(arg) for arg in node.args)
            return f"{func}({args})"
        elif isinstance(node, ast.BinOp):
            left = self._ast_to_string(node.left)
            right = self._ast_to_string(node.right)
            if isinstance(node.op, ast.Add):
                return f"{left} + {right}"
            elif isinstance(node.op, ast.Mult):
                return f"{left} * {right}"
        return "?"
    
    def compute_shapes(self, input_shape: List[int]):
        """
        Compute output shapes for all layers.
        
        Args:
            input_shape: Input tensor shape [batch, channels, ...]
        """
        self.model_ir.input_shape = input_shape
        current_shape = input_shape
        
        for layer in self.model_ir.layers:
            # Compute output shape based on layer type
            if layer.layer_type == 'Linear':
                in_features = layer.parameters.get('in_features', layer.parameters.get('arg_0'))
                out_features = layer.parameters.get('out_features', layer.parameters.get('arg_1'))
                
                if out_features:
                    # [batch, ..., in_features] -> [batch, ..., out_features]
                    current_shape = current_shape[:-1] + [out_features]
            
            elif layer.layer_type == 'Conv2d':
                out_channels = layer.parameters.get('out_channels', layer.parameters.get('arg_1'))
                kernel_size = layer.parameters.get('kernel_size', layer.parameters.get('arg_2', 3))
                padding = layer.parameters.get('padding', 0)
                
                if out_channels:
                    # Simple shape computation (assuming stride=1)
                    if isinstance(kernel_size, int):
                        k = kernel_size
                    else:
                        k = 3
                    
                    if isinstance(padding, int):
                        p = padding
                    else:
                        p = 0
                    
                    # [batch, in_channels, H, W] -> [batch, out_channels, H, W]
                    current_shape = [current_shape[0], out_channels, current_shape[2], current_shape[3]]
            
            elif layer.layer_type in ['ReLU', 'Tanh', 'Sigmoid', 'Softmax']:
                # Activation functions don't change shape
                pass
            
            layer.input_shape = layer.output_shape
            layer.output_shape = current_shape[:]
        
        self.model_ir.output_shape = current_shape
    
    def count_parameters(self):
        """Count total trainable parameters."""
        total = 0
        
        for layer in self.model_ir.layers:
            if layer.layer_type == 'Linear':
                in_features = layer.parameters.get('in_features', layer.parameters.get('arg_0', 0))
                out_features = layer.parameters.get('out_features', layer.parameters.get('arg_1', 0))
                
                if isinstance(in_features, int) and isinstance(out_features, int):
                    # weight + bias
                    total += in_features * out_features + out_features
            
            elif layer.layer_type == 'Conv2d':
                in_channels = layer.parameters.get('in_channels', layer.parameters.get('arg_0', 0))
                out_channels = layer.parameters.get('out_channels', layer.parameters.get('arg_1', 0))
                kernel_size = layer.parameters.get('kernel_size', layer.parameters.get('arg_2', 3))
                
                if isinstance(in_channels, int) and isinstance(out_channels, int):
                    if isinstance(kernel_size, int):
                        k = kernel_size
                    else:
                        k = 3
                    
                    # weight + bias
                    total += in_channels * out_channels * k * k + out_channels
        
        self.model_ir.parameters_count = total
    
    def detect_architecture_patterns(self):
        """Detect common architecture patterns."""
        layer_types = [l.layer_type for l in self.model_ir.layers]
        
        # Detect skip connections (ResNet-style)
        for op in self.model_ir.forward_operations:
            if '+' in op.get('value', ''):
                self.model_ir.has_skip_connections = True
                break
        
        # Detect attention mechanisms
        for layer in self.model_ir.layers:
            if 'Attention' in layer.layer_type or 'Multihead' in layer.layer_type:
                self.model_ir.has_attention = True
                break


def parse_pytorch_file(filepath: str) -> PyTorchModelIR:
    """
    Parse PyTorch model from file.
    
    Args:
        filepath: Path to Python file
        
    Returns:
        PyTorchModelIR
    """
    with open(filepath, 'r') as f:
        code = f.read()
    
    parser = PyTorchModelParser()
    model_ir = parser.parse_model(code)
    
    # Compute shapes (assuming batch size 1)
    if model_ir.input_shape:
        parser.compute_shapes(model_ir.input_shape)
    
    parser.count_parameters()
    parser.detect_architecture_patterns()
    
    return model_ir


if __name__ == "__main__":
    # Example usage
    example_code = '''
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
'''
    
    parser = PyTorchModelParser()
    model_ir = parser.parse_model(example_code)
    
    print(f"Model: {model_ir.model_name}")
    print(f"Layers: {len(model_ir.layers)}")
    
    for layer in model_ir.layers:
        print(f"  {layer.name}: {layer.layer_type} {layer.parameters}")
    
    # Compute shapes
    parser.compute_shapes([1, 784])
    parser.count_parameters()
    
    print(f"\nInput shape: {model_ir.input_shape}")
    print(f"Output shape: {model_ir.output_shape}")
    print(f"Parameters: {model_ir.parameters_count:,}")
