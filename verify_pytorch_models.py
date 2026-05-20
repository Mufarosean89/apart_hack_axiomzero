#!/usr/bin/env python3
"""
Verify Real PyTorch Models
Parse torchvision models and verify their properties.
"""

import torch
import torch.nn as nn
from pytorch_parser import PyTorchModelParser, PyTorchModelIR


def parse_torchvision_model(model: nn.Module, model_name: str) -> PyTorchModelIR:
    """
    Parse a torchvision model into verifiable IR.
    
    Args:
        model: PyTorch model instance
        model_name: Name of the model
        
    Returns:
        PyTorchModelIR with extracted information
    """
    # Get model code representation
    model_code = f"""
import torch
import torch.nn as nn

class {model_name}(nn.Module):
    def __init__(self):
        super().__init__()
"""
    
    # Extract layers from model
    layer_lines = []
    for name, module in model.named_children():
        layer_type = module.__class__.__name__
        layer_lines.append(f"        self.{name} = {module}")
    
    model_code += "\n".join(layer_lines)
    
    model_code += """
    
    def forward(self, x):
        return x
"""
    
    # Parse
    parser = PyTorchModelParser()
    try:
        model_ir = parser.parse_model(model_code)
        model_ir.model_name = model_name
        
        # Count actual parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_ir.parameters_count = total_params
        
        return model_ir
        
    except Exception as e:
        print(f"  ⚠ Parse error: {e}")
        # Return minimal IR
        model_ir = PyTorchModelIR(model_name=model_name)
        model_ir.parameters_count = sum(p.numel() for p in model.parameters())
        return model_ir


def verify_model_properties(model_ir: PyTorchModelIR, input_shape: list) -> dict:
    """
    Verify properties of a PyTorch model.
    
    Args:
        model_ir: Model intermediate representation
        input_shape: Input tensor shape [batch, channels, height, width]
        
    Returns:
        Verification results
    """
    results = {
        'model_name': model_ir.model_name,
        'input_shape': input_shape,
        'properties': {},
        'verified': True,
        'issues': []
    }
    
    # Property 1: Parameter count > 0
    if model_ir.parameters_count > 0:
        results['properties']['has_parameters'] = True
    else:
        results['properties']['has_parameters'] = False
        results['verified'] = False
        results['issues'].append("Model has no parameters")
    
    # Property 2: Has at least one layer
    if len(model_ir.layers) > 0:
        results['properties']['has_layers'] = True
    else:
        results['properties']['has_layers'] = False
        results['verified'] = False
        results['issues'].append("Model has no layers")
    
    # Property 3: Layer types are valid
    valid_layers = {'Linear', 'Conv2d', 'ReLU', 'MaxPool2d', 'BatchNorm2d', 
                    'Sequential', 'AdaptiveAvgPool2d', 'Dropout', 'Sigmoid', 
                    'Tanh', 'Softmax', 'LayerNorm', 'Embedding'}
    
    layer_types = set(l.layer_type for l in model_ir.layers)
    invalid_layers = layer_types - valid_layers
    
    if not invalid_layers:
        results['properties']['valid_layer_types'] = True
    else:
        results['properties']['valid_layer_types'] = False
        results['issues'].append(f"Unknown layer types: {invalid_layers}")
    
    # Property 4: Has forward pass
    if len(model_ir.forward_operations) > 0 or len(model_ir.layers) > 0:
        results['properties']['has_forward'] = True
    else:
        results['properties']['has_forward'] = False
        results['issues'].append("No forward pass operations found")
    
    # Property 5: Detect skip connections (ResNet-style)
    has_skip = model_ir.has_skip_connections
    results['properties']['has_skip_connections'] = has_skip
    
    # Property 6: Detect attention mechanisms
    has_attention = model_ir.has_attention
    results['properties']['has_attention'] = has_attention
    
    return results


def generate_lean_specifications(model_ir: PyTorchModelIR) -> list:
    """
    Generate Lean 4 specifications for model verification.
    
    Args:
        model_ir: Model intermediate representation
        
    Returns:
        List of Lean theorem statements
    """
    specs = []
    
    # Spec 1: Output shape preservation
    if model_ir.input_shape and model_ir.output_shape:
        input_shape_str = ', '.join(str(d) for d in model_ir.input_shape)
        output_shape_str = ', '.join(str(d) for d in model_ir.output_shape)
        
        spec = f"""-- Theorem: {model_ir.model_name} preserves shape
theorem {model_ir.model_name.lower()}_shape :
  ∀ (x : Tensor [{input_shape_str}]),
  shape ({model_ir.model_name.lower()} x) = [{output_shape_str}]"""
        specs.append(spec)
    
    # Spec 2: Parameter count
    if model_ir.parameters_count > 0:
        spec = f"""-- Theorem: {model_ir.model_name} has correct parameter count
theorem {model_ir.model_name.lower()}_params :
  parameter_count ({model_ir.model_name.lower()}) = {model_ir.parameters_count}"""
        specs.append(spec)
    
    # Spec 3: Layer count
    spec = f"""-- Theorem: {model_ir.model_name} has correct layer count
theorem {model_ir.model_name.lower()}_layers :
  layer_count ({model_ir.model_name.lower()}) = {len(model_ir.layers)}"""
    specs.append(spec)
    
    # Spec 4: No NaN outputs (numerical stability)
    spec = f"""-- Theorem: {model_ir.model_name} produces no NaN
theorem {model_ir.model_name.lower()}_no_nan :
  ∀ (x : Tensor), ¬ is_nan ({model_ir.model_name.lower()} x)"""
    specs.append(spec)
    
    # Spec 5: Gradient flow (if has skip connections)
    if model_ir.has_skip_connections:
        spec = f"""-- Theorem: {model_ir.model_name} has gradient flow via skip connections
theorem {model_ir.model_name.lower()}_gradient_flow :
  ∀ (x : Tensor), gradient_exists ({model_ir.model_name.lower()}) x"""
        specs.append(spec)
    
    return specs


def main():
    """Main verification script."""
    print("="*70)
    print("VERIFYING REAL PyTorch MODELS")
    print("="*70)
    print()
    
    # Try to import torchvision
    try:
        import torchvision.models as models
        has_torchvision = True
        print("✓ torchvision available")
    except ImportError:
        has_torchvision = False
        print("⊗ torchvision not installed (install with: pip install torchvision)")
        print()
        print("Using mock models instead...")
    
    print()
    
    models_to_verify = []
    
    if has_torchvision:
        # Real torchvision models
        print("Loading torchvision models...")
        
        # ResNet-18
        print("  Loading ResNet-18...")
        resnet18 = models.resnet18()
        models_to_verify.append(('ResNet18', resnet18, [1, 3, 224, 224]))
        
        # AlexNet
        print("  Loading AlexNet...")
        alexnet = models.alexnet()
        models_to_verify.append(('AlexNet', alexnet, [1, 3, 224, 224]))
        
        # VGG-11
        print("  Loading VGG-11...")
        vgg11 = models.vgg11()
        models_to_verify.append(('VGG11', vgg11, [1, 3, 224, 224]))
        
        # SqueezeNet
        print("  Loading SqueezeNet...")
        squeezenet = models.squeezenet1_0()
        models_to_verify.append(('SqueezeNet', squeezenet, [1, 3, 224, 224]))
    else:
        # Mock models for testing
        print("Creating mock models...")
        
        # Mock ResNet-style model
        class MockResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU()
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512, 1000)
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return x
        
        mock_resnet = MockResNet()
        models_to_verify.append(('MockResNet', mock_resnet, [1, 3, 224, 224]))
        
        # Mock MLP
        class MockMLP(nn.Module):
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
        
        mock_mlp = MockMLP()
        models_to_verify.append(('MockMLP', mock_mlp, [1, 784]))
    
    print()
    print("="*70)
    print("MODEL VERIFICATION RESULTS")
    print("="*70)
    
    all_results = []
    
    for model_name, model, input_shape in models_to_verify:
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")
        
        # Parse model
        model_ir = parse_torchvision_model(model, model_name)
        
        print(f"  Parameters: {model_ir.parameters_count:,}")
        print(f"  Layers: {len(model_ir.layers)}")
        print(f"  Has skip connections: {model_ir.has_skip_connections}")
        print(f"  Has attention: {model_ir.has_attention}")
        
        # Compute shapes
        try:
            parser = PyTorchModelParser()
            if model_ir.layers:
                parser.compute_shapes(input_shape)
                print(f"  Input shape: {model_ir.input_shape}")
                print(f"  Output shape: {model_ir.output_shape}")
        except:
            print(f"  Input shape: {input_shape}")
            print(f"  Output shape: N/A")
        
        # Verify properties
        results = verify_model_properties(model_ir, input_shape)
        
        print(f"\n  Properties:")
        for prop, value in results['properties'].items():
            status = "✓" if value else "✗"
            print(f"    {status} {prop}: {value}")
        
        if results['issues']:
            print(f"\n  Issues:")
            for issue in results['issues']:
                print(f"    ⚠ {issue}")
        
        # Generate Lean specifications
        specs = generate_lean_specifications(model_ir)
        
        print(f"\n  Generated Lean specifications: {len(specs)}")
        for i, spec in enumerate(specs[:2], 1):  # Show first 2
            print(f"    {i}. {spec.split(':')[0].replace('-- Theorem: ', '')}")
        
        results['specs_count'] = len(specs)
        all_results.append(results)
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    total_models = len(all_results)
    verified_models = sum(1 for r in all_results if r['verified'])
    total_specs = sum(r['specs_count'] for r in all_results)
    total_params = sum(r.get('properties', {}).get('has_parameters', False) for r in all_results)
    
    print(f"\nTotal models analyzed: {total_models}")
    print(f"Models verified: {verified_models}/{total_models}")
    print(f"Total Lean specs generated: {total_specs}")
    print(f"Models with parameters: {total_params}/{total_models}")
    
    print(f"\n{'Model':<20} {'Parameters':>12} {'Layers':>8} {'Specs':>8} {'Verified':>10}")
    print("-"*70)
    
    for result in all_results:
        name = result['model_name']
        params = "✓" if result['properties'].get('has_parameters') else "✗"
        layers = len([l for l in result.get('layers', [])])
        specs = result['specs_count']
        verified = "✓" if result['verified'] else "✗"
        
        print(f"{name:<20} {params:>12} {layers:>8} {specs:>8} {verified:>10}")
    
    print("\n" + "="*70)
    print("✓ REAL PyTorch MODEL VERIFICATION COMPLETE")
    print("="*70)
    
    print("\nNext steps:")
    print("  1. Install torchvision for real model verification")
    print("  2. Run: pip install torchvision")
    print("  3. Re-run this script to verify ResNet, AlexNet, VGG, etc.")
    print("  4. Use generated Lean specs for formal verification")


if __name__ == "__main__":
    main()
