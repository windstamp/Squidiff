#!/usr/bin/env python3
"""
Profiling utilities for Squidiff models.
Adapted from ALPHAQUBIT profiling tools.
"""

import json
import os


def register_hooks(model, hook_dict):
    """Register forward hooks to capture layer inputs/outputs."""
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(input, tuple):
                input_shapes = [tuple(x.shape) if hasattr(x, 'shape') else str(type(x)) for x in input]
                input_dtypes = [str(x.dtype) if hasattr(x, 'dtype') else str(type(x)) for x in input]
            else:
                input_shapes = [tuple(input.shape) if hasattr(input, 'shape') else str(type(input))]
                input_dtypes = [str(input.dtype) if hasattr(input, 'dtype') else str(type(input))]
            
            if isinstance(output, tuple):
                output_shapes = [tuple(x.shape) if hasattr(x, 'shape') else str(type(x)) for x in output]
                output_dtypes = [str(x.dtype) if hasattr(x, 'dtype') else str(type(x)) for x in output]
            else:
                output_shapes = [tuple(output.shape) if hasattr(output, 'shape') else str(type(output))]
                output_dtypes = [str(output.dtype) if hasattr(output, 'dtype') else str(type(output))]
            
            hook_dict[name] = {
                'input_shapes': input_shapes,
                'input_dtypes': input_dtypes,
                'output_shapes': output_shapes,
                'output_dtypes': output_dtypes
            }
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            hook = module.register_forward_hook(make_hook(name))
            hooks.append(hook)
    return hooks


def analyze_layer_shapes(json_file):
    """Analyze layer shapes from hooks."""
    print(f"\n{'='*80}")
    print(f"Layer Shapes Analysis: {json_file}")
    print(f"{'='*80}\n")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"{'Layer Name':<50} {'Input Shape':<30} {'Output Shape':<30}")
    print("-" * 110)
    
    for name, info in data.items():
        input_shapes = info.get('input_shapes', [])
        output_shapes = info.get('output_shapes', [])
        input_dtypes = info.get('input_dtypes', [])
        output_dtypes = info.get('output_dtypes', [])
        
        input_str = str(input_shapes[0]) if input_shapes else "N/A"
        output_str = str(output_shapes[0]) if output_shapes else "N/A"
        
        print(f"{name:<50} {input_str:<30} {output_str:<30}")
        
        if len(input_dtypes) > 0 and input_dtypes[0] != str(type(None)):
            print(f"{'':>50} dtype: {input_dtypes[0]:<23} dtype: {output_dtypes[0] if output_dtypes else 'N/A'}")


def analyze_chrome_trace(json_file):
    """Analyze Chrome trace JSON."""
    print(f"\n{'='*80}")
    print(f"Chrome Trace Analysis: {json_file}")
    print(f"{'='*80}\n")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if 'traceEvents' not in data:
        print("No trace events found in the file.")
        return
    
    events = data['traceEvents']
    
    op_times = {}
    for event in events:
        if event.get('cat') == 'kernel' or event.get('cat') == 'cpu_op':
            name = event.get('name', 'Unknown')
            dur = event.get('dur', 0)
            
            if name not in op_times:
                op_times[name] = {'count': 0, 'total_time': 0, 'avg_time': 0}
            
            op_times[name]['count'] += 1
            op_times[name]['total_time'] += dur
    
    for name, stats in op_times.items():
        stats['avg_time'] = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
    
    sorted_ops = sorted(op_times.items(), key=lambda x: x[1]['total_time'], reverse=True)
    
    print(f"{'Operator Name':<50} {'Count':<10} {'Total Time (us)':<20} {'Avg Time (us)':<20}")
    print("-" * 100)
    
    for name, stats in sorted_ops[:30]:
        print(f"{name:<50} {stats['count']:<10} {stats['total_time']:<20.2f} {stats['avg_time']:<20.2f}")
    
    return sorted_ops


def extract_operator_info(json_file, output_file=None):
    """Extract detailed operator information including tensor shapes."""
    print(f"\n{'='*80}")
    print(f"Detailed Operator Information: {json_file}")
    print(f"{'='*80}\n")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if 'traceEvents' not in data:
        print("No trace events found.")
        return []
    
    events = data['traceEvents']
    
    operators = []
    for event in events:
        if 'args' in event and 'Input Dims' in event.get('args', {}):
            op_info = {
                'name': event.get('name', 'Unknown'),
                'input_dims': event['args'].get('Input Dims', []),
                'output_dims': event['args'].get('Output Dims', []),
                'duration': event.get('dur', 0),
                'timestamp': event.get('ts', 0)
            }
            operators.append(op_info)
    
    print(f"Found {len(operators)} operators with shape information\n")
    print(f"{'Operator':<40} {'Input Shapes':<40} {'Output Shapes':<40} {'Duration (us)':<15}")
    print("-" * 135)
    
    for op in operators[:20]:  # Show first 20
        input_str = str(op['input_dims'])[:38]
        output_str = str(op['output_dims'])[:38]
        print(f"{op['name']:<40} {input_str:<40} {output_str:<40} {op['duration']:<15.2f}")
    
    if len(operators) > 20:
        print(f"\n... and {len(operators) - 20} more operators")
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(operators, f, indent=2)
        print(f"\nFull operator list saved to: {output_file}")
    
    return operators
