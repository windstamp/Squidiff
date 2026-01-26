#!/usr/bin/env python3
"""
Analyze profiling results from Squidiff training.

Usage:
    python analyze_squidiff_profile.py --profile_dir ./profiling_logs
"""

import argparse
from profile_utils import analyze_layer_shapes, analyze_chrome_trace, extract_operator_info


def main():
    parser = argparse.ArgumentParser(description='Analyze Squidiff profiling results')
    parser.add_argument('--profile_dir', type=str, default='./profiling_logs',
                      help='Directory containing profiling results')
    args = parser.parse_args()
    
    import os
    layer_shapes_file = os.path.join(args.profile_dir, 'layer_shapes.json')
    trace_file = os.path.join(args.profile_dir, 'trace.json')
    
    print(f"\n{'='*80}")
    print(f"Analyzing Squidiff MLPModel Profiling Results")
    print(f"{'='*80}\n")
    
    # Analyze layer shapes
    if os.path.exists(layer_shapes_file):
        analyze_layer_shapes(layer_shapes_file)
    else:
        print(f"Layer shapes file not found: {layer_shapes_file}")
    
    # Analyze Chrome trace
    if os.path.exists(trace_file):
        sorted_ops = analyze_chrome_trace(trace_file)
        
        # Extract detailed operator info
        operators_file = os.path.join(args.profile_dir, 'operators_detail.json')
        extract_operator_info(trace_file, operators_file)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"Summary")
        print(f"{'='*80}\n")
        print(f"Total unique operators: {len(sorted_ops)}")
        if sorted_ops:
            print(f"Most expensive operator: {sorted_ops[0][0]} ({sorted_ops[0][1]['total_time']:.2f} us)")
        print(f"\nFull results saved to: {args.profile_dir}")
    else:
        print(f"Trace file not found: {trace_file}")


if __name__ == '__main__':
    main()
