#!/usr/bin/env python3
"""
Simple STL to GLB converter
Usage: python stl_to_glb.py input.stl output.glb
"""

import trimesh


def stl_to_glb(input_stl, output_glb):
    """Convert STL file to GLB format"""
    print(f"Loading STL: {input_stl}")

    # Load the STL file
    mesh = trimesh.load(input_stl)

    # Print mesh info
    print("✓ Loaded mesh:")
    print(f"  - Vertices: {len(mesh.vertices)}")
    print(f"  - Faces: {len(mesh.faces)}")
    print(f"  - Bounds: {mesh.bounds}")

    # Export to GLB
    print(f"\nExporting to GLB: {output_glb}")
    mesh.export(output_glb)

    print("✓ Conversion complete!")


if __name__ == "__main__":
    input_stl = "/home/martinaccou/work/TRELLIS.2/data/new_plane.stl"
    output_glb = "/home/martinaccou/work/TRELLIS.2/data/new_plane.glb"

    stl_to_glb(input_stl, output_glb)
