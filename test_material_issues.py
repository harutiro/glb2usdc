#!/usr/bin/env python3
"""
マテリアルの問題をテストするスクリプト
"""

import sys
sys.path.append('.')

from glb_to_usdc import GLBToUSDCConverter
from pxr import Usd, UsdShade, UsdGeom
import struct
import json
import numpy as np

def create_problematic_glb():
    """問題のあるマテリアル情報を持つGLBファイルを作成"""
    
    # 三角形の頂点データ
    vertices = np.array([
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [ 0.0,  1.0, 0.0],
    ], dtype=np.float32)
    
    indices = np.array([0, 1, 2], dtype=np.uint16)
    binary_data = vertices.tobytes() + indices.tobytes()
    
    while len(binary_data) % 4 != 0:
        binary_data += b'\x00'
    
    # 問題のあるマテリアル定義を作成
    gltf_json = {
        "asset": {"version": "2.0", "generator": "Problem Test"},
        "scene": 0,
        "scenes": [{"name": "Scene", "nodes": [0, 1, 2]}],
        "nodes": [
            {"mesh": 0, "name": "NormalMaterial"},
            {"mesh": 1, "name": "NoBaseColor"},
            {"mesh": 2, "name": "NoPBRSection"}
        ],
        "meshes": [
            {
                "name": "Mesh1",
                "primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]
            },
            {
                "name": "Mesh2", 
                "primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 1}]
            },
            {
                "name": "Mesh3",
                "primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 2}]
            }
        ],
        "materials": [
            {
                "name": "NormalMaterial",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.8, 0.2, 0.2, 1.0],
                    "metallicFactor": 0.1,
                    "roughnessFactor": 0.8
                }
            },
            {
                "name": "NoBaseColorMaterial",
                "pbrMetallicRoughness": {
                    # baseColorFactorが意図的に欠落
                    "metallicFactor": 0.5,
                    "roughnessFactor": 0.3
                }
            },
            {
                "name": "NoPBRMaterial"
                # pbrMetallicRoughnessセクション全体が欠落
            }
        ],
        "buffers": [{"byteLength": len(binary_data)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(vertices.tobytes()), "target": 34962},
            {"buffer": 0, "byteOffset": len(vertices.tobytes()), "byteLength": len(indices.tobytes()), "target": 34963}
        ],
        "accessors": [
            {"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [-1.0, -1.0, 0.0], "max": [1.0, 1.0, 0.0]},
            {"bufferView": 1, "byteOffset": 0, "componentType": 5123, "count": 3, "type": "SCALAR", "min": [0], "max": [2]}
        ]
    }
    
    # GLBファイルの構築
    json_bytes = json.dumps(gltf_json, separators=(',', ':')).encode('utf-8')
    json_padding = b' ' * ((4 - len(json_bytes) % 4) % 4)
    json_bytes += json_padding
    
    json_chunk_type = 0x4E4F534A
    json_chunk = struct.pack('<I', len(json_bytes)) + struct.pack('<I', json_chunk_type) + json_bytes
    
    bin_chunk_type = 0x004E4942
    bin_chunk = struct.pack('<I', len(binary_data)) + struct.pack('<I', bin_chunk_type) + binary_data
    
    magic = b'glTF'
    version = 2
    total_length = 12 + len(json_chunk) + len(bin_chunk)
    header = magic + struct.pack('<I', version) + struct.pack('<I', total_length)
    
    glb_data = header + json_chunk + bin_chunk
    
    with open('problematic_materials.glb', 'wb') as f:
        f.write(glb_data)
    
    print(f"Created problematic_materials.glb ({len(glb_data)} bytes)")
    return 'problematic_materials.glb'

def test_problematic_materials():
    """問題のあるマテリアルファイルをテスト"""
    
    # 問題のあるGLBを作成
    glb_file = create_problematic_glb()
    
    # 変換を実行
    print("\n=== 変換実行 ===")
    converter = GLBToUSDCConverter(glb_file, 'problematic_materials.usdc', verbose=True)
    if not converter.convert():
        print("❌ 変換に失敗しました")
        return False
    
    # 結果を検証
    print("\n=== 変換結果の検証 ===")
    stage = Usd.Stage.Open('problematic_materials.usdc')
    
    # マテリアルの詳細を確認
    materials = [p for p in stage.Traverse() if p.IsA(UsdShade.Material)]
    print(f"マテリアル数: {len(materials)}")
    
    for mat_prim in materials:
        material = UsdShade.Material(mat_prim)
        print(f"\n  マテリアル: {mat_prim.GetPath()}")
        
        # シェーダーの入力値を確認
        shaders = [p for p in mat_prim.GetChildren() if p.IsA(UsdShade.Shader)]
        for shader_prim in shaders:
            shader = UsdShade.Shader(shader_prim)
            inputs = shader.GetInputs()
            
            for inp in inputs:
                value = inp.Get()
                print(f"    {inp.GetBaseName()}: {value}")

if __name__ == '__main__':
    test_problematic_materials()