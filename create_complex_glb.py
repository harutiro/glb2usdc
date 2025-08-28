#!/usr/bin/env python3
"""
複雑なマテリアルを持つGLBファイルを作成
"""

import struct
import json
import numpy as np

def create_complex_glb():
    """複雑なマテリアルを持つGLBファイルを作成"""
    
    # 立方体の頂点データ（8頂点）
    vertices = np.array([
        # 前面
        [-1.0, -1.0,  1.0],  # 0
        [ 1.0, -1.0,  1.0],  # 1
        [ 1.0,  1.0,  1.0],  # 2
        [-1.0,  1.0,  1.0],  # 3
        # 背面
        [-1.0, -1.0, -1.0],  # 4
        [ 1.0, -1.0, -1.0],  # 5
        [ 1.0,  1.0, -1.0],  # 6
        [-1.0,  1.0, -1.0],  # 7
    ], dtype=np.float32)
    
    # 法線データ（前面向き）
    normals = np.array([
        [0.0, 0.0, 1.0],  # 0
        [0.0, 0.0, 1.0],  # 1
        [0.0, 0.0, 1.0],  # 2
        [0.0, 0.0, 1.0],  # 3
        [0.0, 0.0, -1.0], # 4
        [0.0, 0.0, -1.0], # 5
        [0.0, 0.0, -1.0], # 6
        [0.0, 0.0, -1.0], # 7
    ], dtype=np.float32)
    
    # UV座標
    uvs = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1
        [1.0, 1.0],  # 2
        [0.0, 1.0],  # 3
        [0.0, 0.0],  # 4
        [1.0, 0.0],  # 5
        [1.0, 1.0],  # 6
        [0.0, 1.0],  # 7
    ], dtype=np.float32)
    
    # 立方体のインデックス（12個の三角形、前面のみ簡単化）
    indices = np.array([
        # 前面
        0, 1, 2,  2, 3, 0,
        # 上面
        3, 2, 6,  6, 7, 3,
        # 右面
        1, 5, 6,  6, 2, 1,
        # 左面
        4, 0, 3,  3, 7, 4
    ], dtype=np.uint16)
    
    # バイナリデータを結合
    vertices_bytes = vertices.tobytes()
    normals_bytes = normals.tobytes()
    uvs_bytes = uvs.tobytes()
    indices_bytes = indices.tobytes()
    
    binary_data = vertices_bytes + normals_bytes + uvs_bytes + indices_bytes
    
    # パディングを追加（4バイト境界に合わせる）
    while len(binary_data) % 4 != 0:
        binary_data += b'\x00'
    
    # JSON部分を作成（複数のマテリアル）
    gltf_json = {
        "asset": {
            "version": "2.0",
            "generator": "Python Complex Material Test"
        },
        "scene": 0,
        "scenes": [{
            "name": "ComplexScene",
            "nodes": [0]
        }],
        "nodes": [{
            "mesh": 0,
            "name": "ComplexCube"
        }],
        "meshes": [{
            "name": "ComplexMesh",
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "NORMAL": 1,
                    "TEXCOORD_0": 2
                },
                "indices": 3,
                "material": 0  # 赤いマテリアル
            }]
        }],
        "materials": [
            {
                "name": "RedMetallic",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.8, 0.1, 0.1, 1.0],  # 濃い赤
                    "metallicFactor": 0.9,  # 高いメタリック
                    "roughnessFactor": 0.1   # 低いラフネス（光沢）
                },
                "emissiveFactor": [0.1, 0.0, 0.0],  # 弱い赤の発光
                "doubleSided": True
            }
        ],
        "buffers": [{
            "byteLength": len(binary_data)
        }],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(vertices_bytes),
                "target": 34962
            },
            {
                "buffer": 0,
                "byteOffset": len(vertices_bytes),
                "byteLength": len(normals_bytes),
                "target": 34962
            },
            {
                "buffer": 0,
                "byteOffset": len(vertices_bytes) + len(normals_bytes),
                "byteLength": len(uvs_bytes),
                "target": 34962
            },
            {
                "buffer": 0,
                "byteOffset": len(vertices_bytes) + len(normals_bytes) + len(uvs_bytes),
                "byteLength": len(indices_bytes),
                "target": 34963
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": 8,
                "type": "VEC3",
                "min": [-1.0, -1.0, -1.0],
                "max": [1.0, 1.0, 1.0]
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": 8,
                "type": "VEC3",
                "min": [-1.0, -1.0, -1.0],
                "max": [1.0, 1.0, 1.0]
            },
            {
                "bufferView": 2,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": 8,
                "type": "VEC2",
                "min": [0.0, 0.0],
                "max": [1.0, 1.0]
            },
            {
                "bufferView": 3,
                "byteOffset": 0,
                "componentType": 5123,  # UNSIGNED_SHORT
                "count": 24,
                "type": "SCALAR",
                "min": [0],
                "max": [7]
            }
        ]
    }
    
    # GLBファイルの構築
    json_bytes = json.dumps(gltf_json, separators=(',', ':')).encode('utf-8')
    json_padding = b' ' * ((4 - len(json_bytes) % 4) % 4)
    json_bytes += json_padding
    
    # チャンクを作成
    json_chunk_type = 0x4E4F534A  # "JSON"
    json_chunk = struct.pack('<I', len(json_bytes)) + struct.pack('<I', json_chunk_type) + json_bytes
    
    bin_chunk_type = 0x004E4942  # "BIN\x00"
    bin_chunk = struct.pack('<I', len(binary_data)) + struct.pack('<I', bin_chunk_type) + binary_data
    
    # GLBヘッダー
    magic = b'glTF'
    version = 2
    total_length = 12 + len(json_chunk) + len(bin_chunk)
    header = magic + struct.pack('<I', version) + struct.pack('<I', total_length)
    
    # GLBファイルを組み立て
    glb_data = header + json_chunk + bin_chunk
    
    # ファイルに書き込み
    filename = 'complex_test.glb'
    with open(filename, 'wb') as f:
        f.write(glb_data)
    
    print(f"Created {filename} ({len(glb_data)} bytes)")
    print(f"  - Vertices: {len(vertices)}")
    print(f"  - Triangles: {len(indices)//3}")
    print(f"  - Materials: 1 (RedMetallic)")
    
    return filename

if __name__ == '__main__':
    create_complex_glb()