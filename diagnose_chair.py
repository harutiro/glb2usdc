#!/usr/bin/env python3
"""
Chair GLBファイルの詳細診断スクリプト
メッシュデータ、頂点、インデックスを分析して異常を検出
"""

import sys
import numpy as np
from pathlib import Path

try:
    from gltflib import GLTF
except ImportError:
    print("gltflib not found. Please install it with: pip install gltflib")
    sys.exit(1)

def analyze_glb(file_path):
    """GLBファイルを詳細分析"""
    print(f"=== Analyzing GLB: {file_path} ===")
    
    # GLTFをロード
    gltf = GLTF.load(str(file_path))
    
    # バイナリデータをロード
    with open(file_path, 'rb') as f:
        # GLBヘッダー（12バイト）をスキップ
        f.seek(12)
        
        # JSONチャンクのサイズを読む
        json_chunk_length = int.from_bytes(f.read(4), 'little')
        f.seek(20)  # JSONチャンクタイプの後
        
        # JSONデータをスキップして、BINチャンクに移動
        f.seek(20 + json_chunk_length)
        
        # BINチャンクのサイズを読む
        bin_chunk_length = int.from_bytes(f.read(4), 'little')
        bin_chunk_type = f.read(4)  # should be b'BIN\x00'
        
        # バイナリデータを読み込む
        buffer_data = f.read(bin_chunk_length)
        print(f"Binary buffer size: {len(buffer_data):,} bytes")
    
    # シーン情報
    print(f"\nScenes: {len(gltf.model.scenes)}")
    if gltf.model.scenes:
        scene = gltf.model.scenes[0]
        if hasattr(scene, 'nodes'):
            print(f"Root nodes in scene: {len(scene.nodes)}")
    
    # ノード情報
    print(f"\nTotal nodes: {len(gltf.model.nodes)}")
    mesh_nodes = []
    for i, node in enumerate(gltf.model.nodes):
        name = getattr(node, 'name', f'node_{i}')
        has_mesh = hasattr(node, 'mesh') and node.mesh is not None
        has_children = hasattr(node, 'children') and node.children
        child_count = len(node.children) if has_children else 0
        
        print(f"  Node {i:2d}: {name:<25} mesh={has_mesh:<5} children={child_count}")
        
        if has_mesh:
            mesh_nodes.append((i, name, node.mesh))
    
    print(f"\nNodes with meshes: {len(mesh_nodes)}")
    
    # メッシュ情報詳細
    print(f"\nMesh analysis:")
    print(f"Total meshes: {len(gltf.model.meshes)}")
    
    total_vertices = 0
    total_triangles = 0
    
    for mesh_idx, mesh in enumerate(gltf.model.meshes):
        mesh_name = getattr(mesh, 'name', f'mesh_{mesh_idx}')
        print(f"\n  Mesh {mesh_idx}: {mesh_name}")
        print(f"    Primitives: {len(mesh.primitives)}")
        
        for prim_idx, primitive in enumerate(mesh.primitives):
            print(f"    Primitive {prim_idx}:")
            
            # 頂点位置を取得
            if hasattr(primitive.attributes, 'POSITION'):
                position_accessor_idx = primitive.attributes.POSITION
                position_accessor = gltf.model.accessors[position_accessor_idx]
                vertex_count = position_accessor.count
                total_vertices += vertex_count
                print(f"      Vertices: {vertex_count}")
                
                # 実際の頂点データを読み込んで分析
                positions = get_buffer_data(gltf, buffer_data, position_accessor_idx)
                if positions is not None:
                    min_pos = np.min(positions, axis=0)
                    max_pos = np.max(positions, axis=0)
                    center = np.mean(positions, axis=0)
                    print(f"      Position range: [{min_pos[0]:.3f}, {min_pos[1]:.3f}, {min_pos[2]:.3f}] to [{max_pos[0]:.3f}, {max_pos[1]:.3f}, {max_pos[2]:.3f}]")
                    print(f"      Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
                    
                    # 異常な値をチェック
                    if np.any(np.abs(positions) > 1000):
                        print(f"      ⚠️  WARNING: Extreme vertex positions detected!")
                        extreme_indices = np.where(np.abs(positions) > 1000)
                        print(f"         Extreme vertices: {len(extreme_indices[0])} vertices")
                    
                    # NaNやInfをチェック
                    if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
                        print(f"      ❌ ERROR: NaN or Inf values in vertex positions!")
                    
                    # 重複頂点をチェック
                    unique_positions = np.unique(positions.view(np.void), return_counts=True)
                    duplicate_count = np.sum(unique_positions[1] > 1)
                    if duplicate_count > 0:
                        print(f"      🔄 Duplicate vertices: {duplicate_count}")
            
            # インデックスを取得
            if hasattr(primitive, 'indices') and primitive.indices is not None:
                indices_accessor_idx = primitive.indices
                indices_accessor = gltf.model.accessors[indices_accessor_idx]
                indices_count = indices_accessor.count
                triangle_count = indices_count // 3
                total_triangles += triangle_count
                print(f"      Indices: {indices_count} ({triangle_count} triangles)")
                
                # インデックスデータを読み込んで分析
                indices = get_buffer_data(gltf, buffer_data, indices_accessor_idx)
                if indices is not None:
                    indices = indices.flatten()
                    max_vertex_idx = vertex_count - 1 if 'vertex_count' in locals() else 0
                    
                    # インデックス範囲チェック
                    if np.any(indices > max_vertex_idx):
                        invalid_indices = np.sum(indices > max_vertex_idx)
                        print(f"      ❌ ERROR: {invalid_indices} indices exceed vertex count!")
                        print(f"         Max valid index: {max_vertex_idx}, Found indices up to: {np.max(indices)}")
                    
                    # 重複三角形をチェック
                    triangles = indices.reshape(-1, 3)
                    sorted_triangles = np.sort(triangles, axis=1)
                    unique_triangles = np.unique(sorted_triangles.view(np.void), return_counts=True)
                    duplicate_triangles = np.sum(unique_triangles[1] > 1)
                    if duplicate_triangles > 0:
                        print(f"      🔄 Duplicate triangles: {duplicate_triangles}")
                    
                    # 退化三角形をチェック（同じ頂点を持つ三角形）
                    degenerate_count = 0
                    for triangle in triangles:
                        if len(np.unique(triangle)) < 3:
                            degenerate_count += 1
                    if degenerate_count > 0:
                        print(f"      ⚠️  Degenerate triangles: {degenerate_count}")
            
            # マテリアル情報
            if hasattr(primitive, 'material') and primitive.material is not None:
                material_idx = primitive.material
                if material_idx < len(gltf.model.materials):
                    material = gltf.model.materials[material_idx]
                    material_name = getattr(material, 'name', f'material_{material_idx}')
                    print(f"      Material: {material_idx} ({material_name})")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total vertices across all meshes: {total_vertices:,}")
    print(f"Total triangles across all meshes: {total_triangles:,}")
    print(f"Average vertices per mesh: {total_vertices // len(gltf.model.meshes) if gltf.model.meshes else 0}")

def get_buffer_data(gltf, buffer_data, accessor_idx):
    """アクセサーからバッファデータを取得"""
    try:
        if accessor_idx is None or not gltf.model.accessors or accessor_idx >= len(gltf.model.accessors):
            return None
            
        accessor = gltf.model.accessors[accessor_idx]
        buffer_view = gltf.model.bufferViews[accessor.bufferView]
        
        # オフセットとサイズを計算
        byte_offset = buffer_view.byteOffset or 0
        if hasattr(accessor, 'byteOffset'):
            byte_offset += accessor.byteOffset or 0
        
        # コンポーネントタイプに基づいてデータ型を決定
        dtype_map = {
            5120: np.int8,    # BYTE
            5121: np.uint8,   # UNSIGNED_BYTE
            5122: np.int16,   # SHORT
            5123: np.uint16,  # UNSIGNED_SHORT
            5125: np.uint32,  # UNSIGNED_INT
            5126: np.float32, # FLOAT
        }
        
        dtype = dtype_map.get(accessor.componentType, np.float32)
        
        # タイプに基づいて要素数を決定
        type_sizes = {
            'SCALAR': 1,
            'VEC2': 2,
            'VEC3': 3,
            'VEC4': 4,
            'MAT2': 4,
            'MAT3': 9,
            'MAT4': 16
        }
        
        component_count = type_sizes.get(accessor.type, 1)
        element_size = np.dtype(dtype).itemsize * component_count
        
        # データを読み取り
        data = []
        for i in range(accessor.count):
            start = byte_offset + i * element_size
            end = start + element_size
            element_data = buffer_data[start:end]
            element = np.frombuffer(element_data, dtype=dtype, count=component_count)
            data.append(element)
        
        return np.array(data)
    except Exception as e:
        print(f"Error reading buffer data: {e}")
        return None

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        file_path = Path('chair.glb')
    
    if file_path.exists():
        analyze_glb(file_path)
    else:
        print(f"File not found: {file_path}")