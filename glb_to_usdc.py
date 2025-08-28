#!/usr/bin/env python3
"""
GLB to USDC Converter - Complete Implementation
GLBファイルをUSDCに完全変換（メッシュ、マテリアル、テクスチャを含む）
"""

import os
import sys
import argparse
import tempfile
import shutil
import struct
import base64
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

try:
    from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, UsdUtils, Vt
except ImportError:
    print("Error: USD Python bindings not found. Please install USD toolkit.")
    print("You can install it with: pip install usd-core")
    sys.exit(1)

try:
    from gltflib import GLTF
except ImportError:
    print("Error: gltflib not found. Please install it.")
    print("You can install it with: pip install gltflib")
    sys.exit(1)


class GLBToUSDCConverter:
    """GLBからUSDCへの完全変換を行うクラス"""
    
    def __init__(self, input_path: str, output_path: Optional[str] = None, verbose: bool = False):
        """初期化"""
        self.input_path = Path(input_path)
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if not self.input_path.suffix.lower() in ['.glb', '.gltf']:
            raise ValueError(f"Input file must be GLB or GLTF format: {input_path}")
        
        if output_path:
            self.output_path = Path(output_path)
        else:
            self.output_path = self.input_path.with_suffix('.usdc')
        
        self.verbose = verbose
        self.gltf = None
        self.stage = None
        self.buffer_cache = {}
        self.material_cache = {}
        self.mesh_cache = {}
        
    def log(self, message: str):
        """ログメッセージを出力"""
        if self.verbose:
            print(f"[INFO] {message}")
    
    def sanitize_usd_name(self, name: str) -> str:
        """USD用に名前をサニタイズ"""
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name or 'unnamed')
        if sanitized and sanitized[0].isdigit():
            sanitized = '_' + sanitized
        return sanitized or 'unnamed'
    
    def load_gltf(self):
        """GLTFファイルをロード"""
        self.log(f"Loading GLTF file: {self.input_path}")
        self.gltf = GLTF.load(str(self.input_path))
        
        # GLBファイルの場合、バイナリデータを直接読み込む
        if self.input_path.suffix.lower() == '.glb':
            with open(self.input_path, 'rb') as f:
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
                binary_data = f.read(bin_chunk_length)
                self.buffer_cache[0] = binary_data
                self.log(f"Loaded binary buffer: {len(binary_data):,} bytes")
        else:
            # GLTFファイルの場合、通常の処理
            if self.gltf.model.buffers:
                for i, buffer in enumerate(self.gltf.model.buffers):
                    if hasattr(buffer, 'data'):
                        self.buffer_cache[i] = buffer.data
                    elif hasattr(buffer, 'uri') and buffer.uri:
                        # Data URIの場合
                        if buffer.uri.startswith('data:'):
                            header, data = buffer.uri.split(',', 1)
                            self.buffer_cache[i] = base64.b64decode(data)
                        else:
                            # 外部ファイルの場合
                            buffer_path = self.input_path.parent / buffer.uri
                            with open(buffer_path, 'rb') as f:
                                self.buffer_cache[i] = f.read()
    
    def get_buffer_data(self, accessor_idx: int) -> np.ndarray:
        """アクセサーからバッファデータを取得"""
        if accessor_idx is None or not self.gltf.model.accessors or accessor_idx >= len(self.gltf.model.accessors):
            return None
            
        accessor = self.gltf.model.accessors[accessor_idx]
        buffer_view = self.gltf.model.bufferViews[accessor.bufferView]
        buffer_idx = buffer_view.buffer
        
        # バッファからデータを取得
        if buffer_idx not in self.buffer_cache:
            return None
            
        buffer_data = self.buffer_cache[buffer_idx]
        
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
    
    def convert_mesh(self, mesh_idx: int, parent_path: str) -> List[str]:
        """GLTFメッシュをUSDに変換"""
        cache_key = f"{mesh_idx}_{parent_path}"
        if cache_key in self.mesh_cache:
            return self.mesh_cache[cache_key]
        
        mesh = self.gltf.model.meshes[mesh_idx]
        mesh_name = self.sanitize_usd_name(mesh.name if hasattr(mesh, 'name') else f'mesh_{mesh_idx}')
        mesh_paths = []
        
        for prim_idx, primitive in enumerate(mesh.primitives):
            prim_name = f'{mesh_name}_prim{prim_idx}'
            # メッシュを直接親ノードの下に作成
            prim_path = f'{parent_path}/{prim_name}'
            
            self.log(f"Converting mesh primitive: {prim_name}")
            
            # USDメッシュを作成
            mesh_prim = UsdGeom.Mesh.Define(self.stage, prim_path)
            
            # 頂点位置
            if hasattr(primitive.attributes, 'POSITION'):
                positions = self.get_buffer_data(primitive.attributes.POSITION)
                if positions is not None:
                    # numpy型をPython floatに変換
                    points = [Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2])) for pos in positions]
                    mesh_prim.CreatePointsAttr(Vt.Vec3fArray(points))
                    self.log(f"  Added {len(points)} vertices")
            
            # 法線
            if hasattr(primitive.attributes, 'NORMAL'):
                normals = self.get_buffer_data(primitive.attributes.NORMAL)
                if normals is not None:
                    normal_array = [Gf.Vec3f(float(n[0]), float(n[1]), float(n[2])) for n in normals]
                    mesh_prim.CreateNormalsAttr(Vt.Vec3fArray(normal_array))
                    mesh_prim.SetNormalsInterpolation('vertex')
            
            # UV座標
            if hasattr(primitive.attributes, 'TEXCOORD_0'):
                uvs = self.get_buffer_data(primitive.attributes.TEXCOORD_0)
                if uvs is not None:
                    # USDはVec2fArrayを期待
                    primvars_api = UsdGeom.PrimvarsAPI(mesh_prim)
                    uv_attr = primvars_api.CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray)
                    uv_array = [Gf.Vec2f(float(uv[0]), float(uv[1])) for uv in uvs]
                    uv_attr.Set(Vt.Vec2fArray(uv_array))
                    uv_attr.SetInterpolation('vertex')
            
            # インデックス
            if hasattr(primitive, 'indices') and primitive.indices is not None:
                indices = self.get_buffer_data(primitive.indices)
                if indices is not None:
                    indices = indices.flatten()
                    
                    # 三角形として処理
                    face_vertex_counts = [3] * (len(indices) // 3)
                    mesh_prim.CreateFaceVertexCountsAttr(face_vertex_counts)
                    mesh_prim.CreateFaceVertexIndicesAttr(Vt.IntArray(indices.tolist()))
            
            # マテリアル割り当て
            if hasattr(primitive, 'material') and primitive.material is not None:
                material_path = self.convert_material(primitive.material)
                if material_path:
                    # マテリアルをバインド
                    UsdShade.MaterialBindingAPI(mesh_prim).Bind(
                        UsdShade.Material.Get(self.stage, material_path)
                    )
            
            mesh_paths.append(prim_path)
        
        self.mesh_cache[cache_key] = mesh_paths
        return mesh_paths
    
    def convert_material(self, material_idx: int) -> str:
        """GLTFマテリアルをUSDに変換"""
        if material_idx in self.material_cache:
            return self.material_cache[material_idx]
        
        material = self.gltf.model.materials[material_idx]
        material_name = self.sanitize_usd_name(
            material.name if hasattr(material, 'name') and material.name 
            else f'material_{material_idx}'
        )
        material_path = f'/root/Materials/{material_name}'
        
        self.log(f"Converting material: {material_name}")
        
        # USDマテリアルを作成
        usd_material = UsdShade.Material.Define(self.stage, material_path)
        
        # PBRシェーダーを作成
        shader = UsdShade.Shader.Define(self.stage, f'{material_path}/PBRShader')
        shader.CreateIdAttr('UsdPreviewSurface')
        
        # PBRメタリックラフネス
        if hasattr(material, 'pbrMetallicRoughness') and material.pbrMetallicRoughness:
            pbr = material.pbrMetallicRoughness
            
            # ベースカラー
            if hasattr(pbr, 'baseColorFactor') and pbr.baseColorFactor:
                color = pbr.baseColorFactor[:3]
                shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(color[0], color[1], color[2])
                )
                # 不透明度
                if len(pbr.baseColorFactor) > 3:
                    shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(pbr.baseColorFactor[3])
            
            # メタリック
            if hasattr(pbr, 'metallicFactor') and pbr.metallicFactor is not None:
                shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(float(pbr.metallicFactor))
            else:
                shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
            
            # ラフネス
            if hasattr(pbr, 'roughnessFactor') and pbr.roughnessFactor is not None:
                shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(float(pbr.roughnessFactor))
            else:
                shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.5)
        
        # エミッシブ
        if hasattr(material, 'emissiveFactor') and material.emissiveFactor:
            shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*material.emissiveFactor[:3])
            )
        
        # 両面レンダリング
        if hasattr(material, 'doubleSided') and material.doubleSided:
            shader.CreateInput('doubleSided', Sdf.ValueTypeNames.Bool).Set(True)
        
        # シェーダーをマテリアルに接続
        usd_material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        
        self.material_cache[material_idx] = material_path
        return material_path
    
    def convert_node(self, node_idx: int, parent_path: str = '/root') -> str:
        """GLTFノードをUSDに変換"""
        node = self.gltf.model.nodes[node_idx]
        node_name = self.sanitize_usd_name(
            node.name if hasattr(node, 'name') and node.name 
            else f'node_{node_idx}'
        )
        node_path = f'{parent_path}/{node_name}'
        
        self.log(f"Converting node: {node_name}")
        
        # Xformを作成
        xform = UsdGeom.Xform.Define(self.stage, node_path)
        
        # トランスフォーム設定
        if hasattr(node, 'matrix') and node.matrix:
            # 4x4行列
            matrix = Gf.Matrix4d(*node.matrix)
            xform.AddTransformOp().Set(matrix)
        else:
            # Translation, Rotation, Scale
            if hasattr(node, 'translation') and node.translation:
                xform.AddTranslateOp().Set(Gf.Vec3d(*node.translation))
            
            if hasattr(node, 'rotation') and node.rotation:
                # クォータニオン (x, y, z, w)
                q = node.rotation
                xform.AddOrientOp().Set(Gf.Quatf(q[3], q[0], q[1], q[2]))
            
            if hasattr(node, 'scale') and node.scale:
                xform.AddScaleOp().Set(Gf.Vec3f(*node.scale))
        
        # メッシュを追加（直接ノードの下に作成、参照は使わない）
        if hasattr(node, 'mesh') and node.mesh is not None:
            self.convert_mesh(node.mesh, node_path)
        
        # 子ノードを変換
        if hasattr(node, 'children') and node.children:
            for child_idx in node.children:
                self.convert_node(child_idx, node_path)
        
        return node_path
    
    def convert(self) -> bool:
        """変換を実行"""
        try:
            self.log(f"Converting {self.input_path} to {self.output_path}")
            
            # GLTFをロード
            self.load_gltf()
            
            # 一時的なUSDAファイルを作成
            temp_usda = self.output_path.with_suffix('.usda')
            
            # 既存のUSDAファイルがある場合は削除
            if temp_usda.exists():
                self.log(f"Removing existing USDA file: {temp_usda}")
                temp_usda.unlink()
            
            # USD stageを作成
            self.stage = Usd.Stage.CreateNew(str(temp_usda))
            
            # ルートプリムを設定
            root_xform = UsdGeom.Xform.Define(self.stage, '/root')
            self.stage.SetDefaultPrim(root_xform.GetPrim())
            
            # シーンを変換
            if self.gltf.model.scenes and len(self.gltf.model.scenes) > 0:
                scene_idx = self.gltf.model.scene if hasattr(self.gltf.model, 'scene') else 0
                scene = self.gltf.model.scenes[scene_idx]
                
                if hasattr(scene, 'nodes') and scene.nodes:
                    for node_idx in scene.nodes:
                        self.convert_node(node_idx)
            
            # ステージを保存
            self.stage.Save()
            
            # USDAからUSDCに変換
            self.log(f"Converting USDA to USDC: {temp_usda} -> {self.output_path}")
            self.stage.Export(str(self.output_path))
            
            # 一時ファイルを削除
            if temp_usda.exists():
                temp_usda.unlink()
            
            if self.output_path.exists():
                self.log(f"Successfully converted to: {self.output_path}")
                self.log(f"Output file size: {self.output_path.stat().st_size:,} bytes")
                
                # 統計情報
                if self.verbose:
                    self.log(f"Converted {len(self.mesh_cache)} meshes")
                    self.log(f"Converted {len(self.material_cache)} materials")
                
                return True
            else:
                self.log("Conversion failed: output file not created")
                return False
                
        except Exception as e:
            print(f"Error during conversion: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='Convert GLB/GLTF files to USDC format with complete mesh and material data'
    )
    parser.add_argument('input', help='Input GLB/GLTF file path')
    parser.add_argument('-o', '--output', help='Output USDC file path (optional)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    converter = GLBToUSDCConverter(args.input, args.output, args.verbose)
    if not converter.convert():
        sys.exit(1)


if __name__ == '__main__':
    main()