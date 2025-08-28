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
    
    def __init__(self, input_path: str, output_path: Optional[str] = None, verbose: bool = False, extract_textures: bool = False):
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
        self.extract_textures = extract_textures
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
    
    def _process_texture(self, texture_index: int, material_name: str, texture_type: str) -> Optional[str]:
        """GLTFテクスチャを処理してUSDで使用可能な形式に変換"""
        try:
            if not self.gltf.model.textures or texture_index >= len(self.gltf.model.textures):
                self.log(f"    Warning: Texture index {texture_index} not found")
                return None
            
            texture = self.gltf.model.textures[texture_index]
            
            # イメージのインデックスを取得
            if not hasattr(texture, 'source') or texture.source is None:
                self.log(f"    Warning: Texture {texture_index} has no source image")
                return None
            
            image_index = texture.source
            
            if not self.gltf.model.images or image_index >= len(self.gltf.model.images):
                self.log(f"    Warning: Image index {image_index} not found")
                return None
            
            image = self.gltf.model.images[image_index]
            
            # 埋め込みテクスチャ（バイナリ）の処理
            if hasattr(image, 'bufferView') and image.bufferView is not None:
                # バッファから画像データを抽出
                buffer_view_index = image.bufferView
                if buffer_view_index < len(self.gltf.model.bufferViews):
                    buffer_view = self.gltf.model.bufferViews[buffer_view_index]
                    buffer_idx = buffer_view.buffer
                    
                    if buffer_idx in self.buffer_cache:
                        buffer_data = self.buffer_cache[buffer_idx]
                        byte_offset = buffer_view.byteOffset or 0
                        byte_length = buffer_view.byteLength
                        
                        # 画像データを抽出
                        image_data = buffer_data[byte_offset:byte_offset + byte_length]
                        
                        # ファイル拡張子を推定（MIMEタイプから）
                        mime_type = getattr(image, 'mimeType', 'image/jpeg')
                        if 'png' in mime_type.lower():
                            ext = '.png'
                        elif 'jpeg' in mime_type.lower() or 'jpg' in mime_type.lower():
                            ext = '.jpg'
                        else:
                            ext = '.jpg'  # デフォルト
                        
                        # テクスチャファイルを出力ディレクトリに保存
                        texture_filename = f"{material_name}_{texture_type}_{texture_index}{ext}"
                        
                        if self.extract_textures:
                            texture_path = self.output_path.parent / texture_filename
                            with open(texture_path, 'wb') as f:
                                f.write(image_data)
                            self.log(f"    Extracted texture: {texture_filename} ({len(image_data)} bytes)")
                            # 相対パスを返す
                            return f"./{texture_filename}"
                        else:
                            self.log(f"    Texture extraction disabled: {texture_filename}")
                            # テクスチャファイルを出力せずに、USDで直接バイナリデータを使用
                            return None
            
            # 外部ファイル参照（URI）の処理
            elif hasattr(image, 'uri') and image.uri:
                # データURIかファイルパスか判定
                if image.uri.startswith('data:'):
                    self.log(f"    Warning: Data URI textures not fully supported yet")
                    return None
                else:
                    # 外部ファイル参照
                    original_path = self.input_path.parent / image.uri
                    if original_path.exists():
                        texture_filename = f"{material_name}_{texture_type}_{texture_index}{original_path.suffix}"
                        
                        if self.extract_textures:
                            # ファイルを出力ディレクトリにコピー
                            texture_path = self.output_path.parent / texture_filename
                            import shutil
                            shutil.copy2(original_path, texture_path)
                            self.log(f"    Copied texture: {texture_filename}")
                            return f"./{texture_filename}"
                        else:
                            self.log(f"    Texture extraction disabled: {texture_filename}")
                            return None
                    else:
                        self.log(f"    Warning: External texture file not found: {image.uri}")
                        return None
            
            return None
            
        except Exception as e:
            self.log(f"    Error processing texture {texture_index}: {e}")
            return None
    
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
        
        # デフォルト値を設定
        default_base_color = [1.0, 1.0, 1.0, 1.0]  # 白色
        default_metallic = 0.0
        default_roughness = 0.5
        
        # PBRメタリックラフネス
        if hasattr(material, 'pbrMetallicRoughness') and material.pbrMetallicRoughness:
            pbr = material.pbrMetallicRoughness
            self.log(f"  Processing PBR material: {material_name}")
            
            # ベースカラーテクスチャをチェック
            has_base_color_texture = False
            if hasattr(pbr, 'baseColorTexture') and pbr.baseColorTexture is not None:
                texture_info = pbr.baseColorTexture
                if hasattr(texture_info, 'index') and texture_info.index is not None:
                    # テクスチャがある場合
                    texture_path = self._process_texture(texture_info.index, material_name, 'baseColor')
                    if texture_path:
                        # テクスチャサンプラーを作成
                        sampler = UsdShade.Shader.Define(self.stage, f'{material_path}/baseColorSampler')
                        sampler.CreateIdAttr('UsdUVTexture')
                        sampler.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(texture_path)
                        
                        # テクスチャ座標を設定（通常はst）
                        primvar_reader = UsdShade.Shader.Define(self.stage, f'{material_path}/stReader')
                        primvar_reader.CreateIdAttr('UsdPrimvarReader_float2')
                        primvar_reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
                        
                        # 接続: primvarReader -> sampler -> shader
                        sampler.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(
                            primvar_reader.ConnectableAPI(), 'result')
                        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(
                            sampler.ConnectableAPI(), 'rgb')
                        
                        has_base_color_texture = True
                        self.log(f"    Base color texture: {texture_path}")
            
            # テクスチャがない場合はベースカラーファクターを使用
            if not has_base_color_texture:
                base_color = default_base_color
                if hasattr(pbr, 'baseColorFactor') and pbr.baseColorFactor is not None:
                    if isinstance(pbr.baseColorFactor, (list, tuple)) and len(pbr.baseColorFactor) >= 3:
                        base_color = pbr.baseColorFactor
                        self.log(f"    Base color: {base_color[:3]}")
                    else:
                        self.log(f"    Warning: Invalid baseColorFactor format: {pbr.baseColorFactor}")
                else:
                    self.log(f"    Using default base color: {default_base_color[:3]}")
                
                # 色を設定
                color = base_color[:3]
                shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
                )
            
            # 不透明度（テクスチャがない場合のみ）
            if not has_base_color_texture:
                if len(base_color) > 3:
                    shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(float(base_color[3]))
                else:
                    shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(1.0)
            else:
                # テクスチャがある場合はデフォルトの不透明度
                shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(1.0)
            
            # メタリック
            metallic_value = default_metallic
            if hasattr(pbr, 'metallicFactor') and pbr.metallicFactor is not None:
                metallic_value = float(pbr.metallicFactor)
                self.log(f"    Metallic: {metallic_value}")
            else:
                self.log(f"    Using default metallic: {default_metallic}")
            shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic_value)
            
            # ラフネス
            roughness_value = default_roughness
            if hasattr(pbr, 'roughnessFactor') and pbr.roughnessFactor is not None:
                roughness_value = float(pbr.roughnessFactor)
                self.log(f"    Roughness: {roughness_value}")
            else:
                self.log(f"    Using default roughness: {default_roughness}")
            shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness_value)
        else:
            # PBRセクションがない場合はデフォルト値を設定
            self.log(f"  No PBR section found, using defaults for: {material_name}")
            shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(float(default_base_color[0]), float(default_base_color[1]), float(default_base_color[2]))
            )
            shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(1.0)
            shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(default_metallic)
            shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(default_roughness)
        
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
    parser.add_argument('input', help='Input GLB/GLTF file path or directory for batch conversion')
    parser.add_argument('-o', '--output', help='Output USDC file path (optional, ignored in batch mode)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--batch', action='store_true', help='Batch convert all GLB files in directory')
    parser.add_argument('--extract-textures', action='store_true', help='Extract texture files to disk (default: no extraction)')
    
    args = parser.parse_args()
    
    if args.batch:
        # バッチ変換モード
        input_path = Path(args.input)
        if not input_path.is_dir():
            print(f"Error: {input_path} is not a directory")
            sys.exit(1)
        
        # GLB/GLTFファイルを検索
        glb_files = list(input_path.glob('*.glb')) + list(input_path.glob('*.gltf'))
        if not glb_files:
            print(f"No GLB/GLTF files found in {input_path}")
            sys.exit(1)
        
        print(f"Found {len(glb_files)} files to convert:")
        for f in glb_files:
            print(f"  - {f.name}")
        print()
        
        success_count = 0
        failed_files = []
        
        for glb_file in glb_files:
            print(f"Processing: {glb_file.name}")
            try:
                converter = GLBToUSDCConverter(str(glb_file), verbose=args.verbose, extract_textures=args.extract_textures)
                if converter.convert():
                    success_count += 1
                    if not args.verbose:
                        print(f"  ✅ Converted to {converter.output_path.name}")
                else:
                    failed_files.append(glb_file.name)
                    print(f"  ❌ Failed to convert {glb_file.name}")
            except Exception as e:
                failed_files.append(glb_file.name)
                print(f"  ❌ Error converting {glb_file.name}: {e}")
            print()
        
        # バッチ変換結果のサマリー
        print("="*50)
        print(f"Batch conversion complete!")
        print(f"Successfully converted: {success_count}/{len(glb_files)} files")
        
        if failed_files:
            print(f"Failed files:")
            for failed_file in failed_files:
                print(f"  - {failed_file}")
        
        if success_count < len(glb_files):
            sys.exit(1)
        
    else:
        # 単一ファイル変換モード
        converter = GLBToUSDCConverter(args.input, args.output, args.verbose, extract_textures=args.extract_textures)
        if not converter.convert():
            sys.exit(1)


if __name__ == '__main__':
    main()