#!/usr/bin/env python3
"""
既存のGLBファイルのマテリアル情報を診断
"""

import sys
from pathlib import Path
from gltflib import GLTF

def diagnose_glb_materials(glb_path):
    """GLBファイルのマテリアル情報を診断"""
    
    if not Path(glb_path).exists():
        print(f"❌ ファイルが見つかりません: {glb_path}")
        return
    
    try:
        # GLTFをロード
        gltf = GLTF.load(glb_path)
        print(f"=== {glb_path} のマテリアル診断 ===")
        
        if not hasattr(gltf.model, 'materials') or not gltf.model.materials:
            print("❌ マテリアルが見つかりませんでした")
            return
        
        print(f"マテリアル数: {len(gltf.model.materials)}")
        
        for i, material in enumerate(gltf.model.materials):
            print(f"\n--- マテリアル {i} ---")
            print(f"名前: {getattr(material, 'name', 'Unknown')}")
            
            # PBRメタリックラフネスの確認
            if hasattr(material, 'pbrMetallicRoughness') and material.pbrMetallicRoughness:
                pbr = material.pbrMetallicRoughness
                print("✅ PBRセクションあり")
                
                # ベースカラー
                if hasattr(pbr, 'baseColorFactor'):
                    if pbr.baseColorFactor is not None:
                        print(f"  ベースカラー: {pbr.baseColorFactor}")
                        if isinstance(pbr.baseColorFactor, (list, tuple)):
                            if len(pbr.baseColorFactor) >= 3:
                                color = pbr.baseColorFactor[:3]
                                if all(c == 0.5 for c in color):
                                    print("    ⚠️  グレー色 (0.5, 0.5, 0.5)")
                                elif all(c == 1.0 for c in color):
                                    print("    ⚠️  白色 (1.0, 1.0, 1.0)")
                            else:
                                print("    ❌ 色データが不完全")
                        else:
                            print(f"    ❌ 無効な形式: {type(pbr.baseColorFactor)}")
                    else:
                        print("  ❌ baseColorFactor is None")
                else:
                    print("  ❌ baseColorFactorなし")
                
                # メタリック
                if hasattr(pbr, 'metallicFactor'):
                    print(f"  メタリック: {pbr.metallicFactor}")
                else:
                    print("  メタリック: なし")
                
                # ラフネス
                if hasattr(pbr, 'roughnessFactor'):
                    print(f"  ラフネス: {pbr.roughnessFactor}")
                else:
                    print("  ラフネス: なし")
                
                # テクスチャ
                if hasattr(pbr, 'baseColorTexture'):
                    if pbr.baseColorTexture:
                        print("  ✅ ベースカラーテクスチャあり")
                    else:
                        print("  ❌ ベースカラーテクスチャなし")
                else:
                    print("  ❌ ベースカラーテクスチャなし")
                    
            else:
                print("❌ PBRセクションなし")
            
            # エミッシブ
            if hasattr(material, 'emissiveFactor'):
                if material.emissiveFactor:
                    print(f"  エミッシブ: {material.emissiveFactor}")
            
            # 両面レンダリング
            if hasattr(material, 'doubleSided'):
                print(f"  両面レンダリング: {material.doubleSided}")
    
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

def main():
    # 既存のGLBファイルをチェック
    test_files = [
        'problematic_materials.glb',
        'complex_test.glb',
        'material_test.glb'
    ]
    
    # もしくはコマンドライン引数があれば使用
    if len(sys.argv) > 1:
        test_files = sys.argv[1:]
    
    for glb_file in test_files:
        if Path(glb_file).exists():
            diagnose_glb_materials(glb_file)
            print()

if __name__ == '__main__':
    main()