# GLB to USDC Converter

GLBファイルをUSDC形式に完全変換するPythonツールです。メッシュデータ、マテリアル、階層構造を保持したまま変換を行います。

## 特徴

- ✅ **完全なメッシュ変換**: 全ての頂点データ、法線、UV座標を保持
- ✅ **PBRマテリアル保持**: ベースカラー、メタリック、ラフネス、透明度
- ✅ **テクスチャサポート**: 埋め込みテクスチャの抽出と変換（オプション）
- ✅ **階層構造の保持**: GLTFノード構造をUSD階層に正確に変換
- ✅ **バイナリデータの直接処理**: GLBファイルのバイナリチャンクから直接データを読み込み
- ✅ **重複防止**: メッシュの重複や増殖を防ぐ最適化された変換
- ✅ **バッチ変換対応**: ディレクトリ内の全GLBファイルを一括変換
- ✅ **シンプルな操作**: コマンドライン一発で変換完了

## インストール

```bash
pip install -r requirements.txt
```

### システム要件

- Python 3.8以上
- USD Python bindings (usd-core) - `pip install usd-core`で自動インストール
- gltflib - GLBファイルの読み込み
- numpy - 数値計算
- Pillow - 画像処理（将来のテクスチャ対応用）

## 使用方法

### 基本的な使用方法

```bash
# 単一ファイルの変換
python glb_to_usdc.py input.glb

# 出力ファイル名を指定
python glb_to_usdc.py input.glb -o output.usdc

# 詳細ログを表示（推奨）
python glb_to_usdc.py input.glb -v
```

### バッチ変換

ディレクトリ内のすべてのGLB/GLTFファイルを一括変換：

```bash
# ディレクトリ内の全GLBファイルをバッチ変換
python glb_to_usdc.py /path/to/glb/directory --batch

# 詳細ログ付きでバッチ変換
python glb_to_usdc.py /path/to/glb/directory --batch -v
```

### テクスチャ抽出オプション

デフォルトでは、テクスチャファイル（PNG等）も出力されます：

```bash
# デフォルト動作（テクスチャファイル出力あり）
python glb_to_usdc.py chair.glb

# テクスチャファイル出力を無効にする場合
python glb_to_usdc.py chair.glb --no-extract-textures

# バッチ変換でテクスチャ出力を無効化
python glb_to_usdc.py /path/to/glb/directory --batch --no-extract-textures
```

### コマンドラインオプション

- `input`: 入力GLB/GLTFファイルパス、またはバッチ変換用ディレクトリパス（必須）
- `-o, --output`: 出力USDCファイルパス（単一ファイル変換時のみ有効）
- `-v, --verbose`: 詳細なログを表示（変換状況を確認できます）
- `--batch`: バッチ変換モード - ディレクトリ内の全GLB/GLTFファイルを変換
- `--no-extract-textures`: テクスチャファイルの出力を無効化（デフォルト：抽出する）

## 変換の詳細

### サポートされる機能

**ジオメトリ:**
- 頂点位置 (POSITION)
- 法線ベクトル (NORMAL)  
- UV座標 (TEXCOORD_0)
- インデックス付き三角形メッシュ

**マテリアル (PBR Metallic-Roughness):**
- ベースカラー (baseColorFactor)
- ベースカラーテクスチャ (baseColorTexture)
- メタリック値 (metallicFactor)
- ラフネス値 (roughnessFactor)
- 透明度 (alpha)
- 両面レンダリング (doubleSided)
- エミッシブカラー (emissiveFactor)

**テクスチャサポート:**
- 埋め込みテクスチャの抽出（PNG、JPEG）
- UsdUVTextureシェーダーの自動作成
- UV座標マッピング（primvar readers）
- テクスチャ参照の最適化

**階層構造:**
- GLTFノードのトランスフォーム（平行移動、回転、スケール）
- 親子関係の保持
- マテリアル割り当ての維持

### 変換処理の流れ

1. **GLBファイルの解析**: バイナリチャンクから直接データを読み込み
2. **バッファデータの抽出**: 1.5MB以上の大容量メッシュデータも正確に処理
3. **USD階層の構築**: GLTFノード構造をUSD Xformに変換
4. **メッシュ変換**: 頂点、法線、UVデータをUsdGeom.Meshに変換
5. **マテリアル変換**: PBRパラメータをUsdPreviewSurfaceに変換
6. **USDC出力**: 効率的なバイナリUSD形式で保存

## 変換例

### 基本変換例
**入力ファイル:**
- `table.glb` (1.5MB) - 13個のメッシュ、3個のマテリアル、1,704個の頂点

**出力結果:**
- `table.usdc` (56KB) - 完全なメッシュデータとマテリアルを保持
- 重複なし、単一のテーブルモデル

### テクスチャ付きモデル変換例
**入力ファイル:**
- `chair.glb` (47KB) - 16個のメッシュ、16個のマテリアル、木目調テクスチャ

**デフォルト出力:**
- `chair.usdc` (38KB) - テクスチャ参照付きマテリアル
- `texture3material_effect_012_baseColor_0.png` など16個のテクスチャファイル

**テクスチャ出力無効時（`--no-extract-textures`）:**
- `chair.usdc` (36KB) - テクスチャ情報を含むマテリアル、外部テクスチャファイルなし

## トラブルシューティング

### ImportError: USD Python bindings not found

```bash
pip install usd-core
```

### ImportError: gltflib not found

```bash
pip install gltflib
```

### メッシュデータが正しく変換されない

- `-v`オプションを使用して詳細ログを確認
- バイナリデータサイズが正しく読み込まれているか確認

### ファイルサイズが小さすぎる

従来の実装では数KBでしたが、この完全版では元ファイルの3-4%程度のサイズ（圧縮効率の高いUSDC形式）になります。

## ファイル構成

- `glb_to_usdc.py` - メイン変換スクリプト
- `requirements.txt` - 必要な依存パッケージ
- `README.md` - このドキュメント
- `CLAUDE.md` - プロジェクト設定ファイル

## 技術仕様

- GLB 2.0フォーマット完全対応
- バイナリチャンクの直接解析
- numpy配列からUSD Vt配列への効率的な変換
- UsdPreviewSurface準拠のマテリアル変換
- メモリ効率的なバッファキャッシュ

## ライセンス

MITライセンス

## 注意事項

- 複雑なアニメーションやスキニングはサポートしていません
- テクスチャは基本的に外部ファイルとして出力されます（無効化はオプション）
- 大容量ファイル（100MB以上）では処理時間がかかる場合があります
- テクスチャ抽出時は、出力ディレクトリに十分な容量があることを確認してください