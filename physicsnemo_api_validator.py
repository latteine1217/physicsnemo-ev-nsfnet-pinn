# Copyright (c) 2025 NVIDIA Corporation. All Rights Reserved.
"""
PhysicsNeMo API 相容性驗證工具
檢查重構後的 EV-NSFnet PINN 專案是否使用正確的 PhysicsNeMo API
"""
import os
import sys
import ast
import importlib
import torch
from typing import Dict, List, Any, Optional


class PhysicsNeMoAPIValidator:
    """PhysicsNeMo API 相容性驗證器"""
    
    def __init__(self):
        self.validation_results = {}
        self.api_errors = []
        self.api_warnings = []
        
    def validate_imports(self, file_path: str) -> Dict[str, Any]:
        """驗證檔案中的 PhysicsNeMo 匯入"""
        
        try:
            with open(file_path, 'r') as f:
                source = f.read()
                
            tree = ast.parse(source, filename=file_path)
            
            imports = []
            physicsnemo_imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                        if 'physicsnemo' in alias.name:
                            physicsnemo_imports.append(alias.name)
                            
                elif isinstance(node, ast.ImportFrom):
                    if node.module and 'physicsnemo' in node.module:
                        for alias in node.names:
                            full_import = f"{node.module}.{alias.name}"
                            physicsnemo_imports.append(full_import)
            
            return {
                "file": file_path,
                "all_imports": imports,
                "physicsnemo_imports": physicsnemo_imports,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "file": file_path,
                "error": str(e),
                "status": "error"
            }
    
    def check_physicsnemo_api_usage(self, file_path: str) -> Dict[str, Any]:
        """檢查 PhysicsNeMo API 使用正確性"""
        
        api_issues = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # 檢查已知的 API 模式
            api_patterns = {
                # 正確的 API 模式
                "correct_patterns": [
                    "from physicsnemo.solver import Solver",
                    "from physicsnemo.models.model import Model", 
                    "from physicsnemo.models.mlp import FullyConnected",
                    "from physicsnemo.datasets import Dataset",
                    "from physicsnemo.utils.loggers import get_logger",
                    "from physicsnemo.distributed import DistributedManager",
                    "from physicsnemo.trainer import Trainer",
                    "from physicsnemo.pdes import PDE",
                    "from physicsnemo.utils.derivatives import gradient"
                ],
                
                # 可能有問題的舊 API 模式
                "problematic_patterns": [
                    "from physicsnemo.models.mlp.fully_connected import FullyConnected",
                    "from physicsnemo.models.layers.activation import Activation",
                    "from physicsnemo.datapipes.benchmarks.dataset import Dataset",
                    "from physicsnemo.launch.logging import",
                    "from physicsnemo.constants import tf_dt",
                    "from physicsnemo.utils.io import ValidateInput"
                ]
            }
            
            # 檢查正確模式
            correct_count = 0
            for pattern in api_patterns["correct_patterns"]:
                if pattern in content:
                    correct_count += 1
            
            # 檢查有問題的模式
            for pattern in api_patterns["problematic_patterns"]:
                if pattern in content:
                    api_issues.append(f"發現可能過時的 API: {pattern}")
            
            # 檢查 FullyConnected 參數使用
            if "FullyConnected(" in content:
                if "input_dim=" in content and "output_dim=" in content:
                    # 正確的參數名稱
                    pass
                elif "in_features=" in content or "out_features=" in content:
                    api_issues.append("FullyConnected 使用了錯誤的參數名稱 (應該是 input_dim, output_dim)")
            
            return {
                "file": file_path,
                "correct_api_count": correct_count,
                "api_issues": api_issues,
                "status": "checked"
            }
            
        except Exception as e:
            return {
                "file": file_path,
                "error": str(e),
                "status": "error"
            }
    
    def validate_class_inheritance(self, file_path: str) -> Dict[str, Any]:
        """驗證類別繼承是否使用正確的 PhysicsNeMo 基類"""
        
        inheritance_issues = []
        
        try:
            with open(file_path, 'r') as f:
                source = f.read()
                
            tree = ast.parse(source, filename=file_path)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            base_name = base.id
                            
                            # 檢查 PhysicsNeMo 相關的基類
                            if base_name == "Module" and "physicsnemo.models.module" in source:
                                inheritance_issues.append(
                                    f"類別 {node.name} 繼承自過時的 Module，應該使用 Solver 或 Model"
                                )
                            elif base_name == "Dataset" and "physicsnemo.datapipes" in source:
                                inheritance_issues.append(
                                    f"類別 {node.name} 使用過時的 Dataset 匯入路徑"
                                )
            
            return {
                "file": file_path,
                "inheritance_issues": inheritance_issues,
                "status": "checked"
            }
            
        except Exception as e:
            return {
                "file": file_path,
                "error": str(e),
                "status": "error"
            }
    
    def validate_all_files(self) -> Dict[str, Any]:
        """驗證所有 Python 檔案"""
        
        python_files = [
            "physicsnemo_solver.py",
            "physicsnemo_net.py",
            "physicsnemo_equations.py", 
            "physicsnemo_data.py",
            "physicsnemo_train.py",
            "physicsnemo_test.py"
        ]
        
        results = {
            "total_files": len(python_files),
            "validated_files": 0,
            "files_with_issues": 0,
            "detailed_results": {}
        }
        
        for file_path in python_files:
            if os.path.exists(file_path):
                # 匯入驗證
                import_result = self.validate_imports(file_path)
                
                # API 使用驗證
                api_result = self.check_physicsnemo_api_usage(file_path)
                
                # 繼承驗證
                inheritance_result = self.validate_class_inheritance(file_path)
                
                # 合併結果
                file_result = {
                    "imports": import_result,
                    "api_usage": api_result,
                    "inheritance": inheritance_result
                }
                
                # 檢查是否有問題
                has_issues = (
                    len(api_result.get("api_issues", [])) > 0 or
                    len(inheritance_result.get("inheritance_issues", [])) > 0 or
                    import_result.get("status") == "error"
                )
                
                if has_issues:
                    results["files_with_issues"] += 1
                
                results["detailed_results"][file_path] = file_result
                results["validated_files"] += 1
            else:
                results["detailed_results"][file_path] = {
                    "error": "檔案不存在",
                    "status": "missing"
                }
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成驗證報告"""
        
        report = []
        report.append("=" * 60)
        report.append("🔍 PhysicsNeMo API 相容性驗證報告")
        report.append("=" * 60)
        
        # 總結
        total = results["total_files"]
        validated = results["validated_files"]
        issues = results["files_with_issues"]
        
        report.append(f"\n📊 總體統計:")
        report.append(f"  - 總檔案數: {total}")
        report.append(f"  - 已驗證: {validated}")
        report.append(f"  - 有問題的檔案: {issues}")
        
        if issues == 0:
            report.append("\n🎉 所有檔案都通過 PhysicsNeMo API 相容性檢查！")
        else:
            report.append(f"\n⚠️  發現 {issues} 個檔案有 API 相容性問題")
        
        # 詳細結果
        report.append("\n" + "=" * 40)
        report.append("📋 詳細驗證結果")
        report.append("=" * 40)
        
        for file_path, file_result in results["detailed_results"].items():
            report.append(f"\n📄 {file_path}:")
            
            if "error" in file_result:
                report.append(f"  ❌ 錯誤: {file_result['error']}")
                continue
            
            # 匯入檢查
            imports = file_result.get("imports", {})
            physicsnemo_imports = imports.get("physicsnemo_imports", [])
            
            if physicsnemo_imports:
                report.append(f"  ✅ PhysicsNeMo 匯入: {len(physicsnemo_imports)} 個")
                for imp in physicsnemo_imports[:3]:  # 只顯示前3個
                    report.append(f"    - {imp}")
                if len(physicsnemo_imports) > 3:
                    report.append(f"    - ... 還有 {len(physicsnemo_imports)-3} 個")
            
            # API 問題
            api_issues = file_result.get("api_usage", {}).get("api_issues", [])
            if api_issues:
                report.append(f"  ⚠️  API 問題:")
                for issue in api_issues:
                    report.append(f"    - {issue}")
            
            # 繼承問題
            inheritance_issues = file_result.get("inheritance", {}).get("inheritance_issues", [])
            if inheritance_issues:
                report.append(f"  ⚠️  繼承問題:")
                for issue in inheritance_issues:
                    report.append(f"    - {issue}")
            
            if not api_issues and not inheritance_issues:
                report.append(f"  ✅ API 使用正確")
        
        return "\n".join(report)


def main():
    """主驗證函數"""
    
    print("🔍 開始 PhysicsNeMo API 相容性驗證...")
    
    # 創建驗證器
    validator = PhysicsNeMoAPIValidator()
    
    # 執行驗證
    results = validator.validate_all_files()
    
    # 生成報告
    report = validator.generate_report(results)
    
    # 顯示報告
    print(report)
    
    # 返回結果
    if results["files_with_issues"] == 0:
        print("\n🎯 結論: PhysicsNeMo API 重構成功！")
        return 0
    else:
        print(f"\n🔧 結論: 需要修正 {results['files_with_issues']} 個檔案的 API 問題")
        return 1


if __name__ == "__main__":
    sys.exit(main())