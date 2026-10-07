"""
RetailPulse - Installation Verification Script
Run this script to check if all required libraries are installed correctly.
"""

import sys

def check_library(name, import_name=None):
    """Try to import a library and report success/failure."""
    if import_name is None:
        import_name = name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'version not available')
        print(f"  ✅ {name:20s} → {version}")
        return True
    except ImportError as e:
        print(f"  ❌ {name:20s} → NOT INSTALLED ({e})")
        return False

def main():
    print("=" * 60)
    print("  RetailPulse - Installation Check")
    print("=" * 60)
    print(f"\nPython Version: {sys.version}")
    print(f"Python Path: {sys.executable}\n")

    results = []

    print("--- Core Data Science ---")
    results.append(check_library("pandas"))
    results.append(check_library("numpy"))
    results.append(check_library("openpyxl"))

    print("\n--- Visualization ---")
    results.append(check_library("matplotlib"))
    results.append(check_library("seaborn"))
    results.append(check_library("plotly"))

    print("\n--- Machine Learning ---")
    results.append(check_library("scikit-learn", "sklearn"))
    results.append(check_library("xgboost"))

    print("\n--- Time Series ---")
    results.append(check_library("prophet"))

    print("\n--- Deep Learning ---")
    results.append(check_library("torch"))

    print("\n--- Explainability ---")
    results.append(check_library("shap"))

    print("\n--- Dashboard ---")
    results.append(check_library("streamlit"))

    print("\n--- MLOps ---")
    results.append(check_library("mlflow"))
    results.append(check_library("evidently"))

    print("\n--- Utilities ---")
    results.append(check_library("joblib"))
    results.append(check_library("tqdm"))
    results.append(check_library("scipy"))

    # Summary
    total = len(results)
    passed = sum(results)
    failed = total - passed

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed}/{total} libraries installed successfully")

    if failed > 0:
        print(f"  ⚠️  {failed} libraries failed - see above for details")
        print("  Don't worry! Tell your mentor which ones failed.")
        print("  Some libraries (Prophet, PyTorch) can be tricky to install.")
    else:
        print("  🎉 ALL LIBRARIES INSTALLED SUCCESSFULLY!")
        print("  You are ready to start the project!")

    print("=" * 60)

if __name__ == "__main__":
    main()