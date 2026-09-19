import subprocess, sys
steps=[
    ["src/download_dataset.py"],
    ["src/prepare_dataset.py"],
    ["src/analyze_dataset.py"],
    ["src/train_cnn.py","--epochs","20"],
    ["src/evaluate_cnn.py"],
    ["src/compare_opencv_cnn.py"],
]
for args in steps:
    print("\n==>", " ".join(args))
    subprocess.run([sys.executable,*args],check=True)
print("\nPipeline complete. Run: python app.py")
